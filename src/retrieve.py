"""
retrieve.py — Query-time retrieval for the Oncology RAG pipeline.

Pipeline position:
    extract.py -> chunk.py -> embed.py -> [retrieve.py] -> prompt.py -> answer.py

Given a query, returns the top_k most relevant chunks (with metadata) from
the FAISS + BM25 indexes built by embed.py. Three retrieval modes:

    dense   FAISS dense search only.
    hybrid  dense + BM25, combined by weighted score (normalised to [0, 1]).
    rrf     dense + BM25, combined by Reciprocal Rank Fusion (rank-based,
            no weight to tune). Matches verify.py's Check 4 reference logic.

A max_per_doc cap is applied after fusion so one large document (the NCCN
guideline is most of this corpus) can't fill the whole result list.

Usage:
    from src.retrieve import retrieve
    results = retrieve("first-line treatment for HR+/HER2- metastatic breast cancer")

    python -m src.retrieve --query "..." --mode rrf --top-k 5
"""

from __future__ import annotations

# Must be set before sentence_transformers/torch is imported, or macOS segfaults.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text())

EMB_CFG = cfg["embedding"]
RET_CFG = cfg["retrieval"]
VALID_MODES = ("dense", "hybrid", "rrf")

FAISS_PATH     = _ROOT / cfg["paths"]["faiss_index"]
BM25_PATH      = _ROOT / cfg["paths"]["bm25_index"]
CHUNK_MAP_PATH = FAISS_PATH.parent / "chunk_map.json"
ID_MAP_PATH    = FAISS_PATH.parent / "id_map.json"

TOP_K          = RET_CFG["top_k"]
CANDIDATE_POOL = RET_CFG["candidate_pool"]
RRF_K          = RET_CFG["rrf_k"]
HYBRID_ALPHA   = RET_CFG["hybrid_alpha"]
MAX_PER_DOC    = RET_CFG["max_per_doc"]
DEFAULT_MODE   = RET_CFG["mode"]


# ── Index bundle ──────────────────────────────────────────────────────────
# Loading the embedding model + FAISS + BM25 takes a few seconds, so we do it
# once and cache it — cli.py will call retrieve() once per question asked.

@dataclass
class Indexes:
    chunk_map:    dict[str, dict]   # chunk_id -> full chunk metadata
    faiss_id_map: dict[int, str]    # FAISS row position -> chunk_id
    faiss_index:  faiss.Index
    bm25:         object
    bm25_id_map:  dict[int, str]    # BM25 row position -> chunk_id
    architecture: str = "symmetric"                    # "symmetric" | "asymmetric" — decisions.md D53/D56
    model:           Optional[SentenceTransformer] = None   # symmetric case: one model embeds both sides
    query_model:     Optional[object] = None                # asymmetric case: transformers.AutoModel (query side)
    query_tokenizer: Optional[object] = None                # asymmetric case: matching tokenizer


_indexes_cache: dict[tuple, Indexes] = {}


def load_indexes(embedding_model: Optional[str] = None, index_dir: Optional[Path] = None) -> Indexes:
    """
    Load everything needed for retrieval once per (architecture, model, index_dir)
    combination, then reuse on later calls.

    embedding_model / index_dir let eval/eval_retrieval.py point this at a
    *symmetric* candidate embedding model's own faiss.index + id_map.json (built
    by one of the eval/compare-embedding-models/<model>/build_index.py scripts)
    instead of the live index — see decisions.md D46. Passing embedding_model
    always means "symmetric", since that's the only shape these overrides were
    built for; the live config's own architecture (symmetric or asymmetric) is
    used whenever no override is given. BM25 and chunk_map.json are always read
    from the live index/ dir: neither depends on the embedding model.
    """
    architecture = "symmetric" if embedding_model else EMB_CFG.get("architecture", "symmetric")
    faiss_path = (index_dir / "faiss.index") if index_dir else FAISS_PATH
    id_map_path = (index_dir / "id_map.json") if index_dir else ID_MAP_PATH

    if architecture == "symmetric":
        model_id = embedding_model or EMB_CFG["model"]
    else:
        model_id = EMB_CFG["model"]["query"]   # for cache key / logging only

    cache_key = (architecture, model_id, str(faiss_path))
    if cache_key in _indexes_cache:
        return _indexes_cache[cache_key]

    for path in (faiss_path, BM25_PATH, CHUNK_MAP_PATH, id_map_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.embed` first.")

    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    faiss_id_map = {int(k): v for k, v in json.loads(id_map_path.read_text()).items()}
    faiss_index = faiss.read_index(str(faiss_path))

    with open(BM25_PATH, "rb") as fh:
        bm25_payload = pickle.load(fh)

    if architecture == "symmetric":
        model = SentenceTransformer(model_id, device=EMB_CFG["device"])
        idx = Indexes(
            chunk_map=chunk_map, faiss_id_map=faiss_id_map, faiss_index=faiss_index,
            bm25=bm25_payload["bm25"], bm25_id_map=bm25_payload["id_map"],
            architecture="symmetric", model=model,
        )
    else:
        # Asymmetric dual-encoder (e.g. MedCPT): queries go through the
        # Query-Encoder here; chunks were embedded with the *different*
        # Article-Encoder at index-build time (src/embed.py) — see
        # decisions.md D53/D56. No sentence-transformers support for either
        # half, hence raw transformers + manual [CLS]-token pooling below.
        from transformers import AutoModel, AutoTokenizer
        query_tokenizer = AutoTokenizer.from_pretrained(model_id)
        query_model     = AutoModel.from_pretrained(model_id)
        query_model.eval()
        idx = Indexes(
            chunk_map=chunk_map, faiss_id_map=faiss_id_map, faiss_index=faiss_index,
            bm25=bm25_payload["bm25"], bm25_id_map=bm25_payload["id_map"],
            architecture="asymmetric", query_model=query_model, query_tokenizer=query_tokenizer,
        )

    _indexes_cache[cache_key] = idx
    return idx


# ── Search ────────────────────────────────────────────────────────────────

def dense_search(query: str, idx: Indexes, top_k: int) -> list[tuple[str, float]]:
    """FAISS dense search. Returns (chunk_id, cosine_score) pairs, best first."""
    if idx.architecture == "symmetric":
        query_vector = idx.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    else:
        import torch
        with torch.no_grad():
            encoded = idx.query_tokenizer([query], truncation=True, padding=True, return_tensors="pt", max_length=64)
            embed = idx.query_model(**encoded).last_hidden_state[:, 0, :]   # [CLS] token
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)
            query_vector = embed.numpy()
    scores, positions = idx.faiss_index.search(query_vector.astype(np.float32), k=top_k)

    results = []
    for score, pos in zip(scores[0], positions[0]):
        chunk_id = idx.faiss_id_map.get(int(pos))
        if chunk_id is not None:   # FAISS returns -1 when there are fewer than top_k results
            results.append((chunk_id, float(score)))
    return results


def sparse_search(query: str, idx: Indexes, top_k: int) -> list[tuple[str, float]]:
    """BM25 keyword search. Returns (chunk_id, bm25_score) pairs, best first."""
    scores = idx.bm25.get_scores(query.lower().split())
    top_positions = np.argsort(scores)[::-1][:top_k]
    return [(idx.bm25_id_map[int(p)], float(scores[p])) for p in top_positions]


# ── Fusing dense + sparse results ────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Combine two ranked lists using rank position only — no score tuning needed."""
    fused_scores: dict[str, float] = {}
    for results in (dense_results, sparse_results):
        for rank, (chunk_id, _) in enumerate(results):
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    # Tie-break by chunk_id so tied scores sort the same way on every run — see
    # decisions.md D47 for why this matters (weighted_score_fusion below had a
    # real non-determinism bug from an unordered tie-break; this mirrors the fix
    # here too, even though plain dict/list iteration made this function safe).
    return sorted(fused_scores.items(), key=lambda pair: (-pair[1], pair[0]))


def weighted_score_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    alpha: float = HYBRID_ALPHA,
) -> list[tuple[str, float]]:
    """Combine two ranked lists by score, after scaling each to [0, 1] so they're comparable."""
    def normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        scores = [score for _, score in results]
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return {chunk_id: 1.0 for chunk_id, _ in results}
        return {chunk_id: (score - lo) / (hi - lo) for chunk_id, score in results}

    dense_norm  = normalize(dense_results)
    sparse_norm = normalize(sparse_results)
    # sorted(), not the raw set union: `a | b` on string keys iterates in an order
    # that depends on Python's per-process hash seed (randomised by default), so
    # ties in fused_scores below broke differently across separate runs of the
    # exact same query — a real non-determinism bug found while comparing
    # embedding models (decisions.md D47). Sorting here makes iteration order
    # fixed regardless of hash seed.
    all_chunk_ids = sorted(dense_norm.keys() | sparse_norm.keys())

    fused_scores = {
        chunk_id: alpha * dense_norm.get(chunk_id, 0.0) + (1 - alpha) * sparse_norm.get(chunk_id, 0.0)
        for chunk_id in all_chunk_ids
    }
    # Tie-break by chunk_id (secondary key) so tied fused scores sort identically
    # on every run, not just in the order they happened to land in the dict.
    return sorted(fused_scores.items(), key=lambda pair: (-pair[1], pair[0]))


# ── Diversity cap ─────────────────────────────────────────────────────────

def cap_per_document(
    ranked_results: list[tuple[str, float]],
    chunk_map: dict[str, dict],
    max_per_doc: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Walk the ranked list, keeping at most max_per_doc chunks from any one document."""
    seen_per_doc: dict[str, int] = {}
    kept: list[tuple[str, float]] = []

    for chunk_id, score in ranked_results:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            continue
        doc_id = chunk["doc_id"]
        if seen_per_doc.get(doc_id, 0) >= max_per_doc:
            continue
        seen_per_doc[doc_id] = seen_per_doc.get(doc_id, 0) + 1
        kept.append((chunk_id, score))
        if len(kept) == top_k:
            break

    return kept


# ── Public entry point ────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = TOP_K,
    mode: str = DEFAULT_MODE,
    candidate_pool: int = CANDIDATE_POOL,
    max_per_doc: int = MAX_PER_DOC,
    rrf_k: int = RRF_K,
    hybrid_alpha: float = HYBRID_ALPHA,
    embedding_model: Optional[str] = None,
    index_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    rrf_k and hybrid_alpha are exposed here (not just as module constants)
    so eval/eval_retrieval.py can sweep them per-call without touching
    config.yaml — see decisions.md D40 for why those defaults were never
    validated in the first place.

    embedding_model and index_dir let eval/eval_retrieval.py run the same
    sweep against a candidate embedding model's own index instead of the
    live BGE-base one — see decisions.md D46.

    Each returned chunk is the full chunk_map.json record plus two extra
    keys: "score" (the ranking score — not comparable across modes) and
    "rank" (1-indexed position in the result list).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    idx = load_indexes(embedding_model=embedding_model, index_dir=index_dir)

    if mode == "dense":
        ranked = dense_search(query, idx, top_k=candidate_pool)
    else:
        dense  = dense_search(query, idx, top_k=candidate_pool)
        sparse = sparse_search(query, idx, top_k=candidate_pool)
        ranked = reciprocal_rank_fusion(dense, sparse, k=rrf_k) if mode == "rrf" \
            else weighted_score_fusion(dense, sparse, alpha=hybrid_alpha)

    top_results = cap_per_document(ranked, idx.chunk_map, max_per_doc, top_k)

    output = []
    for rank, (chunk_id, score) in enumerate(top_results, start=1):
        chunk = dict(idx.chunk_map[chunk_id])
        chunk["score"] = score
        chunk["rank"] = rank
        output.append(chunk)
    return output


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Oncology RAG retrieval index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=VALID_MODES, default=DEFAULT_MODE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    results = retrieve(args.query, top_k=args.top_k, mode=args.mode)

    print(f'\nQuery: "{args.query}"   mode={args.mode}   top_k={args.top_k}\n')
    for chunk in results:
        print(f"[{chunk['rank']}] score={chunk['score']:.4f}  doc={chunk['doc_id']}  "
              f"type={chunk['content_type']}")
        print(f"    {chunk.get('section_h1')} > {chunk.get('section_h2')}")
        print(f"    {chunk['chunk_text'][:160]}...\n")


if __name__ == "__main__":
    main()

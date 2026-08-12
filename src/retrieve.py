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
    model:        SentenceTransformer


_indexes: Optional[Indexes] = None


def load_indexes() -> Indexes:
    """Load everything needed for retrieval once, then reuse on later calls."""
    global _indexes
    if _indexes is not None:
        return _indexes

    for path in (FAISS_PATH, BM25_PATH, CHUNK_MAP_PATH, ID_MAP_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m src.embed` first.")

    chunk_map = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
    faiss_id_map = {int(k): v for k, v in json.loads(ID_MAP_PATH.read_text()).items()}
    faiss_index = faiss.read_index(str(FAISS_PATH))

    with open(BM25_PATH, "rb") as fh:
        bm25_payload = pickle.load(fh)

    model = SentenceTransformer(EMB_CFG["model"], device=EMB_CFG["device"])

    _indexes = Indexes(
        chunk_map=chunk_map,
        faiss_id_map=faiss_id_map,
        faiss_index=faiss_index,
        bm25=bm25_payload["bm25"],
        bm25_id_map=bm25_payload["id_map"],
        model=model,
    )
    return _indexes


# ── Search ────────────────────────────────────────────────────────────────

def dense_search(query: str, idx: Indexes, top_k: int) -> list[tuple[str, float]]:
    """FAISS dense search. Returns (chunk_id, cosine_score) pairs, best first."""
    query_vector = idx.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
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
    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


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
    all_chunk_ids = dense_norm.keys() | sparse_norm.keys()

    fused_scores = {
        chunk_id: alpha * dense_norm.get(chunk_id, 0.0) + (1 - alpha) * sparse_norm.get(chunk_id, 0.0)
        for chunk_id in all_chunk_ids
    }
    return sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)


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
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query.

    Each returned chunk is the full chunk_map.json record plus two extra
    keys: "score" (the ranking score — not comparable across modes) and
    "rank" (1-indexed position in the result list).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    idx = load_indexes()

    if mode == "dense":
        ranked = dense_search(query, idx, top_k=candidate_pool)
    else:
        dense  = dense_search(query, idx, top_k=candidate_pool)
        sparse = sparse_search(query, idx, top_k=candidate_pool)
        ranked = reciprocal_rank_fusion(dense, sparse) if mode == "rrf" \
            else weighted_score_fusion(dense, sparse)

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

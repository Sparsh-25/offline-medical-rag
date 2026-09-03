"""
eval_medcpt.py — runs the same mode-comparison + parameter sweep used for the
other 3 embedding-model candidates (BGE-base, PubMedBERT, BGE-M3), but for
MedCPT.

MedCPT can't use src.retrieve.retrieve() directly: that function always
encodes queries with a SentenceTransformer, but MedCPT queries need the
Query-Encoder (a different model from the Article-Encoder used to build
index/faiss.index in this folder) via raw transformers + manual [CLS]-token
pooling — see build_index.py's docstring and decisions.md D46/D53.

What's reused vs. new:
  - sparse_search, reciprocal_rank_fusion, weighted_score_fusion,
    cap_per_document — imported directly from src.retrieve, unchanged. These
    four don't know or care which model produced the dense scores, so there
    is nothing MedCPT-specific about them.
  - medcpt_dense_search / medcpt_retrieve (below) — new, MedCPT-specific,
    mirrors src.retrieve.retrieve()'s exact logic and return shape so it can
    be dropped into eval.eval_retrieval's evaluate_answerable /
    evaluate_unanswerable / run_sweep via their retrieve_fn parameter — same
    recall@k/MRR code as the other 3 models, not a second copy of it.

Usage:
    python "eval/compare-embedding-models/medcpt/eval_medcpt.py" --mode all --by-type
    python "eval/compare-embedding-models/medcpt/eval_medcpt.py" --sweep hybrid_alpha
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from src.retrieve import (
    Indexes, VALID_MODES, RRF_K, HYBRID_ALPHA, CANDIDATE_POOL, MAX_PER_DOC, TOP_K,
    sparse_search, reciprocal_rank_fusion, weighted_score_fusion, cap_per_document,
)
from eval.eval_retrieval import (
    load_gold, evaluate_answerable, evaluate_unanswerable, summarize,
    summarize_by_type, print_summary, run_sweep, SWEEP_VALUES,
)

QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
MAX_QUERY_LENGTH = 64   # per the model card — see build_index.py for the Article-Encoder side

MEDCPT_DIR   = Path(__file__).parent
LIVE_INDEX   = _ROOT / "index"   # BM25 + chunk_map are embedding-model-independent, reused as-is

_idx: Indexes | None = None
_query_model = None
_query_tokenizer = None


def load_medcpt_indexes() -> Indexes:
    """Load MedCPT's own dense index + the live (model-independent) BM25/chunk_map, once."""
    global _idx
    if _idx is not None:
        return _idx

    chunk_map = json.loads((LIVE_INDEX / "chunk_map.json").read_text(encoding="utf-8"))
    faiss_id_map = {int(k): v for k, v in json.loads((MEDCPT_DIR / "index" / "id_map.json").read_text()).items()}
    faiss_index = faiss.read_index(str(MEDCPT_DIR / "index" / "faiss.index"))
    with open(LIVE_INDEX / "bm25.pkl", "rb") as fh:
        bm25_payload = pickle.load(fh)

    _idx = Indexes(
        chunk_map=chunk_map,
        faiss_id_map=faiss_id_map,
        faiss_index=faiss_index,
        bm25=bm25_payload["bm25"],
        bm25_id_map=bm25_payload["id_map"],
        model=None,   # not used — medcpt_dense_search() below has its own model
    )
    return _idx


def load_query_model():
    global _query_model, _query_tokenizer
    if _query_model is None:
        _query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME)
        _query_model.eval()
    return _query_model, _query_tokenizer


def medcpt_dense_search(query: str, idx: Indexes, top_k: int) -> list[tuple[str, float]]:
    """Same contract as src.retrieve.dense_search, but via MedCPT's Query-Encoder."""
    model, tokenizer = load_query_model()
    with torch.no_grad():
        encoded = tokenizer([query], truncation=True, padding=True, return_tensors="pt", max_length=MAX_QUERY_LENGTH)
        embed = model(**encoded).last_hidden_state[:, 0, :]
        embed = torch.nn.functional.normalize(embed, p=2, dim=1).numpy()

    scores, positions = idx.faiss_index.search(embed.astype(np.float32), k=top_k)
    results = []
    for score, pos in zip(scores[0], positions[0]):
        chunk_id = idx.faiss_id_map.get(int(pos))
        if chunk_id is not None:
            results.append((chunk_id, float(score)))
    return results


def medcpt_retrieve(
    query: str,
    top_k: int = TOP_K,
    mode: str = "hybrid",
    candidate_pool: int = CANDIDATE_POOL,
    max_per_doc: int = MAX_PER_DOC,
    rrf_k: int = RRF_K,
    hybrid_alpha: float = HYBRID_ALPHA,
    **_ignored,   # absorbs embedding_model/index_dir — eval_retrieval.py's sweep
                  # always passes these, but MedCPT's index/model are fixed to
                  # this file's constants, so they're accepted and unused here.
) -> list[dict]:
    """Mirrors src.retrieve.retrieve()'s logic and return shape exactly, so it
    can be passed as eval.eval_retrieval's retrieve_fn — see module docstring."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    idx = load_medcpt_indexes()

    if mode == "dense":
        ranked = medcpt_dense_search(query, idx, top_k=candidate_pool)
    else:
        dense = medcpt_dense_search(query, idx, top_k=candidate_pool)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedCPT against eval/gold.jsonl")
    parser.add_argument("--mode", choices=list(VALID_MODES) + ["all"], default="all")
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--sweep", choices=list(SWEEP_VALUES.keys()), default=None)
    args = parser.parse_args()

    gold = load_gold()
    answerable = [e for e in gold if e["answerable"]]
    unanswerable = [e for e in gold if not e["answerable"]]

    if args.sweep:
        run_sweep(args.sweep, answerable, unanswerable, retrieve_fn=medcpt_retrieve)
        return

    modes = list(VALID_MODES) if args.mode == "all" else [args.mode]
    print(f"eval/gold.jsonl: {len(answerable)} answerable, {len(unanswerable)} unanswerable  "
          f"(top_k=10, candidate_pool={CANDIDATE_POOL}, max_per_doc={MAX_PER_DOC}, "
          f"rrf_k={RRF_K}, hybrid_alpha={HYBRID_ALPHA}, embedding_model=ncbi/MedCPT)\n")

    for mode in modes:
        kwargs = dict(mode=mode, candidate_pool=CANDIDATE_POOL, max_per_doc=MAX_PER_DOC,
                      rrf_k=RRF_K, hybrid_alpha=HYBRID_ALPHA)
        per_query = evaluate_answerable(answerable, retrieve_fn=medcpt_retrieve, **kwargs)
        agg = summarize(per_query)
        print(f"[{mode}]")
        print_summary("overall", agg)
        if args.by_type:
            for qtype, sub_agg in summarize_by_type(per_query).items():
                print_summary(f"  {qtype}", sub_agg)

        neg_rows = evaluate_unanswerable(unanswerable, retrieve_fn=medcpt_retrieve, **kwargs)
        scores = [r["top1_score"] for r in neg_rows if r["top1_score"] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  unanswerable top-1 scores: min={min(scores):.3f} "
                  f"max={max(scores):.3f} avg={avg:.3f}  (n={len(scores)})")
        print()


if __name__ == "__main__":
    main()

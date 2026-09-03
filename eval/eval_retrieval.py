"""
eval_retrieval.py — Retrieval quality metrics against eval/gold.jsonl.

Metrics
───────
For answerable queries (99 of 109 gold entries):
  Recall@k (k=1,3,5,10) — fraction of relevant_chunk_ids found in the top-k
    results. For multi-hop entries with more than one relevant_chunk_id,
    this is the *fraction* found, not a binary hit/miss (see decisions.md
    D32 for why — a jointly-needed chunk set is only half-answered if only
    one of two chunks is retrieved).
  MRR — reciprocal rank of the first relevant chunk found, 0 if none of the
    top EVAL_TOP_K results are relevant.

For unanswerable queries (10 of 109 gold entries):
  Top-1 score only. Dense mode's score is cosine similarity (0-1,
  comparable to the existing 0.5 "confident match" threshold from
  verify.py / decisions.md D23). RRF and hybrid scores are NOT on a
  comparable absolute scale (RRF produces small rank-based fractions;
  hybrid is normalised per-query, not globally) — this script reports
  them anyway for reference, but only dense mode's numbers are usable for
  the existing threshold. Full threshold recalibration is a separate,
  not-yet-done task (decisions.md D24).

Usage
─────
    python -m eval.eval_retrieval                        # all 3 modes, current config defaults
    python -m eval.eval_retrieval --mode rrf
    python -m eval.eval_retrieval --rrf-k 30
    python -m eval.eval_retrieval --hybrid-alpha 0.25
    python -m eval.eval_retrieval --candidate-pool 50
    python -m eval.eval_retrieval --max-per-doc 5
    python -m eval.eval_retrieval --by-type               # break down recall/MRR by query_type
    python -m eval.eval_retrieval --sweep rrf_k            # one-at-a-time parameter sweep, see decisions.md D42
    python -m eval.eval_retrieval --sweep hybrid_alpha
    python -m eval.eval_retrieval --sweep candidate_pool
    python -m eval.eval_retrieval --sweep max_per_doc

    # Compare a candidate embedding model instead of the live BGE-base index
    # (index built by one of eval/compare-embedding-models/<model>/build_index.py
    # — see decisions.md D46):
    python -m eval.eval_retrieval --sweep max_per_doc \
        --embedding-model "NeuML/pubmedbert-base-embeddings" \
        --index-dir "eval/compare-embedding-models/pubmedbert/index"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.retrieve import retrieve, VALID_MODES, RRF_K, HYBRID_ALPHA, CANDIDATE_POOL, MAX_PER_DOC, EMB_CFG

GOLD_PATH = Path(__file__).parent / "gold.jsonl"
EVAL_TOP_K = 10  # depth searched for recall@10/MRR — deeper than the default top_k=5
                 # so a "just outside the returned window" near-miss is still visible

# One-at-a-time sweep values and the modes each applies to — see decisions.md
# D42 for why these specific values were chosen, not others.
SWEEP_VALUES = {
    "rrf_k":          {"values": [10, 30, 60, 100],       "modes": ["rrf"]},
    "hybrid_alpha":   {"values": [0.0, 0.25, 0.5, 0.75, 1.0], "modes": ["hybrid"]},
    "candidate_pool": {"values": [10, 20, 50],            "modes": list(VALID_MODES)},
    # 10 stands in for "uncapped" — EVAL_TOP_K is 10, so a cap of 10 or more
    # can never actually bind (no single document could supply more than
    # all of the top 10 results anyway).
    "max_per_doc":    {"values": [1, 2, 5, 10],           "modes": list(VALID_MODES)},
}


def load_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate_answerable(entries: list[dict], retrieve_fn=retrieve, **retrieve_kwargs) -> list[dict]:
    """
    retrieve_fn defaults to src.retrieve.retrieve (BGE-base/PubMedBERT/BGE-M3 all
    use this, via the embedding_model/index_dir overrides). MedCPT's asymmetric
    dual-encoder can't plug into that function's SentenceTransformer-based
    loading, so eval/compare-embedding-models/medcpt/eval_medcpt.py passes its
    own retrieve_fn here instead — same call shape (query, top_k=..., **kwargs),
    so this evaluation loop doesn't need to know or care which embedding
    architecture is behind it. See decisions.md D46/D53.
    """
    per_query = []
    for e in entries:
        results = retrieve_fn(e["query"], top_k=EVAL_TOP_K, **retrieve_kwargs)
        retrieved_ids = [r["chunk_id"] for r in results]
        relevant = set(e["relevant_chunk_ids"])

        reciprocal_rank = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in relevant:
                reciprocal_rank = 1.0 / rank
                break

        row = {"query_id": e["query_id"], "query_type": e["query_type"], "mrr": reciprocal_rank}
        for k in (1, 3, 5, 10):
            found = len(set(retrieved_ids[:k]) & relevant)
            row[f"recall@{k}"] = found / len(relevant) if relevant else 0.0
        per_query.append(row)
    return per_query


def evaluate_unanswerable(entries: list[dict], retrieve_fn=retrieve, **retrieve_kwargs) -> list[dict]:
    rows = []
    for e in entries:
        results = retrieve_fn(e["query"], top_k=1, **retrieve_kwargs)
        rows.append({
            "query_id": e["query_id"],
            "top1_score": results[0]["score"] if results else None,
        })
    return rows


def summarize(per_query: list[dict]) -> dict:
    n = len(per_query)
    if n == 0:
        return {"n": 0}
    agg = {"n": n, "mrr": sum(q["mrr"] for q in per_query) / n}
    for k in (1, 3, 5, 10):
        agg[f"recall@{k}"] = sum(q[f"recall@{k}"] for q in per_query) / n
    return agg


def summarize_by_type(per_query: list[dict]) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in per_query:
        by_type[q["query_type"]].append(q)
    return {t: summarize(qs) for t, qs in sorted(by_type.items())}


def print_summary(label: str, agg: dict) -> None:
    print(f"  {label:12s}  n={agg['n']:3d}  MRR={agg.get('mrr', 0):.3f}  "
          f"R@1={agg.get('recall@1', 0):.3f}  R@3={agg.get('recall@3', 0):.3f}  "
          f"R@5={agg.get('recall@5', 0):.3f}  R@10={agg.get('recall@10', 0):.3f}")


def run_sweep(
    param: str,
    answerable: list[dict],
    unanswerable: list[dict],
    embedding_model: str | None = None,
    index_dir: Path | None = None,
    retrieve_fn=retrieve,
) -> None:
    """One-at-a-time sweep: vary `param`, hold the other three at their defaults."""
    spec = SWEEP_VALUES[param]
    print(f"=== Sweep: {param} (others held at config defaults) ===\n")
    for mode in spec["modes"]:
        print(f"[{mode}]")
        for val in spec["values"]:
            kwargs = dict(mode=mode, candidate_pool=CANDIDATE_POOL, max_per_doc=MAX_PER_DOC,
                          rrf_k=RRF_K, hybrid_alpha=HYBRID_ALPHA,
                          embedding_model=embedding_model, index_dir=index_dir)
            kwargs[param] = val
            agg = summarize(evaluate_answerable(answerable, retrieve_fn=retrieve_fn, **kwargs))
            neg_scores = [r["top1_score"] for r in evaluate_unanswerable(unanswerable, retrieve_fn=retrieve_fn, **kwargs)
                          if r["top1_score"] is not None]
            neg_avg = sum(neg_scores) / len(neg_scores) if neg_scores else float("nan")
            print_summary(f"{param}={val}", agg)
            print(f"               unanswerable top-1 avg={neg_avg:.3f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against eval/gold.jsonl")
    parser.add_argument("--mode", choices=list(VALID_MODES) + ["all"], default="all")
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--hybrid-alpha", type=float, default=HYBRID_ALPHA)
    parser.add_argument("--candidate-pool", type=int, default=CANDIDATE_POOL)
    parser.add_argument("--max-per-doc", type=int, default=MAX_PER_DOC)
    parser.add_argument("--by-type", action="store_true")
    parser.add_argument("--sweep", choices=list(SWEEP_VALUES.keys()), default=None)
    parser.add_argument("--embedding-model", default=None,
                         help="Override embedding.model from config.yaml — used to test a candidate "
                              "embedding model (decisions.md D46) instead of the live BGE-base one.")
    parser.add_argument("--index-dir", type=Path, default=None,
                         help="Directory holding a candidate model's faiss.index + id_map.json, built "
                              "by eval/compare-embedding-models/<model>/build_index.py. BM25 and "
                              "chunk_map.json are always read from the live index/ dir regardless.")
    args = parser.parse_args()

    gold = load_gold()
    answerable = [e for e in gold if e["answerable"]]
    unanswerable = [e for e in gold if not e["answerable"]]

    if args.sweep:
        run_sweep(args.sweep, answerable, unanswerable,
                  embedding_model=args.embedding_model, index_dir=args.index_dir)
        return

    modes = list(VALID_MODES) if args.mode == "all" else [args.mode]

    # EMB_CFG["model"] is a plain string for "symmetric" architecture, or a
    # {"query": ..., "article": ...} dict for "asymmetric" (e.g. MedCPT) — see
    # decisions.md D53/D56. --embedding-model always implies symmetric.
    model_label = args.embedding_model or EMB_CFG["model"]
    print(f"eval/gold.jsonl: {len(answerable)} answerable, {len(unanswerable)} unanswerable  "
          f"(top_k={EVAL_TOP_K}, candidate_pool={args.candidate_pool}, max_per_doc={args.max_per_doc}, "
          f"rrf_k={args.rrf_k}, hybrid_alpha={args.hybrid_alpha}, "
          f"embedding_model={model_label})\n")

    for mode in modes:
        kwargs = dict(mode=mode, candidate_pool=args.candidate_pool, max_per_doc=args.max_per_doc,
                      rrf_k=args.rrf_k, hybrid_alpha=args.hybrid_alpha,
                      embedding_model=args.embedding_model, index_dir=args.index_dir)
        per_query = evaluate_answerable(answerable, **kwargs)
        agg = summarize(per_query)
        print(f"[{mode}]")
        print_summary("overall", agg)
        if args.by_type:
            for qtype, sub_agg in summarize_by_type(per_query).items():
                print_summary(f"  {qtype}", sub_agg)

        neg_rows = evaluate_unanswerable(unanswerable, **kwargs)
        scores = [r["top1_score"] for r in neg_rows if r["top1_score"] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  unanswerable top-1 scores: min={min(scores):.3f} "
                  f"max={max(scores):.3f} avg={avg:.3f}  (n={len(scores)})")
        print()


if __name__ == "__main__":
    main()

"""
compare_rerankers.py — compare candidate rerankers against the no-rerank baseline.

The question: does reranking the retrieval candidate pool put the right chunk in the
top-5 more often than plain hybrid retrieval does? (Top-5 is what the LLM sees.)

For each answerable gold query we:
  1. retrieve the candidate pool (hybrid, no per-doc cap so the reranker sees them all),
  2. reorder it with the reranker,
  3. re-apply the production per-doc cap and keep the top-5,
  4. score recall@1/3/5 and MRR against gold — the same definitions as eval_retrieval.py.

The no-rerank baseline runs the same steps but skips step 2, so any difference is the
reranker's doing. Run one reranker per invocation (they are large, separate downloads);
each run prints baseline vs. that reranker and appends its numbers to rerank_results.json.

    python -m eval.compare_rerankers --reranker medcpt
    python -m eval.compare_rerankers --reranker bge-v2-m3
    python -m eval.compare_rerankers --summary          # reprint everything saved so far
    python -m eval.compare_rerankers --reranker medcpt --limit 10   # quick smoke test

Runs on the box or the Mac (rerankers are small enough for CPU; the box is just faster).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.retrieve import retrieve, CANDIDATE_POOL, MAX_PER_DOC
from src.rerank import load_reranker, rerank, RERANKERS

GOLD_PATH = Path(__file__).parent / "gold.jsonl"
RESULTS_PATH = Path(__file__).parent / "rerank_results.json"

FINAL_K = 5          # how many chunks reach the LLM
NO_CAP = 999         # a per-doc cap this high never binds — used to get the full pool


def load_answerable() -> list[dict]:
    """Gold entries we can score retrieval on (negatives have no relevant chunk)."""
    gold = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [e for e in gold if e["answerable"]]


def top_k_capped(candidates: list[dict], k: int, max_per_doc: int) -> list[dict]:
    """Keep candidates in order, allowing at most max_per_doc per document, up to k."""
    kept, per_doc = [], defaultdict(int)
    for c in candidates:
        if per_doc[c["doc_id"]] >= max_per_doc:
            continue
        per_doc[c["doc_id"]] += 1
        kept.append(c)
        if len(kept) == k:
            break
    return kept


def score_query(retrieved_ids: list[str], relevant_ids: list[str]) -> dict:
    """recall@1/3/5 and MRR for one query — same definitions as eval_retrieval.py."""
    relevant = set(relevant_ids)
    reciprocal_rank = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            reciprocal_rank = 1.0 / rank
            break
    row = {"mrr": reciprocal_rank}
    for k in (1, 3, 5):
        found = len(set(retrieved_ids[:k]) & relevant)
        row[f"recall@{k}"] = found / len(relevant) if relevant else 0.0
    return row


def build_pools(entries: list[dict]) -> list[tuple[dict, list[dict]]]:
    """Retrieve the candidate pool once per query (reused for baseline and every reranker)."""
    pools = []
    for e in entries:
        pool = retrieve(e["query"], top_k=CANDIDATE_POOL, max_per_doc=NO_CAP)
        pools.append((e, pool))
    return pools


def evaluate(pools: list[tuple[dict, list[dict]]], order_fn) -> list[dict]:
    """Score every query after reordering its pool with order_fn(query, pool)."""
    rows = []
    for entry, pool in pools:
        ordered = order_fn(entry["query"], pool)
        top = top_k_capped(ordered, k=FINAL_K, max_per_doc=MAX_PER_DOC)
        row = score_query([c["chunk_id"] for c in top], entry["relevant_chunk_ids"])
        row["query_type"] = entry["query_type"]
        rows.append(row)
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Average the per-query scores."""
    n = len(rows)
    agg = {"n": n, "mrr": sum(r["mrr"] for r in rows) / n}
    for k in (1, 3, 5):
        agg[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in rows) / n
    return agg


def by_type(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["query_type"]].append(r)
    return {t: aggregate(rs) for t, rs in sorted(groups.items())}


def print_row(label: str, agg: dict) -> None:
    print(f"  {label:16s} n={agg['n']:3d}  MRR={agg['mrr']:.3f}  "
          f"R@1={agg['recall@1']:.3f}  R@3={agg['recall@3']:.3f}  R@5={agg['recall@5']:.3f}")


def save_result(name: str, agg: dict) -> None:
    """Store one reranker's overall metrics in rerank_results.json (keyed by name)."""
    saved = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    saved[name] = agg
    RESULTS_PATH.write_text(json.dumps(saved, indent=2))


def print_summary() -> None:
    if not RESULTS_PATH.exists():
        print("No results saved yet. Run with --reranker <name> first.")
        return
    saved = json.loads(RESULTS_PATH.read_text())
    print("\n=== reranker comparison (saved results) ===")
    # baseline first if present, then the rest
    for name in (["baseline"] + [n for n in saved if n != "baseline"]):
        if name in saved:
            print_row(name, saved[name])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rerankers against the no-rerank baseline.")
    parser.add_argument("--reranker", choices=list(RERANKERS), help="Which reranker to test")
    parser.add_argument("--summary", action="store_true", help="Reprint all saved results and exit")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N queries (smoke test)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return
    if not args.reranker:
        parser.error("give --reranker <name> (or --summary)")

    entries = load_answerable()
    if args.limit:
        entries = entries[:args.limit]

    print(f"Retrieving candidate pools for {len(entries)} answerable queries ...")
    pools = build_pools(entries)

    # Baseline: pool order as-is (no reranking).
    baseline_rows = evaluate(pools, order_fn=lambda query, pool: pool)
    baseline_agg = aggregate(baseline_rows)
    save_result("baseline", baseline_agg)

    # The reranker under test.
    scorer = load_reranker(args.reranker)
    rerank_rows = evaluate(pools, order_fn=lambda query, pool: rerank(query, pool, scorer))
    rerank_agg = aggregate(rerank_rows)
    save_result(args.reranker, rerank_agg)

    print("\n=== overall (top-5) ===")
    print_row("baseline", baseline_agg)
    print_row(args.reranker, rerank_agg)

    print("\n=== by query type (R@5) — baseline vs reranker ===")
    base_by_type, rr_by_type = by_type(baseline_rows), by_type(rerank_rows)
    for t in base_by_type:
        b, r = base_by_type[t]["recall@5"], rr_by_type[t]["recall@5"]
        print(f"  {t:20s} {b:.3f} -> {r:.3f}")

    print(f"\nSaved -> {RESULTS_PATH}   (run --summary to compare all rerankers)")


if __name__ == "__main__":
    main()

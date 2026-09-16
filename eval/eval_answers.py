"""
eval_answers.py — End-to-end answer-generation eval against eval/gold.jsonl.

Runs each gold query through the REAL pipeline (src.answer.answer): live hybrid
retrieval on the box index -> prompt v1.2 -> Qwen3-32B. This tests the whole
Stage-8 path, unlike answer_harness.py which used pre-baked passages to compare
candidate models.

Two kinds of judging, kept honest about what a machine can and cannot grade:

  Automatic (objective) — printed as metrics:
    - Correct abstention on the negatives (answerable=False): did the model output
      the exact NO_EVIDENCE sentence?
    - Over-abstention on the answerables: did it wrongly refuse when it shouldn't?
    - Retrieval support: for each answerable, was a gold chunk actually among the
      sources the model saw? This separates retrieval misses from generation misses
      — if the right chunk never reached the model, no faithful answer is possible.

  Manual (needs reading) — faithfulness and answer correctness. Offline we have no
    trustworthy judge model, so we do NOT auto-grade these (same reason the Stage-8
    bake-off used a hand rubric). Use --show to read each answer next to its gold
    notes and sources, and score by hand.

This runs on the GPU box (it needs llama-cpp-python + the GGUF model, via src.answer).

Usage (from the project root):
    python -m eval.eval_answers                 # generate for all gold queries, then print metrics
    python -m eval.eval_answers --metrics-only   # just recompute metrics from the saved answers
    python -m eval.eval_answers --show           # print each answer + sources + gold, for manual scoring
    python -m eval.eval_answers --limit 5        # generate only the first 5 (quick smoke test)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.answer import answer, NO_EVIDENCE

GOLD_PATH = Path(__file__).parent / "gold.jsonl"
OUT_PATH = Path(__file__).parent / "answers_gold.jsonl"


def load_gold():
    """Read gold.jsonl into a list of entries."""
    return [json.loads(line) for line in GOLD_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_done():
    """Read any answers we've already generated, keyed by query_id (for resuming)."""
    if not OUT_PATH.exists():
        return {}
    done = {}
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["query_id"]] = row
    return done


def is_abstention(text: str) -> bool:
    """True if the answer is the fixed refusal sentence (the whole answer, per prompt v1.2)."""
    return text.strip() == NO_EVIDENCE


def evaluate_one(entry: dict) -> dict:
    """Run one gold query through the pipeline and record what we can judge automatically."""
    result = answer(entry["query"])
    retrieved_ids = [s["chunk_id"] for s in result["sources"]]
    gold_ids = entry.get("relevant_chunk_ids", [])

    # For answerable queries: did retrieval surface at least one gold chunk? (null for negatives)
    retrieval_hit = None
    if entry["answerable"]:
        retrieval_hit = any(cid in retrieved_ids for cid in gold_ids)

    return {
        "query_id":        entry["query_id"],
        "query_type":      entry["query_type"],
        "answerable":      entry["answerable"],
        "query":           entry["query"],
        "gold_notes":      entry.get("notes", ""),
        "gold_chunk_ids":  gold_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "retrieval_hit":   retrieval_hit,
        "abstained":       is_abstention(result["answer"]),
        "answer":          result["answer"],
    }


def run_generation(limit: int | None):
    """Generate answers for every gold query not already done; append each as it finishes."""
    gold = load_gold()
    done = load_done()
    todo = [e for e in gold if e["query_id"] not in done]
    if limit:
        todo = todo[:limit]

    print(f"gold: {len(gold)} queries | already done: {len(done)} | to generate now: {len(todo)}")

    # Append mode so progress survives a crash or a dropped session (32B is slow).
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, entry in enumerate(todo, start=1):
            print(f"[{i}/{len(todo)}] {entry['query_id']} ({entry['query_type']}) ...", flush=True)
            row = evaluate_one(entry)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    print(f"\nSaved -> {OUT_PATH}")


def print_metrics():
    """Print the objective metrics from the saved answers."""
    rows = load_done()
    if not rows:
        print("No answers saved yet. Run without --metrics-only first.")
        return
    rows = list(rows.values())

    answerable = [r for r in rows if r["answerable"]]
    negatives  = [r for r in rows if not r["answerable"]]

    print(f"\n=== answer-generation metrics ({len(rows)} queries) ===\n")

    # 1. Abstention on the negatives — they SHOULD refuse.
    if negatives:
        correct = sum(1 for r in negatives if r["abstained"])
        print(f"negatives (should abstain): {correct}/{len(negatives)} correctly abstained")
        for r in negatives:
            if not r["abstained"]:
                print(f"    MISSED abstention: {r['query_id']} -> {r['answer'][:70]}...")

    # 2. Over-abstention on the answerables — they should NOT refuse.
    if answerable:
        over = [r for r in answerable if r["abstained"]]
        print(f"\nanswerable (should answer): {len(answerable) - len(over)}/{len(answerable)} answered, "
              f"{len(over)} wrongly abstained")
        for r in over:
            hint = "gold chunk WAS retrieved" if r["retrieval_hit"] else "gold chunk was NOT retrieved"
            print(f"    over-abstained: {r['query_id']} ({hint})")

    # 3. Retrieval support — could the model even have answered? (gold chunk in its sources)
    if answerable:
        hit = sum(1 for r in answerable if r["retrieval_hit"])
        print(f"\nretrieval support: {hit}/{len(answerable)} answerable queries had a gold chunk in "
              f"the {len(answerable[0]['retrieved_chunk_ids'])} sources the model saw")
        by_type = defaultdict(lambda: [0, 0])   # type -> [hits, total]
        for r in answerable:
            by_type[r["query_type"]][1] += 1
            if r["retrieval_hit"]:
                by_type[r["query_type"]][0] += 1
        for t in sorted(by_type):
            h, n = by_type[t]
            print(f"    {t:20s} {h}/{n}")

    print("\nFaithfulness & correctness are not auto-graded — use --show to score them by reading.")


def show_for_reading():
    """Print each saved answer next to its sources and gold notes, with a scoring template."""
    rows = load_done()
    if not rows:
        print("No answers saved yet. Run without --show first.")
        return

    for r in rows.values():
        print("\n" + "=" * 80)
        print(f"{r['query_id']}  ({r['query_type']}, answerable={r['answerable']})")
        print(f"Q: {r['query']}")
        print(f"GOLD notes: {r['gold_notes']}")
        print(f"GOLD chunk(s): {r['gold_chunk_ids']}   retrieval_hit: {r['retrieval_hit']}")
        print("-" * 80)
        print(r["answer"])
        print("-" * 80)
        print(f"Sources the model saw: {r['retrieved_chunk_ids']}")
        print("SCORE  faithfulness (0/1/2): ___   correctness (0/1/2): ___   abstention ok (y/n/NA): ___")


def main():
    parser = argparse.ArgumentParser(description="End-to-end answer eval against gold.jsonl.")
    parser.add_argument("--metrics-only", action="store_true", help="Recompute metrics from saved answers, no generation")
    parser.add_argument("--show", action="store_true", help="Print each answer + sources + gold for manual scoring")
    parser.add_argument("--limit", type=int, default=None, help="Generate only the first N (quick smoke test)")
    parser.add_argument("--out", default="answers_gold.jsonl",
                        help="Output filename in eval/ (use a different one, e.g. answers_gold_reranked.jsonl, "
                             "to keep the baseline and reranked answers separate for comparison)")
    args = parser.parse_args()

    # Point every mode (generate / metrics / show) at the chosen output file.
    global OUT_PATH
    OUT_PATH = Path(__file__).parent / args.out

    if args.show:
        show_for_reading()
    elif args.metrics_only:
        print_metrics()
    else:
        run_generation(args.limit)
        print_metrics()


if __name__ == "__main__":
    main()

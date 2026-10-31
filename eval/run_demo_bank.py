"""
run_demo_bank.py — generate the offline demo answer bank + per-stage latency numbers.

Run this on the GPU box while access lasts. For each question in demo_questions.jsonl it
runs the real pipeline, records the answer, the sources (with the passage text), and the
timing, and writes it to demo_bank.jsonl. The point:

  1. a pre-baked bank the offline UI can serve with NO GPU (the LLM lives only on the box);
  2. end-to-end latency numbers for the report that can only be measured with the GPU up.

Timing trick (no changes to answer.py): we time retrieve_top_k() on its own to get the
retrieval+rerank+safety-net cost, then time the full answer() call; generation is the
difference. answer() re-runs retrieval internally, so this double-retrieves (a few ms) but
stays faithful to the real pipeline instead of duplicating its logic here.

It is resumable: already-answered ids in the output are skipped, so a dropped session or a
slow 32B run can be continued.

    python -m eval.run_demo_bank                          # all questions
    python -m eval.run_demo_bank --limit 3                # quick smoke test
    python -m eval.run_demo_bank --metrics-only           # recompute summary from the bank
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import yaml

from src.answer import answer, retrieve_top_k, NO_EVIDENCE
from src.retrieve import TOP_K

_ROOT = Path(__file__).resolve().parent.parent
_CFG = yaml.safe_load((_ROOT / "config.yaml").read_text())
QUESTIONS_PATH = Path(__file__).parent / "demo_questions.jsonl"
BANK_PATH = Path(__file__).parent / "demo_bank.jsonl"

_CHUNK_TEXTS: dict[str, str] = {}


def _load_chunk_texts() -> None:
    path = _ROOT / _CFG["paths"]["chunks"]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            _CHUNK_TEXTS[row["chunk_id"]] = row.get("chunk_text", "")


def _load_done() -> dict[str, dict]:
    if not BANK_PATH.exists():
        return {}
    done = {}
    for line in BANK_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            done[row["id"]] = row
    return done


def run_one(entry: dict) -> dict:
    """Run one question, timing retrieval and generation separately."""
    query = entry["question"]

    t0 = time.time()
    retrieve_top_k(query, TOP_K)              # retrieval + rerank + safety-net, timed alone
    t_retrieval = time.time() - t0

    t0 = time.time()
    result = answer(query)                    # full pipeline (retrieval again + generation)
    t_total = time.time() - t0
    t_generation = max(t_total - t_retrieval, 0.0)

    for source in result["sources"]:
        source["snippet"] = _CHUNK_TEXTS.get(source["chunk_id"], "")[:600]

    return {
        "id":        entry["id"],
        "question":  query,
        "expected":  entry["expected"],
        "category":  entry["category"],
        "origin":    entry["origin"],
        "answer":    result["answer"],
        "abstained": result["answer"].strip() == NO_EVIDENCE,
        "sources":   result["sources"],
        "timing":    {"retrieval_s": round(t_retrieval, 2),
                      "generation_s": round(t_generation, 2),
                      "total_s": round(t_total, 2)},
    }


def generate(limit: int | None) -> None:
    questions = [json.loads(l) for l in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = _load_done()
    todo = [q for q in questions if q["id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"questions: {len(questions)} | done: {len(done)} | to run now: {len(todo)}")

    if todo:
        print("Warming models (first call loads Qwen + reranker) ...", flush=True)
        answer("warmup")   # load everything once so the first timed run isn't a cold load

    with BANK_PATH.open("a", encoding="utf-8") as f:
        for i, entry in enumerate(todo, start=1):
            print(f"[{i}/{len(todo)}] {entry['id']} ({entry['category']}) ...", flush=True)
            row = run_one(entry)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    print(f"\nSaved -> {BANK_PATH}")


def print_metrics() -> None:
    rows = list(_load_done().values())
    if not rows:
        print("No bank yet. Run without --metrics-only first.")
        return

    print(f"\n=== demo bank ({len(rows)} questions) ===\n")

    # Behaviour: did expected-answer answer, and expected-abstain abstain?
    for expected in ("answer", "abstain"):
        group = [r for r in rows if r["expected"] == expected]
        if expected == "answer":
            ok = sum(1 for r in group if not r["abstained"])
            print(f"expected ANSWER : {ok}/{len(group)} answered")
            for r in group:
                if r["abstained"]:
                    print(f"    wrongly abstained: {r['id']} ({r['category']}) — {r['question'][:70]}")
        else:
            ok = sum(1 for r in group if r["abstained"])
            print(f"expected ABSTAIN: {ok}/{len(group)} abstained")
            for r in group:
                if not r["abstained"]:
                    print(f"    wrongly answered: {r['id']} ({r['category']}) — {r['question'][:70]}")
            # break abstention accuracy down by abstain sub-type
            subs = sorted({r["category"] for r in group})
            for s in subs:
                g = [r for r in group if r["category"] == s]
                print(f"      {s:12s} {sum(1 for r in g if r['abstained'])}/{len(g)}")

    # Latency (report numbers) — generation dominates; measured only with the GPU up.
    def summarize(key: str) -> str:
        vals = sorted(r["timing"][key] for r in rows)
        p90 = vals[min(len(vals) - 1, int(0.9 * len(vals)))]
        return f"median {statistics.median(vals):.1f}s | p90 {p90:.1f}s | max {vals[-1]:.1f}s"

    print("\nlatency per stage:")
    print(f"    retrieval+rerank+safety-net : {summarize('retrieval_s')}")
    print(f"    generation (Qwen3-32B)      : {summarize('generation_s')}")
    print(f"    total end-to-end            : {summarize('total_s')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the demo answer bank + latency numbers.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N (smoke test)")
    parser.add_argument("--metrics-only", action="store_true", help="Recompute the summary from the saved bank")
    args = parser.parse_args()

    if args.metrics_only:
        print_metrics()
        return

    _load_chunk_texts()
    generate(args.limit)
    print_metrics()


if __name__ == "__main__":
    main()

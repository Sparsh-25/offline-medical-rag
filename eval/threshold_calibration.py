"""
threshold_calibration.py — graduated query-tier experiment to test whether ANY
single cosine-similarity threshold can separate "genuinely relevant" from
"irrelevant" content on the live corpus/embedding model.

Why dense mode only, not hybrid or rrf:
    hybrid mode's top-1 score is provably uninformative as a confidence
    signal — min-max normalization guarantees the top candidate scores ~1.0
    for any query, regardless of relevance (decisions.md D43 finding 2). RRF's
    score is a rank-based fraction, not a similarity at all. Only dense mode's
    score is a real cosine similarity: vectors are L2-normalised at both
    index-build and query time, and FAISS IndexFlatIP's inner product of two
    unit vectors is mathematically identical to cosine similarity (confirmed
    in code, decisions.md D23). That's the only score meaningful to compare
    against a threshold.

Background:
    decisions.md D41 already found that no single cosine threshold could
    separate relevant from irrelevant on 10 adversarial near-miss negatives,
    using BGE-base. This script instead uses a graduated 4-tier query set
    (unrelated -> surface-level oncology -> in-depth non-breast-cancer
    oncology -> breast cancer) run against the CURRENT production model
    (MedCPT, decisions.md D56) — a different embedding space, so this is a
    genuine re-test on the model actually in production, not a rerun of D41.
    See decisions.md D57 for the full write-up and interpretation.

Usage:
    python -m eval.threshold_calibration
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.retrieve import retrieve

GOLD_PATH = Path(__file__).parent / "gold.jsonl"

TIER_1_UNRELATED = [
    "What's the best way to brew pour-over coffee?",
    "How does a car's manual transmission work?",
    "What are the basic rules of cricket?",
    "How do I train a puppy not to bite furniture?",
    "What causes the aurora borealis?",
]

TIER_2_SURFACE_ONCOLOGY = [
    "What is cancer?",
    "What is chemotherapy?",
    "Why do doctors recommend cancer screening tests?",
    "What does it mean for a tumor to be malignant vs. benign?",
    "What is a biopsy?",
]

TIER_3_DEEP_NON_BREAST_ONCOLOGY = [
    "What is the standard first-line treatment for metastatic EGFR-mutant non-small cell lung cancer?",
    "Explain the R-CHOP chemotherapy regimen for diffuse large B-cell lymphoma.",
    "What are the FIGO staging criteria for ovarian cancer?",
    "Describe the mechanism of CAR-T cell therapy in relapsed acute lymphoblastic leukemia.",
    "What is the Gleason scoring system for prostate cancer biopsies?",
]

N_TIER_4_SAMPLES = 10  # sampled from eval/gold.jsonl's answerable entries


def load_tier_4_queries() -> list[str]:
    """
    Sample real, already-verified answerable queries from the gold set for
    Tier 4 — these have confirmed supporting chunks in the corpus (built and
    reviewed earlier this project, see eval/gold.jsonl), so there's no new
    query-design risk in reusing them here.
    """
    with open(GOLD_PATH, encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh if line.strip()]
    answerable = [e["query"] for e in gold if e["answerable"]]

    # Evenly spaced sample across the file rather than the first N, so the
    # sample isn't biased toward whichever document's queries happen to be
    # listed first.
    step = max(1, len(answerable) // N_TIER_4_SAMPLES)
    return answerable[::step][:N_TIER_4_SAMPLES]


def run_tier(label: str, queries: list[str]) -> list[float]:
    """
    Prints score AND the actual retrieved chunk (doc_id + a text snippet) for
    every query, not just the score — a score alone can't tell you whether a
    "high" number means a genuine match or a coincidentally-similar but
    irrelevant chunk (e.g. an author-affiliations list). See decisions.md D57
    for why this distinction mattered here.
    """
    scores = []
    print(f"\n=== {label} (n={len(queries)}) ===")
    for q in queries:
        results = retrieve(q, top_k=1, mode="dense")
        r = results[0] if results else None
        score = r["score"] if r else 0.0
        scores.append(score)
        print(f"  score={score:.3f}  Q: {q}")
        if r:
            print(f"    -> doc={r['doc_id']}  section={r.get('section_h1')} > {r.get('section_h2')}")
            print(f"    -> {r['chunk_text'][:160].strip()}...")
    return scores


def summarize(label: str, scores: list[float]) -> None:
    print(f"{label:40s}  n={len(scores):2d}  "
          f"min={min(scores):.3f}  max={max(scores):.3f}  "
          f"mean={statistics.mean(scores):.3f}  median={statistics.median(scores):.3f}")


def main() -> None:
    tier_4_queries = load_tier_4_queries()

    tiers = [
        ("Tier 1 - unrelated to oncology",          TIER_1_UNRELATED),
        ("Tier 2 - oncology, surface level",        TIER_2_SURFACE_ONCOLOGY),
        ("Tier 3 - oncology, in-depth, non-breast", TIER_3_DEEP_NON_BREAST_ONCOLOGY),
        ("Tier 4 - breast cancer (in-corpus)",      tier_4_queries),
    ]

    all_scores: dict[str, list[float]] = {}
    for label, queries in tiers:
        all_scores[label] = run_tier(label, queries)

    print("\n=== Summary (dense-mode cosine similarity, top-1 score) ===")
    for label, scores in all_scores.items():
        summarize(label, scores)


if __name__ == "__main__":
    main()

"""
build_harness_inputs.py — Prepare the input file for the Stage 8 LLM comparison.

For a small, hand-picked set of gold-set queries, this runs the normal retrieval
step (retrieve.py — MedCPT + BM25 hybrid) and saves each query together with the
passages that were retrieved for it.

The result, eval/harness_inputs.json, is the FIXED input that every candidate LLM
will be tested on. Retrieval is done once, here, so that the model comparison only
changes the model — not what each model was given to read.

Run this once on the machine that has the index (the laptop), then upload the
output JSON to the GPU box.

    python -m eval.build_harness_inputs
"""

import json
from pathlib import Path

from src.retrieve import retrieve

# The 10 queries we test the models on: 7 answerable (spread across different
# documents and question types, to test faithful grounding) + 3 unanswerable
# (to test whether the model correctly abstains instead of making something up).
CHOSEN_QUERY_IDS = [
    # ── answerable ──
    "q001",  # factoid_numeric — MONALEESA-2 (ribociclib PFS hazard ratio)
    "q019",  # factoid_numeric — KEYNOTE-522 (pembrolizumab pCR rates)
    "q067",  # factoid_numeric — NCCN genetic (BRCA1 breast-cancer risk by age 70)
    "q010",  # comparative     — ASCO HER2 testing (ADC eligibility)
    "q028",  # comparative     — CDK4/6i meta-analysis (severe adverse-event comparison)
    "q015",  # definitional    — OlympiA (synthetic lethality)
    "q051",  # definitional    — multi-omics review (what multiomics is)
    # ── unanswerable (the corpus does not contain the specific answer) ──
    "q073",  # ribociclib median overall survival — not reported
    "q076",  # olaparib FDA-approved starting dose — not in the corpus
    "q078",  # 5-year overall survival for T-DXd — not reported
]

HOW_MANY_PASSAGES = 5   # number of passages to retrieve per query (top_k)

GOLD_PATH = Path("eval/gold.jsonl")
OUTPUT_PATH = Path("eval/harness_inputs.json")


def load_gold_by_id():
    """Read gold.jsonl and return a dict: query_id -> gold entry."""
    gold = {}
    for line in GOLD_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            gold[entry["query_id"]] = entry
    return gold


def section_label(passage):
    """Pick the most specific section heading available, for the SOURCE header."""
    return passage.get("section_h2") or passage.get("section_h1") or "—"


def main():
    gold = load_gold_by_id()
    harness_inputs = []

    for query_id in CHOSEN_QUERY_IDS:
        entry = gold[query_id]
        query = entry["query"]
        print(f"Retrieving for {query_id} ({entry['query_type']}): {query[:70]}...")

        results = retrieve(query, top_k=HOW_MANY_PASSAGES)

        # Keep only the fields the prompt actually needs — smaller, clearer file.
        passages = []
        for p in results:
            passages.append({
                "chunk_id":       p["chunk_id"],
                "doc_id":         p["doc_id"],
                "source_type":    p.get("source_type"),
                "recency_weight": p.get("recency_weight"),
                "title":          p.get("title"),
                "pub_year":       p.get("pub_year"),
                "section":        section_label(p),
                "text":           p["chunk_text"],
            })

        harness_inputs.append({
            "query_id":   query_id,
            "query":      query,
            "query_type": entry["query_type"],
            "answerable": entry["answerable"],
            "passages":   passages,
        })

    OUTPUT_PATH.write_text(
        json.dumps(harness_inputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(harness_inputs)} queries -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

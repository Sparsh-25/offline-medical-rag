"""
build_harness_inputs_natural.py — Build a second harness input file from plain-language
questions (not from gold.jsonl).

These are natural, everyday-worded questions — one per document, plus a few that span
several documents — meant to test how the models handle realistic phrasing. Unlike
eval/gold.jsonl, these are NOT hand-verified with known answer chunks, so use them for
a qualitative read of grounding/answer quality, not for exact abstention scoring.

Same output format and retrieval settings as build_harness_inputs.py, so answer_harness.py
can run on it unchanged (point --inputs at this file).

    python -m eval.build_harness_inputs_natural
"""

import json
from pathlib import Path

from src.retrieve import retrieve

# (query_id, question, kind).  "kind" is just a label shown in the output.
QUESTIONS = [
    # ── one natural question per document ──
    ("n01", "Does adding ribociclib to letrozole actually help women whose breast cancer was already advanced when first diagnosed?", "natural_single"),
    ("n02", "What's the difference between a HER2 score of 0 and 1+, and why does it matter for treatment?", "natural_single"),
    ("n03", "Which CDK4/6 inhibitor works best, and which one has the worst side effects?", "natural_single"),
    ("n04", "Is Enhertu (T-DXd) better than regular chemotherapy for HER2-low breast cancer?", "natural_single"),
    ("n05", "What's the recommended first treatment for ER-positive, HER2-negative metastatic breast cancer?", "natural_single"),
    ("n06", "What is multi-omics, and how is it being used in breast cancer research?", "natural_single"),
    ("n07", "Is T-DXd worth the cost compared to standard chemotherapy?", "natural_single"),
    ("n08", "Does adding immunotherapy before surgery help triple-negative breast cancer, and does it hurt patients' quality of life?", "natural_single"),
    ("n09", "When should a breast cancer patient be offered genetic testing?", "natural_single"),
    ("n10", "What are the cancer risks for someone who carries a BRCA1 mutation?", "natural_single"),
    ("n11", "Does olaparib help stop breast cancer from coming back in people with BRCA mutations?", "natural_single"),
    ("n12", "Is it better to give chemotherapy before or after surgery for hormone-positive breast cancer?", "natural_single"),
    # ── questions that span multiple documents ──
    ("n13", "For HER2-low metastatic breast cancer, how well does T-DXd work — and is it worth the cost?", "natural_multi"),
    ("n14", "If someone has a BRCA mutation and breast cancer, what are their treatment and risk-management options?", "natural_multi"),
    ("n15", "How do the different CDK4/6 inhibitors compare, and what do the guidelines actually recommend for HR-positive metastatic disease?", "natural_multi"),
    ("n16", "What role does immunotherapy play in triple-negative breast cancer treatment?", "natural_multi"),
]

HOW_MANY_PASSAGES = 5   # top_k passages per question (same as the gold harness)

OUTPUT_PATH = Path("eval/harness_inputs_natural.json")


def section_label(passage):
    """Pick the most specific section heading available, for the SOURCE header."""
    return passage.get("section_h2") or passage.get("section_h1") or "—"


def main():
    harness_inputs = []

    for query_id, question, kind in QUESTIONS:
        print(f"Retrieving for {query_id} ({kind}): {question[:70]}...")
        results = retrieve(question, top_k=HOW_MANY_PASSAGES)

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
            "query":      question,
            "query_type": kind,
            "answerable": True,   # intended answerable, but NOT gold-verified
            "passages":   passages,
        })

    OUTPUT_PATH.write_text(
        json.dumps(harness_inputs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(harness_inputs)} questions -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

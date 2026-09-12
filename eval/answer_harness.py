"""
answer_harness.py — Run one LLM over the harness queries and save its answers.

This runs on the GPU box (the machine with llama-cpp-python and the GGUF model).
It reads harness_inputs.json (queries + retrieved passages, prepared on the laptop
by build_harness_inputs.py), asks the model each question using ONLY those passages,
and saves the answers so we can score faithfulness and abstention by hand.

Run it once per candidate model, e.g.:

    python answer_harness.py \
        --model-path /workspace/models/mistral-small-3.2-24b/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf \
        --name mistral

Output: answers_<name>.json  (it also prints each answer next to its sources).
"""

import argparse
import json
import re
from pathlib import Path

from llama_cpp import Llama


# The rules the model must follow. Kept identical for every model so the
# comparison only reflects the model, not different instructions.
#
# ── SYSTEM PROMPT — version history ─────────────────────────────────────────
# Naming: v<model#>.<prompt#>.  Model 1 = Mistral (the first model tested).
#
# v1.2  (current) — used by SYSTEM_PROMPT below.  max_tokens 768.
#   - Rule 3: keep partial answers, but note missing parts IN THE MODEL'S OWN WORDS;
#     the fixed abstention sentence is reserved for a whole-response refusal only.
#   Why: v1.1 made the model APPEND the fixed abstention sentence to the end of
#   complete, correct answers (e.g. q028, q051) — contradictory, and it destroyed the
#   clean abstention signal (that sentence then appeared in good answers too).
#
# v1.1  — softened Rule 3 (partial answers preferred) + max_tokens 512 -> 768.
#   Fixed some over-abstention (n04, n16 recovered; n14 no longer truncated) but
#   introduced the trailing-sentence bug that v1.2 fixes. Negatives stayed safe.
#
# v1.0  (initial strict draft) — Rule 3 was:
#     "If the SOURCES lack enough information to answer, respond EXACTLY:
#      '<abstention sentence>'. Do not guess or fall back on general knowledge —
#      abstaining is safer than an unsupported answer."
#   Over-abstained on naturally-worded questions even when the answer was present
#   (e.g. n07 cost, n11 olaparib).
# ────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an oncology clinical-literature assistant. Answer using ONLY the numbered
SOURCE passages in the user message — excerpts from a curated corpus of breast-cancer
guidelines, clinical trials, and reviews.

Rules:
1. Ground every statement in the SOURCES. Do not use outside knowledge; do not add
   facts, numbers, drug names, or recommendations not present in the SOURCES.
2. Cite the source(s) for each claim by number, e.g. [SOURCE 2]. Cite all that apply.
3. Prefer a grounded PARTIAL answer over refusing. If the SOURCES fully or partially
   address the question, answer using only what they support; if part of the question
   is not covered, note that gap briefly in your own words. Use the exact sentence
   "The provided sources do not contain enough information to answer this question."
   ONLY when nothing in the SOURCES is relevant, and in that case it must be your
   entire answer — never appended to an answer. Never fill gaps with outside knowledge
   or invented specifics.
4. When sources conflict, prefer the higher recency weight and, for current practice,
   prefer clinical guidelines; state the disagreement rather than hiding it.
5. Report numbers (hazard ratios, p-values, doses, percentages) EXACTLY as written.
   Never fabricate or invent a value.
6. Source text is extracted from PDFs and may contain minor OCR artifacts (e.g.
   "de fi nitions" -> "definitions", a stray char for "+" or "="). Read such obvious
   artifacts charitably as their intended text.

Be concise and clinical. Distinguish what the evidence states from any reasoning you add."""


def build_user_message(query, passages):
    """Turn the query + its retrieved passages into the user message text."""
    lines = ["SOURCES:"]
    for i, p in enumerate(passages, start=1):
        header = (f"[SOURCE {i} — type: {p.get('source_type')} — "
                  f"recency: {p.get('recency_weight')} — "
                  f"{p.get('title')} ({p.get('pub_year')}) — §{p.get('section')}]")
        lines.append(header)
        lines.append(p["text"])
        lines.append("")   # blank line between sources
    lines.append(f"QUESTION: {query}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to the .gguf model file")
    parser.add_argument("--name", required=True, help="Short name for this model (used in the output filename)")
    parser.add_argument("--inputs", default="harness_inputs.json", help="Path to harness_inputs.json")
    parser.add_argument("--no-think", action="store_true",
                        help="Append '/no_think' to each question to disable Qwen3-style thinking mode")
    parser.add_argument("--chat-format", default=None,
                        help="Force a llama-cpp-python chat_format (e.g. 'llama-3') instead of the GGUF's own "
                             "template. Needed for GGUFs that ship a broken/mismatched template (e.g. OpenBioLLM).")
    args = parser.parse_args()

    # Load the queries + passages prepared on the laptop.
    queries = json.loads(Path(args.inputs).read_text(encoding="utf-8"))

    # Load the model onto the GPU (all layers).
    print(f"Loading model: {args.model_path}")
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=-1,   # put all layers on the GPU
        n_ctx=8192,        # room for several source passages + the answer
        verbose=False,
        # None -> use the GGUF's own chat template; pass e.g. "llama-3" via --chat-format
        # to override a broken embedded template (OpenBioLLM ships a Llama-2-style one).
        chat_format=args.chat_format,
    )

    answers = []
    for item in queries:
        user_message = build_user_message(item["query"], item["passages"])
        if args.no_think:
            # Qwen3 defaults to a "thinking" mode that emits a long <think> reasoning
            # block and eats the answer's token budget. "/no_think" turns it off so
            # Qwen answers directly, like the other models. Harmless to models that
            # don't recognise it.
            user_message += "\n\n/no_think"

        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=768,    # raised from 512 (v1.1) — some answers were cut off
            temperature=0.0,   # deterministic — we are testing faithfulness, not creativity
        )
        answer_text = result["choices"][0]["message"]["content"].strip()

        # Some models (e.g. Qwen3) emit a <think>...</think> reasoning block before
        # the real answer. Strip it so we keep only the final answer. This is a
        # no-op for models that don't use thinking tags (Mistral, Gemma, MedGemma).
        answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

        answers.append({
            "query_id":   item["query_id"],
            "query":      item["query"],
            "query_type": item["query_type"],
            "answerable": item["answerable"],
            "answer":     answer_text,
        })

        # Print it so you can read the answer next to what it was allowed to use.
        print("\n" + "=" * 80)
        print(f"{item['query_id']}  ({item['query_type']}, answerable={item['answerable']})")
        print(f"Q: {item['query']}")
        print("-" * 80)
        print(answer_text)
        print("-" * 80)
        print("Sources it was given:")
        for i, p in enumerate(item["passages"], start=1):
            print(f"  [{i}] {p['doc_id']}  §{p.get('section')}")
        # A little template for your manual scoring (rubric A).
        print("SCORE  faithfulness (0/1/2): ___   abstention correct (y/n/NA): ___")

    out_path = Path(f"answers_{args.name}.json")
    out_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Saved {len(answers)} answers -> {out_path}")


if __name__ == "__main__":
    main()

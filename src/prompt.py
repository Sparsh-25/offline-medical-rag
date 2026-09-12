"""
prompt.py — builds the RAG prompt for answer generation (Stage 8).

Pipeline position:
    retrieve.py -> [prompt.py] -> answer.py

Two pieces:
  * SYSTEM_PROMPT — the rules the model must follow, locked at v1.2 (chosen through the
    model comparison; see decisions.md D60 and the version history in
    eval/answer_harness.py).
  * build_messages() — turns a query plus the chunks returned by retrieve.py into the
    [system, user] message list the LLM is called with.

This is pure string logic with no model dependency, so it imports anywhere (laptop or
GPU box) and is easy to test on its own.
"""

from __future__ import annotations


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


def _section_label(chunk: dict) -> str:
    """Most specific section heading available on a chunk, for the SOURCE header."""
    return chunk.get("section_h2") or chunk.get("section_h1") or "—"


def build_user_message(query: str, chunks: list[dict], no_think: bool = False) -> str:
    """
    Assemble the retrieved chunks into a numbered SOURCES block followed by the
    question. `chunks` are the dicts returned by retrieve.retrieve() (they carry
    chunk_text, section_h1/h2, source_type, recency_weight, title, pub_year).
    """
    lines = ["SOURCES:"]
    for i, c in enumerate(chunks, start=1):
        header = (f"[SOURCE {i} — type: {c.get('source_type')} — "
                  f"recency: {c.get('recency_weight')} — "
                  f"{c.get('title')} ({c.get('pub_year')}) — §{_section_label(c)}]")
        lines.append(header)
        lines.append(c["chunk_text"])
        lines.append("")   # blank line between sources
    lines.append(f"QUESTION: {query}")

    text = "\n".join(lines)
    if no_think:
        # Qwen3 defaults to a "thinking" mode that emits a long reasoning block and
        # eats the answer's token budget. "/no_think" turns it off (decisions.md D60).
        text += "\n\n/no_think"
    return text


def build_messages(query: str, chunks: list[dict], no_think: bool = False) -> list[dict]:
    """Return the [system, user] message list for llama-cpp-python's chat API."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(query, chunks, no_think=no_think)},
    ]

"""
verify_answer.py — Stage 9 answer grounding / faithfulness verification (in progress).

Pipeline position:
    ... -> answer.py -> [verify_answer.py]

The goal (see decisions.md D69) is to check that an *answered* response is actually
grounded in the sources the model saw, without changing the answer. The stage has two
parts:

  1. Claim decomposition  — split the answer into small checkable claims, each carrying
                            the [SOURCE n] citation(s) it drew from.   <-- THIS FILE, so far
  2. Evidence matching    — score each claim against its cited source with an NLI model
                            PLUS a deterministic numeric/entity check.  <-- next, after the
                            model A/B and the claim-level eval.

This file implements part 1 only. It is deliberately dependency-free (standard library +
`re`) so it runs on a plain machine with no torch, and so the decomposition is fully
deterministic and reproducible — the same answer always yields the same claims.

Why deterministic sentence-level claims instead of an LLM atomic-fact decomposer:
an LLM decomposer is a second generative stage that can silently drop, merge, or reword
claims (a dropped claim is an unchecked claim — a safety hole). We start simple and will
escalate to LLM atomic decomposition ONLY if the claim-level eval shows sentence-level is
too coarse. See decisions.md D69.

Usage (produce the labelling sheet the faithfulness eval will need):
    python -m src.verify_answer --answers eval/answers_gold_47docs_safetynet.jsonl \
                                --out eval/faithfulness_claims_to_label.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The fixed refusal sentence (kept in sync with answer.py's NO_EVIDENCE). An abstention
# has no claims to verify, so we skip those rows. Duplicated here on purpose: importing
# answer.py would pull in torch/retrieval, and this module is meant to stay torch-free.
NO_EVIDENCE = "The provided sources do not contain enough information to answer this question."


# --- sentence splitting -----------------------------------------------------------------
# Medical text is full of periods that do NOT end a sentence: decimals (0.593), p-values
# (P < .001), and a few abbreviations (vs., e.g., i.e.). A naive split on ". " shreds those.
# So we first mask those periods with a placeholder, split on real sentence boundaries,
# then restore. This is a transparent regex approach (no spaCy dependency to download).

_DOT = "\x00"   # placeholder that cannot occur in normal text

# Abbreviations that are followed by a period but not a sentence end (lowercased, no dot).
_ABBREVIATIONS = ["vs", "e.g", "i.e", "approx", "ca", "cf", "al", "fig", "no", "ref"]


def _mask_non_terminal_periods(text: str) -> str:
    """Replace periods that are not sentence terminators with a placeholder."""
    # Decimals / version numbers: a period flanked by digits (0.593, 95.5).
    text = re.sub(r"(\d)\.(\d)", rf"\1{_DOT}\2", text)
    # Leading-dot numbers like "P < .001": a period preceded by a space and followed by a digit.
    text = re.sub(r"(?<=\s)\.(?=\d)", _DOT, text)
    # Known abbreviations: "vs." -> "vs<DOT>" etc.
    for abbr in _ABBREVIATIONS:
        text = re.sub(rf"\b{re.escape(abbr)}\.", abbr + _DOT, text, flags=re.IGNORECASE)
    return text


def split_sentences(text: str) -> list[str]:
    """Split answer text into sentences, protecting decimals/p-values/abbreviations.

    A boundary is sentence-ending punctuation (. ! ?) followed by whitespace and then an
    uppercase letter, a digit, or a bracket (where the next claim usually begins).
    """
    masked = _mask_non_terminal_periods(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", masked)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


# --- citation extraction ----------------------------------------------------------------
# Prompt v1.2 asks for "[SOURCE n]", and that is the common form, but the model also emits
# "[SOURCE 1, SOURCE 2]" and parenthetical/lowercase variants like "(Sources 1, 2, 4, 7)".
# This matches any bracketed or parenthesised run containing the word "source(s)" and pulls
# the numbers out of it. A bare "(3)" (no "source" word) is intentionally NOT treated as a
# citation — too ambiguous against a numeric value in parentheses.
_CITATION = re.compile(r"[\[(]\s*sources?\b[^\])]*[\])]", re.IGNORECASE)


def extract_citations(text: str) -> list[int]:
    """Return the sorted, de-duplicated SOURCE numbers cited in a piece of text."""
    numbers: list[int] = []
    for marker in _CITATION.finditer(text):
        numbers += [int(n) for n in re.findall(r"\d+", marker.group())]
    return sorted(set(numbers))


# --- decomposition ----------------------------------------------------------------------

def decompose_claims(answer_text: str) -> list[dict]:
    """Split an answer into sentence-level claims, each with its cited SOURCE numbers.

    Returns a list of {claim_index, text, cited_sources}. A claim with an empty
    cited_sources list is uncited — something the verifier should flag.
    """
    claims = []
    for sentence in split_sentences(answer_text):
        claims.append({
            "claim_index":   len(claims),
            "text":          sentence,
            "cited_sources": extract_citations(sentence),
        })
    return claims


# --- evidence resolution ----------------------------------------------------------------
# A claim citing [SOURCE n] points at the n-th source the model saw, i.e. the n-th entry of
# the answer's retrieved_chunk_ids (1-indexed, in rank order). We map that back to the
# chunk's text from chunks.jsonl. An UNcited claim has no specific source to check, so we
# fall back to the full set of chunks the model saw (the D69 "cited-first, then full-set"
# rule — an uncited claim is only "unsupported" if nothing retrieved backs it).

def load_chunk_texts(chunks_path: Path) -> dict[str, str]:
    """Map chunk_id -> chunk_text from a chunks.jsonl file."""
    texts = {}
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            texts[row["chunk_id"]] = row.get("chunk_text", "")
    return texts


def resolve_evidence(cited_sources: list[int], sources_seen: list[str], chunk_texts: dict[str, str]) -> str:
    """Return the evidence text a claim should be checked against.

    Cited claim -> the specifically cited chunks; uncited claim -> every chunk the model saw.
    """
    if cited_sources:
        ids = [sources_seen[n - 1] for n in cited_sources if 1 <= n <= len(sources_seen)]
    else:
        ids = list(sources_seen)
    return "\n".join(chunk_texts.get(cid, "") for cid in ids)


# --- numeric grounding check (the deterministic primary signal, D69) --------------------
# Our safety-critical claims are quantitative (HR, %, dose, CI, cost), and this is exactly
# where NLI is blind. So for every number in a claim we require a matching number in the
# cited evidence. "Matching" allows rounding: a claim's 0.59 is satisfied by an evidence
# 0.593 (0.593 rounds to 0.59), but a claim's 23.9 is NOT satisfied by 23.4 — so a wrong
# figure is caught while an honest rounding is not. Commas/%/$ are stripped before compare,
# so this checks the VALUE, not its formatting (a misplaced thousands-comma is a cosmetic
# bug, not an ungrounded claim).

_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")

# Noise that looks numeric but isn't a claim figure: citation markers (bracketed OR bare
# "SOURCE n" prose), and markdown list enumerators ("1.", "2)") at the start of a line.
_BARE_CITATION = re.compile(r"\bsources?\s+\d+", re.IGNORECASE)
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _strip_non_claim_tokens(text: str) -> str:
    """Remove citation markers and list enumerators so their digits aren't read as stats."""
    text = _CITATION.sub("  ", text)
    text = _BARE_CITATION.sub("  ", text)
    text = _LIST_MARKER.sub("  ", text)
    return text


def _to_value(token: str) -> float | None:
    core = token.replace("$", "").replace("%", "").replace(",", "")
    try:
        return float(core)
    except ValueError:
        return None


def _decimals(token: str) -> int:
    core = token.replace(",", "")
    return len(core.split(".")[1]) if "." in core else 0


def _is_identifier_number(text: str, start: int, end: int) -> bool:
    """True if the number is part of an alphanumeric identifier, not a standalone statistic.

    Catches trial/gene/stage codes (KEYNOTE-522, BRCA1, T1c, CDK4/6, QLQ-C30), where the
    digit is glued to letters directly or through a '-'/'/' connector — e.g. "522" in
    "KEYNOTE-522" (before it: '-', then a letter) but NOT "20" in "15-20" (before the '-':
    a digit). Those identifier numbers are not claims we should grounding-check.
    """
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    if before.isalpha() or after.isalpha():
        return True
    if before in "-/" and start - 2 >= 0 and text[start - 2].isalpha():
        return True
    return False


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """Return (raw token, numeric value, decimal places) for each statistic in the text.

    Citation markers ([SOURCE n], (Sources ...)) are removed first so their numbers aren't
    treated as claim figures, and identifier numbers (trial/gene/stage codes) are skipped.
    """
    text = _strip_non_claim_tokens(text)
    found = []
    for m in _NUMBER.finditer(text):
        if _is_identifier_number(text, m.start(), m.end()):
            continue
        token = m.group().rstrip(",")
        value = _to_value(token)
        if value is not None:
            found.append((token, value, _decimals(token)))
    return found


def check_claim_numbers(claim_text: str, evidence_text: str) -> dict:
    """Check that every number in the claim appears (within rounding) in the evidence.

    Returns {n_numbers, unmatched, grounded}. `unmatched` lists claim numbers with no
    rounding-equal match in the evidence — the flag that the claim asserts a figure its
    cited source does not support.
    """
    evidence_values = [v for _, v, _ in extract_numbers(evidence_text)]
    claim_numbers = extract_numbers(claim_text)
    unmatched = []
    for raw, value, places in claim_numbers:
        if not any(round(e, places) == value or e == value for e in evidence_values):
            unmatched.append(raw)
    return {"n_numbers": len(claim_numbers), "unmatched": unmatched, "grounded": not unmatched}


# --- CLI: build the labelling sheet for the faithfulness eval ---------------------------

def build_labelling_sheet(answers_path: Path, out_path: Path) -> None:
    """Decompose every answered answerable into claims and write a sheet to hand-label.

    We only decompose rows that actually answered (abstentions have no claims). Each output
    row is one claim with blank `label_verdict` / `label_note` fields for a human to fill:
    the claim-level gold the model A/B (decisions.md D69) will be scored against.
    """
    rows = [json.loads(l) for l in answers_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    sheet = []
    n_answers = n_claims = n_uncited = 0
    for row in rows:
        if row.get("abstained") or row["answer"].strip() == NO_EVIDENCE:
            continue
        n_answers += 1
        for claim in decompose_claims(row["answer"]):
            n_claims += 1
            if not claim["cited_sources"]:
                n_uncited += 1
            sheet.append({
                "query_id":      row["query_id"],
                "query_type":    row.get("query_type", ""),
                "claim_index":   claim["claim_index"],
                "claim_text":    claim["text"],
                "cited_sources": claim["cited_sources"],
                "sources_seen":  row.get("retrieved_chunk_ids", []),
                # fill one: supported_as_cited | supported_but_miscited | unsupported | not_a_claim
                # (not_a_claim = a meta/hedge sentence, e.g. "X is not detailed in the sources",
                #  or a non-factual lead-in — excluded from scoring, not counted as a failure)
                "label_verdict": "",
                "label_note":    "",
            })

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sheet) + "\n", encoding="utf-8")
    print(f"answers decomposed: {n_answers}")
    print(f"claims extracted:   {n_claims}  (avg {n_claims / max(n_answers, 1):.1f} per answer)")
    print(f"uncited claims:     {n_uncited}  ({100 * n_uncited / max(n_claims, 1):.0f}% — flag candidates)")
    print(f"labelling sheet ->  {out_path}")


def report_numeric_flags(answers_path: Path, chunks_path: Path) -> None:
    """Run the numeric grounding check over every claim and report the flagged ones.

    This is the deterministic half of Stage 9 — no model needed. It shows which claims
    assert a number their cited evidence does not support.
    """
    chunk_texts = load_chunk_texts(chunks_path)
    rows = [json.loads(l) for l in answers_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    n_claims = n_numeric = 0
    miscited, unsupported = [], []
    for row in rows:
        if row.get("abstained") or row["answer"].strip() == NO_EVIDENCE:
            continue
        seen = row.get("retrieved_chunk_ids", [])
        for claim in decompose_claims(row["answer"]):
            n_claims += 1
            cited_ev = resolve_evidence(claim["cited_sources"], seen, chunk_texts)
            cited_result = check_claim_numbers(claim["text"], cited_ev)
            if cited_result["n_numbers"]:
                n_numeric += 1
            if cited_result["grounded"]:
                continue
            # A number is missing from the CITED source. Is it anywhere the model saw?
            full_ev = resolve_evidence([], seen, chunk_texts)   # [] -> all retrieved chunks
            anywhere = check_claim_numbers(claim["text"], full_ev)["unmatched"]
            row_flag = (row["query_id"], claim["cited_sources"], cited_result["unmatched"], claim["text"])
            if anywhere:
                unsupported.append((*row_flag[:2], anywhere, row_flag[3]))   # not in any retrieved chunk
            else:
                miscited.append(row_flag)                                    # in another retrieved chunk

    print(f"claims checked:            {n_claims}")
    print(f"with numbers:              {n_numeric}")
    print(f"MISCITED (number is in another retrieved chunk, not the cited one): {len(miscited)}")
    print(f"UNSUPPORTED (number in no retrieved chunk at all):                  {len(unsupported)}\n")
    print("=== UNSUPPORTED (the serious flag) ===")
    for qid, cites, unmatched, text in unsupported:
        print(f"  {qid} cites {cites or '(uncited)'} | unmatched {unmatched}\n      {text[:110]}")
    print("\n=== MISCITED (citation accuracy, milder) ===")
    for qid, cites, unmatched, text in miscited:
        print(f"  {qid} cites {cites or '(uncited)'} | unmatched {unmatched}\n      {text[:110]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer decomposition + numeric grounding check (Stage 9, parts 1-2a).")
    parser.add_argument("--answers", required=True, help="Answer JSONL from eval_answers (e.g. eval/answers_gold_47docs_safetynet.jsonl)")
    parser.add_argument("--out", help="Write the claim labelling sheet to this JSONL")
    parser.add_argument("--check-numbers", metavar="CHUNKS", help="Run the numeric grounding check using this chunks.jsonl")
    args = parser.parse_args()

    if args.check_numbers:
        report_numeric_flags(Path(args.answers), Path(args.check_numbers))
    elif args.out:
        build_labelling_sheet(Path(args.answers), Path(args.out))
    else:
        parser.error("give --out (build labelling sheet) or --check-numbers CHUNKS (run numeric check)")


if __name__ == "__main__":
    main()

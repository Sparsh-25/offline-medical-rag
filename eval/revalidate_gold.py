"""
revalidate_gold.py — verify every eval/gold.jsonl entry still points at the chunk that
holds its answer, after the corpus was re-chunked (MedCPT tokenizer switch + D62 NCCN
watermark cleanup shifted chunk boundaries), and propose fixes for the ones that drifted.

Method (we have no old chunks to diff against, so we verify by *content*): each answerable
entry's `notes` (and `query`) describe the specific answer. We build a set of "content
tokens" from that description — distinctive words (>=4 chars) and numbers (decimals, %,
and multi-digit integers) — and check what fraction appear in the currently-cited chunk:

  VALID   : the cited chunk still contains most of the answer's content tokens.
  DRIFTED : it doesn't — we propose the chunk (in the same document) with the best match.

Verifying the *cited* chunk directly (not ranking all chunks) keeps false alarms low: a
correct chunk that merely isn't the #1 lexical match still passes. It only prints a report;
confirm each DRIFTED fix before editing gold.jsonl (a wrong gold id corrupts eval — D38).

    python -m eval.revalidate_gold
"""

import json
import re
from collections import defaultdict
from pathlib import Path

GOLD_PATH = Path("eval/gold.jsonl")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")

VALID_FRACTION = 0.40   # cited chunk must contain >= this fraction of the answer's tokens

# words that describe the gold-entry bookkeeping, not the answer — ignore them
_META = set("chunk chunks section states restates same target original trimmed broadened "
            "clinical shorthand phrasing variant cross document multi hop table prose gives "
            "reported this that from with the notes discussion".split())
_WORD = re.compile(r"[a-z][a-z0-9-]{3,}")     # words >= 4 chars
_NUM = re.compile(r"\d+\.\d+|\d+%|\d{3,}")    # decimals, percentages, 3+ digit integers


def content_tokens(text: str) -> set:
    text = re.sub(r"chunk_\d+", " ", text or "")
    words = {w for w in _WORD.findall(text.lower()) if w not in _META}
    return words | set(_NUM.findall(text))


def chunk_tokens(text: str) -> set:
    return {w for w in _WORD.findall((text or "").lower())} | set(_NUM.findall(text or ""))


def load_chunks():
    by_id, by_doc = {}, defaultdict(list)
    for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            by_id[c["chunk_id"]] = c["chunk_text"]
            by_doc[c["doc_id"]].append(c["chunk_id"])
    return by_id, by_doc


def doc_of(cid: str) -> str:
    return cid.rsplit("_chunk_", 1)[0]


def main():
    by_id, by_doc = load_chunks()
    gold = [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    valid = drifted = 0
    per_doc = defaultdict(lambda: [0, 0])   # doc -> [valid, drifted]

    for e in gold:
        if not e.get("answerable"):
            continue
        want = content_tokens((e.get("notes") or "") + "  " + (e.get("query") or ""))
        if not want:
            continue

        entry_ok = True
        report = []
        for rcid in e.get("relevant_chunk_ids", []):
            ctoks = chunk_tokens(by_id.get(rcid, ""))
            frac = len(want & ctoks) / len(want) if rcid in by_id else 0.0
            per_doc[doc_of(rcid)]  # touch
            if frac < VALID_FRACTION:
                entry_ok = False
                report.append(f"   cited {rcid}: {'MISSING' if rcid not in by_id else f'only {frac:.0%} of answer tokens'}")
                # propose the best-matching chunk in the same document
                ranked = sorted(by_doc.get(doc_of(rcid), []),
                                key=lambda c: len(want & chunk_tokens(by_id[c])), reverse=True)
                for cid in ranked[:2]:
                    f = len(want & chunk_tokens(by_id[cid])) / len(want)
                    report.append(f"      -> {cid}  ({f:.0%} match): {by_id[cid][:80].strip()}...")

        if entry_ok:
            valid += 1
            per_doc[doc_of(e['relevant_chunk_ids'][0])][0] += 1
        else:
            drifted += 1
            per_doc[doc_of(e['relevant_chunk_ids'][0])][1] += 1
            print(f"\n[DRIFTED] {e['query_id']} ({e['query_type']}): {e['query'][:80]}")
            print("\n".join(report))

    print("\n=== per-document (valid / drifted) ===")
    for d in sorted(per_doc):
        v, dr = per_doc[d]
        flag = "" if dr == 0 else "  <-- needs fixes"
        print(f"  {d:42s} {v:2d} valid / {dr:2d} drifted{flag}")
    print(f"\n=== TOTAL: {valid} valid, {drifted} drifted ===")


if __name__ == "__main__":
    main()

"""
review_gold.py — Print each eval/gold.jsonl entry next to the chunk(s) it
cites, so you can check the citation is actually correct without manually
looking up chunk_ids yourself.

Usage:
    python -m eval.review_gold                          # everything
    python -m eval.review_gold --doc-id monaleesa2_subanalysis_2018
    python -m eval.review_gold --query-id q005
"""

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH      = _ROOT / "eval" / "gold.jsonl"
CHUNK_MAP_PATH = _ROOT / "index" / "chunk_map.json"


def load_gold() -> list[dict]:
    with open(GOLD_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_chunk_map() -> dict[str, dict]:
    return json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))


def print_entry(entry: dict, chunk_map: dict[str, dict]) -> None:
    print("\n" + "=" * 70)
    print(f"{entry['query_id']}  [{entry['query_type']}]  answerable={entry['answerable']}")
    print(f"Query: {entry['query']}")
    print(f"Notes: {entry.get('notes', '')}")

    for chunk_id in entry["relevant_chunk_ids"]:
        chunk = chunk_map.get(chunk_id)
        if chunk is None:
            print(f"\n  !! chunk_id '{chunk_id}' not found in chunk_map.json")
            continue
        print(f"\n  --- {chunk_id} ---")
        print(f"  {chunk.get('section_h1')} > {chunk.get('section_h2')}")
        print(f"  {chunk['chunk_text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review eval/gold.jsonl entries against their cited chunks.")
    parser.add_argument("--doc-id", help="Only show queries for this doc_id")
    parser.add_argument("--query-id", help="Only show this one query_id")
    args = parser.parse_args()

    gold = load_gold()
    chunk_map = load_chunk_map()

    if args.query_id:
        gold = [e for e in gold if e["query_id"] == args.query_id]
    elif args.doc_id:
        gold = [e for e in gold if args.doc_id in e["relevant_doc_ids"]]

    if not gold:
        print("No matching entries.")
        return

    for entry in gold:
        print_entry(entry, chunk_map)


if __name__ == "__main__":
    main()

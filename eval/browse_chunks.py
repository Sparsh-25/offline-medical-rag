"""
browse_chunks.py — Read chunks while writing eval/gold.jsonl entries.

For each gold query you write, you need to know the exact chunk_id that
answers it. This script just prints chunks so you can read them and note
the chunk_id down — no other purpose.

Usage:
    python -m eval.browse_chunks --list-docs
    python -m eval.browse_chunks --doc-id monaleesa2_subanalysis_2018
    python -m eval.browse_chunks --doc-id nccn_breast_v2_2026 --start 40 --count 10
    python -m eval.browse_chunks --doc-id nccn_breast_v2_2026 --search trastuzumab
"""

import argparse
import json
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((_ROOT / "config.yaml").read_text())
CHUNKS_PATH = _ROOT / cfg["paths"]["chunks"]


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def list_docs(chunks: list[dict]) -> None:
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk["doc_id"]] = counts.get(chunk["doc_id"], 0) + 1

    print(f"\n{len(counts)} documents, {len(chunks)} chunks total:\n")
    for doc_id, count in sorted(counts.items()):
        print(f"  {doc_id:45s} {count:4d} chunks")


def print_chunk(index: int, chunk: dict) -> None:
    print(f"\n[{index}] {chunk['chunk_id']}  ({chunk['content_type']})")
    print(f"    {chunk.get('section_h1')} > {chunk.get('section_h2')}")
    print(f"    {chunk['chunk_text']}")


def show_chunks(chunks: list[dict], doc_id: str, start: int, count: int) -> None:
    doc_chunks = [c for c in chunks if c["doc_id"] == doc_id]
    if not doc_chunks:
        print(f"No chunks found for doc_id '{doc_id}'. Use --list-docs to see valid ids.")
        return

    print(f"\n{doc_id}: showing chunks {start} to {start + count - 1} of {len(doc_chunks)}")
    for i, chunk in enumerate(doc_chunks[start:start + count], start=start):
        print_chunk(i, chunk)


def search_chunks(chunks: list[dict], doc_id: str, keyword: str) -> None:
    doc_chunks = [c for c in chunks if c["doc_id"] == doc_id]
    matches = [c for c in doc_chunks if keyword.lower() in c["chunk_text"].lower()]

    print(f"\n{doc_id}: {len(matches)} chunks contain '{keyword}'")
    for i, chunk in enumerate(matches):
        print_chunk(i, chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse chunks.jsonl for gold eval set writing.")
    parser.add_argument("--list-docs", action="store_true", help="List all doc_ids with chunk counts")
    parser.add_argument("--doc-id", help="Document to browse")
    parser.add_argument("--start", type=int, default=0, help="First chunk index to show (default 0)")
    parser.add_argument("--count", type=int, default=15, help="How many chunks to show (default 15)")
    parser.add_argument("--search", help="Only show chunks containing this keyword")
    args = parser.parse_args()

    chunks = load_chunks()

    if args.list_docs:
        list_docs(chunks)
    elif args.search and args.doc_id:
        search_chunks(chunks, args.doc_id, args.search)
    elif args.doc_id:
        show_chunks(chunks, args.doc_id, args.start, args.count)
    else:
        parser.error("Specify --list-docs, or --doc-id (optionally with --search)")


if __name__ == "__main__":
    main()

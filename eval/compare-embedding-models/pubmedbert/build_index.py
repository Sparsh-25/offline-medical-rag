"""
build_index.py — builds a FAISS dense index using NeuML/pubmedbert-base-embeddings
instead of the project's default BAAI/bge-base-en-v1.5.

This is one of the candidate embedding models being compared against the BGE-base
baseline (decisions.md D45/D46). It re-uses the exact same data/chunks/chunks.jsonl
as the live corpus, but writes its own faiss.index + id_map.json into THIS folder's
index/ subdirectory — the live index/faiss.index is never touched.

BM25 and chunk_map.json are NOT rebuilt here: they don't depend on the embedding
model at all (BM25 is plain keyword search over chunk_text, chunk_map.json is just
chunk metadata), so the live index/bm25.pkl and index/chunk_map.json are reused
as-is by eval_retrieval.py when comparing this model.

Usage:
    python "eval/compare-embedding-models/pubmedbert/build_index.py"
"""

from __future__ import annotations

# Must be set before sentence_transformers/torch is imported, or macOS segfaults.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
DEVICE     = "cpu"   # matches config.yaml embedding.device

_ROOT       = Path(__file__).resolve().parents[3]   # project root
CHUNKS_PATH = _ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT_DIR  = Path(__file__).parent / "index"
FAISS_PATH  = OUTPUT_DIR / "faiss.index"
ID_MAP_PATH = OUTPUT_DIR / "id_map.json"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    texts = [c["chunk_text"] for c in chunks]
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # L2-norm so inner product == cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Encoded {len(texts):,} chunks in {time.time() - t0:.1f}s  (dim={embeddings.shape[1]})")

    # Same index type as the live corpus (IndexFlatIP, exact search — decisions.md D22)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(FAISS_PATH))
    print(f"FAISS index saved -> {FAISS_PATH}  ({FAISS_PATH.stat().st_size / 1e6:.1f} MB)")

    id_map = {i: c["chunk_id"] for i, c in enumerate(chunks)}
    ID_MAP_PATH.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    print(f"ID map saved -> {ID_MAP_PATH}  ({len(id_map):,} entries)")


if __name__ == "__main__":
    main()

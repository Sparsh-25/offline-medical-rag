"""
build_index.py — builds a FAISS dense index using ncbi/MedCPT-Article-Encoder.

MedCPT is different from the other three candidates (BGE-base, PubMedBERT, BGE-M3):
it's an ASYMMETRIC dual-encoder, not a single symmetric SentenceTransformer model.
Chunks (this script) are embedded with the Article-Encoder; queries (see
eval_medcpt.py) are embedded with a *different* model, the Query-Encoder. Both
map into the same 768-dim space, so they're still comparable via cosine
similarity — see decisions.md D46 for why this needs different code from the
other three candidates.

Two deliberate choices, not just following the model card verbatim:
  1. The Article-Encoder expects [title, text] PAIRS (it was trained on PubMed
     title+abstract pairs), not a single string. We use [chunk["title"],
     chunk["chunk_text"]] — the document title already stored on every chunk,
     not invented content.
  2. The official model-card example does NOT L2-normalize the output. We do
     it here anyway, to stay consistent with how the rest of this project uses
     FAISS IndexFlatIP: inner product only equals cosine similarity for unit
     vectors, and every other index in this project is built that way.

Usage:
    python "eval/compare-embedding-models/medcpt/build_index.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
MAX_LENGTH = 512   # per the model card — tuned for title+abstract-length input
BATCH_SIZE = 16    # smaller than the other scripts' 32: raw transformers here,
                    # no sentence-transformers batching/memory optimisation

_ROOT       = Path(__file__).resolve().parents[3]   # project root
CHUNKS_PATH = _ROOT / "data" / "chunks" / "chunks.jsonl"
OUTPUT_DIR  = Path(__file__).parent / "index"
FAISS_PATH  = OUTPUT_DIR / "faiss.index"
ID_MAP_PATH = OUTPUT_DIR / "id_map.json"


def embed_articles(pairs: list[list[str]], model, tokenizer) -> np.ndarray:
    """Encode a list of [title, text] pairs into L2-normalised 768-dim vectors."""
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            encoded = tokenizer(
                batch, truncation=True, padding=True, return_tensors="pt", max_length=MAX_LENGTH
            )
            # [CLS] token's last hidden state — this is the model card's documented
            # way to get the article embedding (no pooler head on this model).
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)  # see module docstring, point 2
            all_embeds.append(embeds.numpy())
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"  encoded {min(i + BATCH_SIZE, len(pairs))}/{len(pairs)}")
    return np.concatenate(all_embeds, axis=0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(line) for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH}")

    print(f"Loading embedding model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # [title, chunk_text] pairs — see module docstring, point 1
    pairs = [[c["title"] or "", c["chunk_text"]] for c in chunks]

    t0 = time.time()
    embeddings = embed_articles(pairs, model, tokenizer)
    print(f"Encoded {len(pairs):,} chunks in {time.time() - t0:.1f}s  (dim={embeddings.shape[1]})")

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

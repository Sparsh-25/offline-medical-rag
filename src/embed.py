"""
embed.py — Dense + Sparse index builder for the Oncology RAG pipeline.

Pipeline position:
    extract.py → chunk.py → [embed.py] → retrieve / answer

What this script does
─────────────────────
1. Reads  data/chunks/chunks.jsonl  produced by chunk.py.
2. Dense path:
   • Batch-encodes chunk_text with the model set in config.yaml → embedding.model
     (currently BAAI/bge-m3 — see decisions.md D52 for why, D22 for index type).
   • Applies the BGE instruction prefix for passage encoding.
   • Builds a FAISS HNSW32 index (cosine similarity via inner-product on
     unit-normalised vectors).
   • Saves  index/faiss.index.
3. Sparse path:
   • Builds a BM25Okapi index over tokenised chunk_text.
   • Saves  index/bm25.pkl  (pickle).
4. Writes  index/id_map.json  —  int_position → chunk_id
   (needed at query time to go from FAISS result slot back to a chunk).
5. Validates the FAISS index with a smoke-test query.

Usage
─────
    python src/embed.py              # build both indexes
    python src/embed.py --dense-only # skip BM25
    python src/embed.py --sparse-only# skip FAISS

Requirements (add to requirements.txt if not present)
─────────────────────────────────────────────────────
    sentence-transformers>=3.0.0
    faiss-cpu>=1.8.0
    rank-bm25>=0.2.2
    numpy>=1.26.0
"""

from __future__ import annotations

# ── macOS / PyTorch segfault guard ────────────────────────────────────────────
# Must be set BEFORE torch / transformers import anything.
# On macOS, forked tokenizer workers + semaphores crash with exit code 139.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # disables HF fast-tokenizer fork
os.environ.setdefault("OMP_NUM_THREADS", "1")             # prevents OpenMP thread oversubscription
os.environ.setdefault("MKL_NUM_THREADS", "1")             # same for MKL backend
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ── Optional heavy imports (fail loudly with clear message) ───────────────────

def _require(pkg: str, install: str) -> Any:
    """Import a package or exit with a helpful error."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ModuleNotFoundError:
        logging.error(
            f"Missing dependency '{pkg}'. Install with:\n"
            f"    pip install {install}"
        )
        sys.exit(1)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent   # project root
cfg = yaml.safe_load((_ROOT / "config.yaml").read_text())

EMB_CFG      = cfg["embedding"]
ARCHITECTURE = EMB_CFG.get("architecture", "symmetric")  # "symmetric" | "asymmetric" — decisions.md D53/D56
MODEL_ID     = EMB_CFG["model"] if ARCHITECTURE == "symmetric" else None  # a string for symmetric;
                                                                            # asymmetric reads model.article directly
BATCH_SZ  = int(EMB_CFG["batch_size"])
DEVICE    = EMB_CFG["device"]          # "cpu" | "cuda" | "mps"
ASYMMETRIC_BATCH_SZ = 16   # smaller than BATCH_SZ: raw transformers here, no
                           # sentence-transformers batching/memory optimisation

CHUNKS_PATH = _ROOT / cfg["paths"]["chunks"]        # data/chunks/chunks.jsonl
FAISS_PATH  = _ROOT / cfg["paths"]["faiss_index"]   # index/faiss.index
BM25_PATH   = _ROOT / cfg["paths"]["bm25_index"]    # index/bm25.pkl
ID_MAP_PATH = FAISS_PATH.parent / "id_map.json"     # index/id_map.json

# BGE-specific instruction prefix that boosts retrieval quality for passages
BGE_PASSAGE_PREFIX = ""   # BGE encodes passages WITHOUT the instruction prefix
                           # (only queries use the prefix at query time)

# FAISS HNSW construction/search params — tune for speed vs recall trade-off
HNSW_M = 32          # neighbours per node; 32 is a good default
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64  # was previously unset (FAISS default 16) — a live recall
                     # bug flagged in decisions.md D4, fixed here regardless
                     # of which index type D22's comparison ultimately picks

INDEX_TYPE = EMB_CFG.get("index_type", "hnsw")  # "flat" | "hnsw" — decisions.md D4/D22


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_chunks() -> list[dict]:
    """Load all chunks from the JSONL file produced by chunk.py."""
    if not CHUNKS_PATH.exists():
        log.error(
            f"Chunks file not found: {CHUNKS_PATH}\n"
            "Run  python src/chunk.py  first."
        )
        sys.exit(1)

    chunks: list[dict] = []
    with open(CHUNKS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    log.info(f"Loaded {len(chunks):,} chunks from {CHUNKS_PATH.name}")
    return chunks


def get_texts(chunks: list[dict]) -> list[str]:
    """
    Extract chunk_text, prepending BGE passage prefix (empty string here —
    kept as a hook so you can switch to instruction-tuned variants like
    bge-large-en-v1.5 or bge-m3 without changing call sites).
    """
    return [BGE_PASSAGE_PREFIX + c["chunk_text"] for c in chunks]


# ── Dense index ───────────────────────────────────────────────────────────────

def _encode_symmetric(chunks: list[dict]) -> np.ndarray:
    """Encode chunks with a single SentenceTransformer model (the default case)."""
    SentenceTransformer = _require("sentence_transformers", "sentence-transformers").SentenceTransformer

    log.info(f"Loading embedding model: {MODEL_ID}  (device={DEVICE})")
    model = SentenceTransformer(MODEL_ID, device=DEVICE)

    texts = get_texts(chunks)
    n     = len(texts)
    # get_embedding_dimension() is the current API (get_sentence_embedding_dimension
    # was deprecated in sentence-transformers v3 and will be removed).
    dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
    dim   = dim_fn()
    log.info(f"Embedding {n:,} chunks  (dim={dim}, batch={BATCH_SZ})")

    t0 = time.time()
    # show_progress_bar gives a tqdm bar per batch
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=BATCH_SZ,
        normalize_embeddings=True,   # L2-norm in-place → cosine via IP
        show_progress_bar=True,
        convert_to_numpy=True,
        # Note: num_workers was removed in sentence-transformers v3.
        # Segfault prevention is handled by TOKENIZERS_PARALLELISM=false
        # and OMP_NUM_THREADS=1 set at the top of this file.
    )
    elapsed = time.time() - t0
    log.info(f"Encoding done in {elapsed:.1f}s  ({n/elapsed:.0f} chunks/s)")

    assert embeddings.shape == (n, dim), (
        f"Unexpected embedding shape {embeddings.shape}, expected ({n}, {dim})"
    )
    return embeddings


def _encode_asymmetric(chunks: list[dict]) -> np.ndarray:
    """
    Encode chunks with an asymmetric dual-encoder model's Article-Encoder half
    (e.g. MedCPT — config.yaml embedding.model.article). Queries at retrieval
    time use the *different* Query-Encoder half (src/retrieve.py), which is why
    this can't reuse _encode_symmetric's single-model logic.

    Input is [title, chunk_text] pairs (the model was trained on PubMed
    title+abstract pairs; `title` is the document title already on every
    chunk, not invented content), pooled via the [CLS] token's last hidden
    state and L2-normalised by hand — this model has no sentence-transformers
    support and no built-in normalisation. Mirrors
    eval/compare-embedding-models/medcpt/build_index.py, the working prototype
    this was promoted from — see decisions.md D53/D56 for the full rationale.
    """
    torch = _require("torch", "torch")
    tf    = _require("transformers", "transformers")

    article_model_name = EMB_CFG["model"]["article"]
    log.info(f"Loading Article-Encoder: {article_model_name}")
    tokenizer = tf.AutoTokenizer.from_pretrained(article_model_name)
    model     = tf.AutoModel.from_pretrained(article_model_name)
    model.eval()

    pairs = [[c.get("title") or "", c["chunk_text"]] for c in chunks]
    n = len(pairs)
    log.info(f"Embedding {n:,} chunks  (dim=768, batch={ASYMMETRIC_BATCH_SZ})")

    t0 = time.time()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, n, ASYMMETRIC_BATCH_SZ):
            batch = pairs[i : i + ASYMMETRIC_BATCH_SZ]
            encoded = tokenizer(batch, truncation=True, padding=True, return_tensors="pt", max_length=512)
            embeds = model(**encoded).last_hidden_state[:, 0, :]   # [CLS] token
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.numpy())
    embeddings = np.concatenate(all_embeds, axis=0)
    elapsed = time.time() - t0
    log.info(f"Encoding done in {elapsed:.1f}s  ({n/elapsed:.0f} chunks/s)")

    assert embeddings.shape == (n, 768), f"Unexpected embedding shape {embeddings.shape}, expected ({n}, 768)"
    return embeddings


def build_dense_index(
    chunks: list[dict],
    index_type: str = INDEX_TYPE,
    output_path: Path = FAISS_PATH,
) -> None:
    """
    Encode all chunks and persist a FAISS index. Encoding path depends on
    config.yaml embedding.architecture — "symmetric" (default) or
    "asymmetric" (see _encode_symmetric / _encode_asymmetric above).

    Index type: "flat" (IndexFlatIP, exact) or "hnsw" (IndexHNSWFlat,
    approximate) per `index_type` — see decisions.md D4/D22 for the choice.
    Vectors are L2-normalised before insertion so IP == cosine similarity.
    """
    if index_type not in ("flat", "hnsw"):
        raise ValueError(f"index_type must be 'flat' or 'hnsw', got {index_type!r}")

    faiss = _require("faiss", "faiss-cpu")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    id_map_path = output_path.parent / "id_map.json"

    embeddings = _encode_symmetric(chunks) if ARCHITECTURE == "symmetric" else _encode_asymmetric(chunks)
    n, dim = embeddings.shape

    # Build the index (in-memory, then serialised)
    t_build0 = time.time()
    if index_type == "flat":
        log.info("Building FAISS IndexFlatIP (exact search)")
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
    else:
        log.info(f"Building FAISS HNSW32 index  (M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, "
                  f"efSearch={HNSW_EF_SEARCH})")
        index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        index.add(embeddings.astype(np.float32))
        index.hnsw.efSearch = HNSW_EF_SEARCH  # previously unset — FAISS default of 16 (D4)
    build_elapsed = time.time() - t_build0
    log.info(f"Index build took {build_elapsed:.2f}s")

    faiss.write_index(index, str(output_path))
    log.info(f"FAISS index saved → {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Save int → chunk_id mapping (FAISS only knows row positions)
    id_map = {i: c["chunk_id"] for i, c in enumerate(chunks)}
    id_map_path.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    log.info(f"ID map saved → {id_map_path}  ({len(id_map):,} entries)")

    # ── Smoke test ────────────────────────────────────────────────────────────
    log.info("Smoke test: querying index with first chunk...")
    query_vec = embeddings[0:1].astype(np.float32)
    distances, indices = index.search(query_vec, k=3)
    top_ids = [id_map[int(i)] for i in indices[0] if i >= 0]
    log.info(f"  Top-3 results: {top_ids}  (scores: {distances[0].tolist()})")
    assert int(indices[0][0]) == 0, "Smoke test failed: top-1 should be the query itself"
    log.info("  Smoke test PASSED ✓")


# ── Sparse (BM25) index ───────────────────────────────────────────────────────

def build_sparse_index(chunks: list[dict]) -> None:
    """
    Build a BM25Okapi index over chunk_text tokens and pickle it.

    BM25 is used in the hybrid retriever alongside FAISS for lexical recall
    on rare oncology terms (drug names, gene symbols, RECIST criteria, etc.)
    that dense embeddings sometimes underweight.
    """
    BM25Okapi = _require("rank_bm25", "rank-bm25").BM25Okapi

    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)

    texts = [c["chunk_text"] for c in chunks]
    log.info(f"Tokenising {len(texts):,} chunks for BM25...")

    # Simple whitespace tokeniser — good enough for retrieval;
    # lowercase to reduce vocabulary size
    tokenised = [t.lower().split() for t in texts]

    t0 = time.time()
    bm25 = BM25Okapi(tokenised)
    log.info(f"BM25 index built in {time.time()-t0:.1f}s")

    with open(BM25_PATH, "wb") as fh:
        pickle.dump(
            {
                "bm25":   bm25,
                "id_map": {i: c["chunk_id"] for i, c in enumerate(chunks)},
            },
            fh,
        )
    log.info(f"BM25 index saved → {BM25_PATH}  ({BM25_PATH.stat().st_size / 1e6:.1f} MB)")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 indexes for the Oncology RAG pipeline."
    )
    parser.add_argument("--dense-only",  action="store_true", help="Skip BM25 index")
    parser.add_argument("--sparse-only", action="store_true", help="Skip FAISS index")
    parser.add_argument("--index-type", choices=["flat", "hnsw"], default=None,
                         help="Override config.yaml embedding.index_type for this run")
    args = parser.parse_args()

    if args.dense_only and args.sparse_only:
        log.error("Cannot specify both --dense-only and --sparse-only")
        sys.exit(1)

    chunks = load_chunks()
    if not chunks:
        log.error("No chunks found — run  python src/chunk.py  first.")
        sys.exit(1)

    index_type = args.index_type or INDEX_TYPE

    if not args.sparse_only:
        log.info(f"─── Dense Index (FAISS {index_type}) ───────────────────────────────")
        build_dense_index(chunks, index_type=index_type)

    if not args.dense_only:
        log.info("─── Sparse Index (BM25) ────────────────────────────────────")
        build_sparse_index(chunks)

    log.info("Done. All indexes up to date.")


if __name__ == "__main__":
    main()

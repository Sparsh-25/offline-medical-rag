"""
compare_index_types.py — Empirical comparison of FAISS IndexFlatIP vs IndexHNSWFlat.

Pipeline position:
    extract.py → chunk.py → embed.py → [compare_index_types.py] (one-off analysis)

Why this exists
────────────────
docs/architecture.md / docs/phase0.md documented IndexHNSWFlat as the chosen
dense index type; docs/pre_phase2_plan.md later argued this was premature at
this corpus's scale and IndexFlatIP (exact search) should be used instead —
see decisions.md D4. Rather than pick a side from documentation debate alone,
this script measures both directly against the real corpus and reports:

  1. Recall — how often HNSW's approximate search agrees with FlatIP's exact
     search (top-1 agreement, self-retrieval recall@K, top-K set overlap).
  2. Speed — query latency for both, batched across every chunk in the corpus.
  3. Size — serialized index size for both.

Encodes the corpus once and builds both indexes from the same embeddings, so
the comparison isn't confounded by encoding-run variance.

Usage
─────
    python -m src.compare_index_types
    python -m src.compare_index_types --k 5      # compare at a different K

See decisions.md D4/D22 for the methodology write-up and result-driven choice.
"""

from __future__ import annotations

# ── macOS / PyTorch segfault guard (see embed.py) ─────────────────────────────
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import time

import numpy as np

from src.embed import (
    load_chunks, get_texts, _require,
    MODEL_ID, DEVICE, BATCH_SZ,
    HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compare(k: int = 10) -> None:
    faiss = _require("faiss", "faiss-cpu")
    SentenceTransformer = _require("sentence_transformers", "sentence-transformers").SentenceTransformer

    chunks = load_chunks()
    n = len(chunks)
    if not chunks:
        log.error("No chunks found — run  python -m src.chunk  first.")
        return
    log.info(f"Loaded {n} chunks")

    model = SentenceTransformer(MODEL_ID, device=DEVICE)
    texts = get_texts(chunks)
    dim_fn = getattr(model, "get_embedding_dimension", None) or getattr(model, "get_sentence_embedding_dimension")
    dim = dim_fn()

    t0 = time.time()
    embeddings = model.encode(
        texts, batch_size=BATCH_SZ, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    ).astype(np.float32)
    log.info(f"Encoded {n} chunks in {time.time() - t0:.1f}s (dim={dim})")

    # Build both indexes from the SAME embeddings — isolates the comparison
    # to the index structure itself, not encoding-run variance.
    flat = faiss.IndexFlatIP(dim)
    flat.add(embeddings)

    hnsw = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    hnsw.add(embeddings)
    hnsw.hnsw.efSearch = HNSW_EF_SEARCH

    log.info(
        f"Built IndexFlatIP and IndexHNSWFlat "
        f"(M={HNSW_M}, efConstruction={HNSW_EF_CONSTRUCTION}, efSearch={HNSW_EF_SEARCH}) "
        f"from {n} vectors"
    )

    # ── Recall: self-retrieval across every chunk in the corpus ──────────────
    # Query with each chunk's own embedding. FlatIP is exact search, so it's
    # ground truth by definition; the comparison measures how much HNSW's
    # approximation actually costs at this corpus's real scale.
    t0 = time.time()
    _, flat_I = flat.search(embeddings, k)
    flat_time = time.time() - t0

    t0 = time.time()
    _, hnsw_I = hnsw.search(embeddings, k)
    hnsw_time = time.time() - t0

    top1_agree = 0
    self_recall_hits = 0
    jaccard_sum = 0.0
    for i in range(n):
        flat_set = set(flat_I[i].tolist())
        hnsw_set = set(hnsw_I[i].tolist())
        inter = len(flat_set & hnsw_set)
        union = len(flat_set | hnsw_set)
        jaccard_sum += inter / union if union else 1.0
        if flat_I[i][0] == hnsw_I[i][0]:
            top1_agree += 1
        if i in hnsw_set:
            self_recall_hits += 1
    avg_jaccard = jaccard_sum / n

    log.info(f"\n=== Recall comparison (n={n} queries, K={k}) ===")
    log.info(f"Top-1 agreement (flat vs hnsw):     {top1_agree}/{n} ({100*top1_agree/n:.2f}%)")
    log.info(f"Self-in-top-{k} recall (hnsw):       {self_recall_hits}/{n} ({100*self_recall_hits/n:.2f}%)")
    log.info(f"Avg top-{k} set overlap (Jaccard):   {avg_jaccard:.4f}  (1.0 = identical results)")

    log.info(f"\n=== Speed comparison ({n} queries, batched) ===")
    log.info(f"Flat search: {flat_time*1000:.1f}ms total  ({flat_time*1000/n:.4f}ms/query)")
    log.info(f"HNSW search: {hnsw_time*1000:.1f}ms total  ({hnsw_time*1000/n:.4f}ms/query)")

    import os as _os
    faiss.write_index(flat, "/tmp/_cmp_flat.index")
    faiss.write_index(hnsw, "/tmp/_cmp_hnsw.index")
    flat_mb = _os.path.getsize("/tmp/_cmp_flat.index") / 1e6
    hnsw_mb = _os.path.getsize("/tmp/_cmp_hnsw.index") / 1e6
    _os.remove("/tmp/_cmp_flat.index")
    _os.remove("/tmp/_cmp_hnsw.index")

    log.info(f"\n=== Index size ===")
    log.info(f"Flat: {flat_mb:.2f} MB | HNSW: {hnsw_mb:.2f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FAISS IndexFlatIP vs IndexHNSWFlat on the real corpus."
    )
    parser.add_argument("--k", type=int, default=10, help="Top-K for recall comparison (default: 10)")
    args = parser.parse_args()
    compare(k=args.k)


if __name__ == "__main__":
    main()

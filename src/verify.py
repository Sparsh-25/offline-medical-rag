"""
verify.py — End-to-end sanity checks for the Oncology RAG index.

Run this AFTER embed.py has successfully built both indexes:
    python src/verify.py

Four checks, in order:
  1. Chunk quality       — metadata completeness, token distribution, content types
  2. Embedding sanity   — norm check, same-doc vs. cross-doc similarity
  3. Dense smoke test   — 6 clinical queries → FAISS, inspect top-3 per query
  4. Sparse + RRF       — BM25 vs. dense vs. fused result comparison

If all four look reasonable, Phase 0 (index build) is complete.
"""

from __future__ import annotations

# ── macOS / PyTorch segfault guard ────────────────────────────────────────────
# MUST be set before sentence_transformers (or torch) is imported.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# ─────────────────────────────────────────────────────────────────────────────

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# ── Root-relative config (works regardless of CWD) ───────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
cfg   = yaml.safe_load((_ROOT / "config.yaml").read_text())

EMB_CFG = cfg["embedding"]

# ── Absolute index paths ──────────────────────────────────────────────────────
CHUNKS_PATH    = _ROOT / cfg["paths"]["chunks"]
FAISS_PATH     = _ROOT / cfg["paths"]["faiss_index"]
BM25_PATH      = _ROOT / cfg["paths"]["bm25_index"]
CHUNK_MAP_PATH = FAISS_PATH.parent / "chunk_map.json"


# ── Load everything ───────────────────────────────────────────────────────────

def load_all() -> tuple[list[dict], Any, dict, dict, np.ndarray]:
    """
    Returns (chunks, faiss_index, bm25_payload, chunk_map, embeddings).

    Embeddings are reconstructed directly from the HNSW index so we don't
    need a separate embeddings.npy file — IndexHNSWFlat stores raw vectors.
    """
    # 1. Chunks JSONL
    chunks: list[dict] = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # 2. FAISS index
    if not FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_PATH}\n"
            "Run  python src/embed.py  first."
        )
    index = faiss.read_index(str(FAISS_PATH))

    # 3. BM25 pickle
    if not BM25_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_PATH}\n"
            "Run  python src/embed.py  first."
        )
    with open(BM25_PATH, "rb") as fh:
        bm25_payload = pickle.load(fh)

    # 4. Chunk map (chunk_id → full chunk dict)
    chunk_map: dict = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))

    # 5. Reconstruct embeddings from HNSW index (no separate .npy needed)
    #    IndexHNSWFlat keeps all raw vectors in memory — reconstruct_n is safe.
    n   = index.ntotal
    dim = index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        index.reconstruct(i, embeddings[i])

    return chunks, index, bm25_payload, chunk_map, embeddings


# ── Check 1: Chunk quality ────────────────────────────────────────────────────

def check_chunk_quality(chunks: list[dict]) -> None:
    """
    WHY THIS MATTERS
    ────────────────
    Before you trust retrieval results, you need to know the raw material is
    clean. Bad chunks — missing metadata, stub-sized text, oversized walls of
    prose — will silently degrade LLM answer quality because the ranker can't
    distinguish them from good chunks.

    WHAT WE CHECK
    ─────────────
    • Chunks-per-doc  — every document got split (no doc silently skipped)
    • Metadata completeness — pub_year / source_type are used at query time for
      recency weighting and source-type filtering; None here breaks those features
    • Token distribution — min/median/max and the tail counts (< 80 = stub,
      > 512 = BGE-base context overflow)
    • Content-type breakdown — prose should dominate; many tables may indicate
      the markdown extraction is treating prose as tables
    • Recency weights — one distinct value per doc; if all docs share the same
      weight, compute_recency_weight() silently failed
    """
    print("\n" + "=" * 60)
    print("CHECK 1: Chunk quality")
    print("=" * 60)

    # Per-doc breakdown
    by_doc = Counter(c["doc_id"] for c in chunks)
    print(f"\nTotal chunks: {len(chunks)}")
    print("\nChunks per document:")
    for doc_id, count in sorted(by_doc.items()):
        sample = next(c for c in chunks if c["doc_id"] == doc_id)
        yr = sample.get("pub_year", "?")
        st = sample.get("source_type", "?")
        rw = sample.get("recency_weight", "?")
        print(f"  {doc_id}: {count:3d} chunks  (year={yr}, type={st}, recency={rw})")

    # Metadata completeness
    missing_year = [c for c in chunks if not c.get("pub_year")]
    missing_type = [c for c in chunks if not c.get("source_type")]
    print(f"\nMetadata completeness:")
    print(f"  Missing pub_year:    {len(missing_year):3d}", "✓" if not missing_year else "  ← fix meta.json")
    print(f"  Missing source_type: {len(missing_type):3d}", "✓" if not missing_type else "  ← fix meta.json")

    # Token distribution
    tokens = [c["token_count"] for c in chunks]
    stubs     = sum(1 for t in tokens if t < 80)
    oversized = sum(1 for t in tokens if t > 512)
    print(f"\nToken distribution:")
    print(f"  Min:    {min(tokens)}")
    print(f"  Median: {int(np.median(tokens))}")
    print(f"  Mean:   {int(np.mean(tokens))}")
    print(f"  Max:    {max(tokens)}")
    print(f"  Under 80 tokens:  {stubs:3d}", "✓" if stubs == 0 else "  ← stub chunks (should be 0 after min_chunk_tokens filter)")
    print(f"  Over 512 tokens:  {oversized:3d}", "✓" if oversized == 0 else "  ← BGE-base will truncate these; check subsplitter")

    # Content type
    by_type = Counter(c["content_type"] for c in chunks)
    print(f"\nContent types: {dict(by_type)}")
    print(f"  Expected: prose >> table > figure_caption")

    # Recency weight spread
    weights        = [c["recency_weight"] for c in chunks]
    unique_weights = sorted(set(weights))
    print(f"\nDistinct recency weights: {unique_weights}")
    print(f"  Expected: one value per document ({len(by_doc)} docs → should be ≤ {len(by_doc)} values)")


# ── Check 2: Embedding sanity ─────────────────────────────────────────────────

def check_embedding_sanity(chunks: list[dict], embeddings: np.ndarray) -> None:
    """
    WHY THIS MATTERS
    ────────────────
    FAISS HNSW with inner-product metric is only equivalent to cosine similarity
    when vectors are unit-normalised. If normalisation failed, similarity scores
    are meaningless and you will see retrieval results that make no clinical sense.

    The same-doc vs. cross-doc test is a sanity check on embedding quality:
    sentences within the same medical paper should cluster together, because they
    share terminology, entity co-references, and domain context. If cross-doc
    pairs score higher than same-doc pairs, something went very wrong (wrong model,
    wrong input preprocessing, corrupted chunks).

    WHAT WE CHECK
    ─────────────
    • Vector norms — all should be 1.0 ± 0.01 after normalize_embeddings=True
    • Same-doc vs. cross-doc avg cosine sim — same-doc should be higher
    """
    print("\n" + "=" * 60)
    print("CHECK 2: Embedding sanity")
    print("=" * 60)

    # Norm check
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\nVector norms (should all be ~1.0 after L2 normalisation):")
    print(f"  Min: {norms.min():.4f}  Max: {norms.max():.4f}  Mean: {norms.mean():.4f}")
    if norms.min() < 0.99 or norms.max() > 1.01:
        print("  ⚠ WARNING: norms not ~1.0 — normalisation failed in embed.py")
    else:
        print("  ✓ All vectors properly normalised")

    # Same-doc vs. cross-doc similarity
    print(f"\nSame-doc vs. cross-doc cosine similarity (sample of 100 pairs each):")
    rng    = np.random.default_rng(42)
    doc_ids = [c["doc_id"] for c in chunks]
    n      = len(chunks)

    same_sims: list[float] = []
    diff_sims: list[float] = []
    attempts = 0
    while (len(same_sims) < 100 or len(diff_sims) < 100) and attempts < 10_000:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            attempts += 1
            continue
        sim = float(embeddings[i] @ embeddings[j])
        if doc_ids[i] == doc_ids[j]:
            same_sims.append(sim)
        else:
            diff_sims.append(sim)
        attempts += 1

    if same_sims and diff_sims:
        print(f"  Same-doc pairs  (n={len(same_sims):3d}) avg cosine:  {np.mean(same_sims):.4f}")
        print(f"  Cross-doc pairs (n={len(diff_sims):3d}) avg cosine:  {np.mean(diff_sims):.4f}")
        if np.mean(same_sims) > np.mean(diff_sims):
            print("  ✓ Same-doc chunks more similar than cross-doc — embedding is healthy")
        else:
            print("  ⚠ Cross-doc similarity ≥ same-doc — check model / chunk preprocessing")
    else:
        print("  ⚠ Could not collect enough pairs; increase attempts limit or check doc count")


# ── Check 3: Dense retrieval smoke test ──────────────────────────────────────

def check_dense_retrieval(
    chunks:    list[dict],
    index:     Any,           # faiss.IndexFlatIP at runtime (decisions.md D22)
    chunk_map: dict,
    model:     SentenceTransformer,
) -> None:
    """
    WHY THIS MATTERS
    ────────────────
    Embedding quality doesn't guarantee retrieval quality — your FAISS index
    could be corrupted, the query encoding path could differ from the passage
    path, or the BGE instruction prefix could be wrong. Running real clinical
    queries and checking which doc surfaces at rank-1 catches all of these.

    Two kinds of check, both needed:
    • Reachable queries (6, one per corpus doc) — should retrieve their target
      document with a high score. Catches "retrieval doesn't work."
    • Unreachable queries (3) — about real oncology topics that this 6-doc
      corpus does not cover. Should score low / not confidently match anything.
      Catches "retrieval falsely claims a match" (semantic over-triggering),
      which a reachable-only test can never detect — it would look identical
      to a reachable query returning a mediocre-but-plausible match.

    WHAT WE CHECK
    ─────────────
    • Top-3 results per query, with score and doc_id
    • Reachable: ✓/✗ whether the expected document appears in top-3
    • Unreachable: ✓/⚠ whether the top-1 score stays below the confident-match
      threshold (>0.5 — see below); a high score here is a false-positive risk
    • Score threshold: >0.5 is a good match for BGE-base with normalised vectors
      (cosine similarity — vectors are L2-normalised, so FAISS inner product
      *is* cosine similarity here, not a different metric)
    • Clinical coherence: the answer should make sense (e.g. ribociclib → MONALEESA-2)

    HOW TO INTERPRET
    ────────────────
    A ✗ on every reachable query usually means: wrong query encoding (missing BGE
    instruction prefix), index built on different chunks than currently in
    chunks.jsonl, or normalized flag not set consistently between index build and
    query time. A ⚠ on an unreachable query means retrieval is over-confident on
    out-of-corpus topics — worth investigating before trusting low scores elsewhere
    as "no evidence found."
    """
    print("\n" + "=" * 60)
    print("CHECK 3: Dense retrieval smoke test")
    print("=" * 60)

    GOOD_MATCH_THRESHOLD = 0.5  # cosine similarity — see docstring

    # 6 reachable queries (one per corpus document, full coverage) + 3
    # unreachable queries (real oncology topics this corpus does not cover —
    # negative controls, expected_fragment=None). Previously 3 of the
    # "reachable" queries referenced doc_id fragments ("meta", "destiny",
    # "waks") that don't exist anywhere in this project's 6-document corpus,
    # making them fail regardless of retrieval quality (decisions.md D23).
    test_queries = [
        # (query_text, expected_doc_id_substring_or_None)
        ("What is the median overall survival for ribociclib plus letrozole?",
         "monaleesa"),
        ("First-line treatment recommendation for HR positive HER2 negative metastatic breast cancer",
         "nccn"),
        ("Multi-omics biomarkers for precision medicine treatment selection in breast cancer",
         "molbiosci"),
        ("Cost-effectiveness of trastuzumab deruxtecan for HER2-low breast cancer",
         "pubhealth"),
        ("Neoadjuvant versus adjuvant chemotherapy timing and survival outcomes in HR positive breast cancer",
         "sci_reports"),
        ("HER2 immunohistochemistry IHC scoring criteria and FISH testing algorithm for breast cancer",
         "asco"),
        # ── Unreachable (negative controls) — not covered by this corpus ──
        ("Immune checkpoint inhibitor pembrolizumab response rate in non-small cell lung cancer",
         None),
        ("CAR-T cell therapy outcomes in relapsed acute lymphoblastic leukemia",
         None),
        ("Colorectal cancer screening guidelines and polypectomy surveillance intervals",
         None),
    ]

    reachable_pass = reachable_total = 0
    unreachable_pass = unreachable_total = 0

    for query, expected_fragment in test_queries:
        q_emb = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores, indices = index.search(q_emb.astype(np.float32), k=3)

        is_unreachable = expected_fragment is None
        print(f'\n  Query: "{query[:70]}"')
        print(f"  Expected: {'*unreachable — should NOT confidently match*' if is_unreachable else f'*{expected_fragment}*'}")

        top1_score = float(scores[0][0]) if len(scores[0]) else 0.0
        if is_unreachable:
            unreachable_total += 1
            ok = top1_score < GOOD_MATCH_THRESHOLD
            unreachable_pass += int(ok)
            print(f"  {'✓' if ok else '⚠'} top-1 score={top1_score:.3f} "
                  f"({'below' if ok else 'ABOVE'} the {GOOD_MATCH_THRESHOLD} confident-match threshold)")
        else:
            reachable_total += 1
            top3_docs = [chunks[idx]["doc_id"] for idx in indices[0] if 0 <= idx < len(chunks)]
            ok = any(expected_fragment.lower() in d.lower() for d in top3_docs)
            reachable_pass += int(ok)

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # Guard against FAISS -1 (no result — should not happen with a full index but be safe)
            if idx < 0 or idx >= len(chunks):
                print(f"  [{rank+1}] FAISS returned invalid index {idx} (index has {len(chunks)} vectors)")
                continue
            chunk = chunks[idx]
            if is_unreachable:
                hit = "·"  # no expected doc to check against — informational only
            else:
                hit = "✓" if expected_fragment.lower() in chunk["doc_id"].lower() else "✗"
            print(f"  [{rank+1}] {hit} score={score:.3f}  doc={chunk['doc_id']}")
            print(f"       {chunk.get('section_h1')} › {chunk.get('section_h2')}")
            print(f"       {chunk['chunk_text'][:120]}…")

    print(f"\n  ── Summary: reachable {reachable_pass}/{reachable_total} correct, "
          f"unreachable {unreachable_pass}/{unreachable_total} correctly low-confidence ──")


# ── Check 4: BM25 smoke test + RRF spot-check ────────────────────────────────

def check_sparse_and_rrf(
    chunks:       list[dict],
    index:        Any,           # faiss.IndexHNSWFlat at runtime
    bm25_payload: dict,
    chunk_map:    dict,
    model:        SentenceTransformer,
) -> None:
    """
    WHY THIS MATTERS
    ────────────────
    Dense embeddings are powerful for semantic matching but can underweight rare,
    exact tokens — drug names like "ribociclib", trial codes like "MONALEESA-2",
    gene symbols like "ESR1 D538G". BM25 (sparse TF-IDF variant) is the opposite:
    excellent at exact lexical matches, poor at synonymy. Reciprocal Rank Fusion
    (RRF) combines both ranked lists without needing a learned weight, giving you
    the best of both retrieval paradigms.

    WHAT WE CHECK
    ─────────────
    • BM25 top-1 on exact-term query  (should beat dense on trial names)
    • Dense top-1 on semantic query   (should beat BM25 on conceptual queries)
    • RRF top-3 combining both lists  (should be better than either alone)
    """
    print("\n" + "=" * 60)
    print("CHECK 4: Sparse BM25 + RRF spot-check")
    print("=" * 60)

    bm25     = bm25_payload["bm25"]
    # BM25 payload stores id_map: {int_position → chunk_id}
    id_map   = bm25_payload["id_map"]   # BUG FIX: was bm25_payload["chunk_ids"]

    def bm25_search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
        tokens     = query.lower().split()
        scores_arr = bm25.get_scores(tokens)
        top_idxs   = np.argsort(scores_arr)[::-1][:top_k]
        return [(id_map[int(i)], float(scores_arr[i])) for i in top_idxs]

    def dense_search(query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores_arr, idxs = index.search(q_emb.astype(np.float32), k=top_k)
        results = []
        for j, idx in enumerate(idxs[0]):
            if idx < 0 or idx >= len(chunks):   # guard against FAISS -1 sentinel
                continue
            results.append((chunks[idx]["chunk_id"], float(scores_arr[0][j])))
        return results

    def rrf(dense_results: list, sparse_results: list, k: int = 60) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion — no learned weights needed."""
        scores: dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (chunk_id, _) in enumerate(sparse_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    test_queries = [
        ("ribociclib MONALEESA-2 hazard ratio",        "exact term  → BM25 should win"),
        ("CDK4/6 inhibitor endocrine therapy benefit", "semantic    → dense should win"),
    ]

    for query, annotation in test_queries:
        dense  = dense_search(query)
        sparse = bm25_search(query)
        fused  = rrf(dense, sparse)

        print(f'\n  Query: "{query}"  [{annotation}]')
        if dense:
            print(f"  Dense  top-1: {dense[0][0]}  score={dense[0][1]:.3f}")
        if sparse:
            print(f"  Sparse top-1: {sparse[0][0]}  score={sparse[0][1]:.3f}")
        print(f"  RRF top-3:")
        for cid, rrf_score in fused[:3]:
            c = chunk_map.get(cid, {})
            print(f"    {cid}  rrf_score={rrf_score:.4f}")
            print(f"    {c.get('section_h1')} › {c.get('section_h2')}")
            print(f"    {c.get('chunk_text', '')[:100]}…")


# ── Run all checks ────────────────────────────────────────────────────────────

def run_all_checks() -> None:
    print("Loading indexes...")
    chunks, index, bm25_payload, chunk_map, embeddings = load_all()
    print(f"  {len(chunks):,} chunks  |  FAISS ntotal={index.ntotal}  |  BM25 corpus={len(bm25_payload['id_map'])}")

    # Load model ONCE — both check_dense_retrieval and check_sparse_and_rrf need it.
    # Loading it twice wastes ~5 seconds and ~450 MB of RAM.
    print(f"\nLoading embedding model: {EMB_CFG['model']}  (device={EMB_CFG['device']})")
    model = SentenceTransformer(EMB_CFG["model"], device=EMB_CFG["device"])

    check_chunk_quality(chunks)
    check_embedding_sanity(chunks, embeddings)
    check_dense_retrieval(chunks, index, chunk_map, model)
    check_sparse_and_rrf(chunks, index, bm25_payload, chunk_map, model)

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("If all four checks look reasonable, Phase 0 is done.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
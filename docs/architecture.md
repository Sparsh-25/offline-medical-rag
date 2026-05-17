# Architecture — Offline Oncology RAG Agent

## Design principles

1. **Offline-first.** Every component after metadata extraction runs with no internet. Models are pre-downloaded. No cloud inference in the retrieval or generation path.
2. **No hallucination by construction.** Every generated claim is checked against retrieved context via NLI before being shown to the user. If entailment score is below threshold, the bot abstains.
3. **Metadata-driven retrieval.** Every chunk carries full document metadata baked in at build time — pub_year, source_type, recency_weight, drug_focus, line_of_therapy. This drives source prioritization in the prompt, not just semantic similarity.
4. **Tiered complexity.** Each phase adds exactly one new capability. Nothing is skipped. Each phase gate requires measurable improvement on RAGCare-QA before advancing.

---

## Full pipeline (steady state — Phase 6 target)

```
User query
    │
    ▼
DistilBERT router          ← Phase 3 (query type classifier)
    │
    ├─ factoid   → dense-only retrieval
    ├─ clinical  → hybrid retrieval
    ├─ compare   → hybrid + wider top-k
    └─ ambiguous → hybrid + lower CRAG threshold
    │
    ▼
Hybrid retrieval
    ├─ Dense:  BGE-base-en-v1.5 → FAISS HNSW → top-20     ← Phase 0
    └─ Sparse: BM25Okapi        → rank_bm25  → top-20     ← Phase 0
    │
    ▼
RRF fusion (k=60)                                          ← Phase 1
    │
    ▼
Cross-encoder reranking                                    ← Phase 2
    ms-marco-MiniLM-L-6-v2 → top-20 → top-5
    │
    ▼
CRAG scorer                                                ← Phase 2
    NLI entailment (query, chunk) → discard below threshold
    │
    ▼
ACC-RAG context compression                                ← Phase 4
    Sentence-level relevance scoring → drop low-signal sentences
    Target: 40-60% context window reduction
    │
    ▼
Prompt assembly                                            ← Phase 1
    [SOURCE N — recency: X.XX — type: rct — Journal Year]
    chunk_text
    ...
    │
    ▼
LLM generation                                             ← Phase 1
    Mistral-Small-3.2 or LLaMA-3-8B-Instruct
    Q8 GGUF (Phase 1) → Q4_K_M via SpinQuant (Phase 4)
    │
    ▼
NLI faithfulness verification                              ← Phase 2 (soft) / Phase 5 (hard)
    DeBERTa-v3-small-mnli
    (answer_sentence, source_chunk) → entail | neutral | contradict
    Contradict → re-retrieve or abstain
    │
    ▼
Citation attachment                                        ← Phase 5
    Every claim → chunk_id → source passage
    │
    ▼
Response with inline citations
```

---

## Current architecture (Phase 0 — what actually exists)

```
data/raw/*.pdf
    │
    ▼
meta_builder.py
    PDF text → DOI regex
    → CrossRef API (title, journal, pub_year, pub_month)
    → PubMed/Entrez API (abstract, MeSH terms)
    → Groq LLM (source_type, population, line_of_therapy, drugs)
    → Defensive merge (offline ground truth always wins)
    → data/extracted/<doc_id>/meta.json
    │
    ▼
extract.py (Docling)
    → Markdown (headers, GFM tables, figure placeholders)
    → figures/*.png (size-filtered, content-deduplicated)
    → extraction_log.json
    │
    ▼
caption.py (LLaVA:7b via Ollama)
    For each saved figure:
        → Pydantic VisualExtraction (archetypes, cohorts, HR, p-value, data_points, ...)
        → build_metadata() → structured filter dict
        → build_enriched_text() → extra markdown for tables/flowcharts/pathways
        → inject into .md as ![{summary}](figures/name.png) + <!-- FIGURE_METADATA: {...} -->
    Status: mostly failing on current corpus (see known bugs)
    │
    ▼
chunk.py
    split_markdown_by_headers() — split only at H1/H2/H3
    subsplit() — paragraph-boundary sub-split if > 512 tokens
    Chunk dataclass — merges meta.json fields + structural fields + chunk_text
    → data/chunks/chunks.jsonl
    → index/chunk_map.json
    │
    ▼
embed.py
    Dense: BGE-base-en-v1.5 → normalize_embeddings=True → FAISS IndexHNSWFlat(M=32)
    Sparse: BM25Okapi → lowercase whitespace tokenization
    → index/faiss.index
    → index/bm25.pkl
    → index/id_map.json
    │
    ▼
verify.py
    Check 1: chunk quality (counts, metadata completeness, token distribution)
    Check 2: embedding sanity (norms, same-doc vs cross-doc cosine)
    Check 3: dense smoke test (6 clinical queries, top-3 inspection)
    Check 4: BM25 + RRF spot-check (exact vs semantic query comparison)
```

---

## Chunk schema (every field the LLM and retriever see)

```python
@dataclass
class Chunk:
    # Identity
    chunk_id: str              # "{doc_id}_chunk_{N:04d}"
    doc_id:   str

    # From meta.json (baked in at chunk.py build time)
    title:             str | None
    pub_year:          int | None
    pub_month:         int | None
    recency_weight:    float          # 0.5 (2015 or earlier) → 1.0 (2024 or later)
    source_type:       str | None     # rct | guideline | meta_analysis | review | mechanism | retrospective | staging
    journal:           str | None
    doi:               str | None
    guideline_version: str | None
    cancer_type:       str | None
    cancer_subtype:    list | None    # ["HR+", "HER2-"]
    population:        str | None
    line_of_therapy:   str | None     # first_line | second_line | third_line_plus | adjuvant | neoadjuvant | all
    drug_focus:        list | None    # ["ribociclib", "letrozole"]
    drug_class:        str | None

    # From markdown structure (set by chunk.py header walk)
    section_h1:   str | None
    section_h2:   str | None
    section_h3:   str | None
    content_type: str              # prose | table | figure_caption
    page_hint:    int | None       # not yet implemented

    # Content
    chunk_text:  str
    token_count: int               # rough estimate: word_count × 1.3
```

---

## meta.json schema

```json
{
  "doc_id":            "monaleesa2_subanalysis_2018",
  "title":             "...",
  "pub_year":          2018,
  "pub_month":         2,
  "source_type":       "rct",
  "journal":           "Breast Cancer Research and Treatment",
  "doi":               "10.1007/s10549-017-4518-8",
  "guideline_version": null,
  "cancer_type":       "Breast",
  "cancer_subtype":    ["HR+", "HER2-"],
  "population":        "Postmenopausal women with HR+, HER2- advanced breast cancer",
  "line_of_therapy":   "first_line",
  "drug_focus":        ["ribociclib", "letrozole"],
  "drug_class":        "CDK4/6 inhibitor",
  "notes":             "MONALEESA-2 sub-analysis ...",
  "_needs_review":     false,
  "_llm_confidence":   1.0,
  "_conflicts":        []
}
```

`_needs_review: false` locks the file — meta_builder.py will never overwrite it on re-runs.

---

## Recency weighting

```
pub_year ≤ 2015 → recency_weight = 0.5
pub_year = 2020 → recency_weight = 0.75
pub_year ≥ 2024 → recency_weight = 1.0
pub_year = None → recency_weight = 0.7 (neutral fallback)
```

Injected into every prompt source header so the LLM can reason about evidence currency:

```
[SOURCE 1 — recency: 1.0 — type: guideline — NCCN Breast v2.2026]
...
[SOURCE 2 — recency: 0.65 — type: rct — MONALEESA-2, NEJM 2016]
...
```

LLM system instruction: "When sources conflict on current practice, prefer higher recency_weight. When a guideline cites a trial, the trial source provides supporting statistical evidence."

---

## Figure captioning architecture (caption.py)

```
extraction_log.json (figures list)
    │
    for each figure:
    │
    ▼
Ollama VLM (currently llava:7b, planned: llava:13b on GPU)
    input:  base64 PNG + medical figure prompt + docling_context (Docling's extracted caption)
    output: JSON → VisualExtraction (Pydantic)
    │
    ▼
_sanitise_llava_output()     ← coerces LLM type errors before Pydantic validation
VisualExtraction(**data)     ← Pydantic validation
apply_safe_validations()     ← HR > 0, p-value in [0,1], CI lower < upper
    │
    ▼
build_metadata()             ← structured filter dict (has_hr, is_significant, is_survival_curve, ...)
get_extraction_status()      ← success | partial | failed
build_enriched_text()        ← GFM tables from table_rows, flow steps, pathway relations
    │
    ▼
inject_captions_into_markdown()
    Handles 5 placeholder formats:
        html_comment   <!-- image ... -->
        empty_alt      ![]( path )
        image_alt      ![image]( path )
        fallback_alt   ![Figure from oncology document: ...]( path )
        any_alt_figure catch-all
    Output: ![{VLM summary}](figures/name.png)\n<!-- FIGURE_METADATA: {...} -->
```

---

## Index architecture (embed.py)

| Component | Technology | Config |
|---|---|---|
| Dense index | FAISS IndexHNSWFlat, M=32, efConstruction=200 | Inner-product metric (cosine via L2-norm) |
| Sparse index | BM25Okapi (rank_bm25) | Lowercase whitespace tokenization |
| Embedding model | BAAI/bge-base-en-v1.5 | 438MB, dim=768, 512 token limit |
| ID mapping | index/id_map.json | int position → chunk_id (FAISS), int → chunk_id (BM25) |
| Chunk lookup | index/chunk_map.json | chunk_id → full chunk dict |

HNSW is used instead of IndexFlatIP because it scales to Tier 1/2 corpus sizes (50–500 docs) without rewriting the index. At 6 documents and 610 chunks the difference is negligible, but the upgrade is free.

---

## GPU strategy

All CPU-only work runs locally on laptop.
GPU-heavy work is submitted to university on-demand cluster.

| Task | Compute | When |
|---|---|---|
| Docling extraction | CPU | Phase 0 (done) |
| LLaVA:7b captioning | CPU (failing) | Phase 0 (pending fix) |
| BGE-base embedding | CPU | Phase 0 (done) |
| LLaVA:13b captioning | GPU | Phase 0 fix / Phase 1 |
| E5-mistral-7b embedding | GPU | Phase 6 |
| SpinQuant quantization | GPU | Phase 4 |
| DistilBERT fine-tuning | GPU | Phase 3 |
| LLM inference (Q4 GGUF) | CPU (llama.cpp) | Phase 1+ |

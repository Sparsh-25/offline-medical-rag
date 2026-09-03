# Offline Oncology RAG Agent

**A citation-grounded, offline retrieval system for breast cancer clinical literature.**
Given a natural-language question, the system returns the specific passages from a curated document corpus that
answer it, each one tagged with its source, section, publication year, and evidence type — so a claim can always
be traced back to where it came from.

This is a capstone project. See `decisions.md` (why things were built the way they were) and `state.md` (what's
done, in progress, and blocked) for the authoritative, up-to-date project record — this README is a usage guide,
not the design log.

---

## Where the project stands

**Built and evaluated (stages 1-6):** metadata extraction, document parsing, figure captioning, chunking,
indexing, and hybrid retrieval. Given a query, the system returns ranked, source-attributed passages — it does
not yet compose a synthesized natural-language answer on its own.

**Planned, not yet built (stages 7-9):** a retrieval quality gate / reranking step, LLM-based answer generation,
and an NLI-based output-verification step that checks a generated answer's faithfulness before it's shown.

Corpus: 12 breast cancer documents (NCCN/ASCO/ESMO guidelines, landmark RCTs, meta-analyses) → 1,473 indexed
chunks. Retrieval is evaluated against a 109-entry hand-verified query set (`eval/gold.jsonl`); at the current
tuned configuration the production model (MedCPT) reaches MRR 0.498 / R@5 0.677. See `eval/eval_retrieval.py`
and `decisions.md` (D45-D57) for the full comparison across four embedding models and the parameter sweeps
behind every default in `config.yaml`.

---

## Pipeline

```
data/raw/*.pdf
      │
      ▼
[1] Metadata Extraction  ──  meta_builder.py   ──→ data/extracted/<doc_id>/meta.json
      │   offline ground truth → CrossRef → PubMed/Entrez → LLM (Groq/Ollama), defensive merge
      ▼
[2] Document Parsing      ──  extract.py       ──→ data/extracted/<doc_id>/<doc_id>.md (+ figures/, tables)
      │   Docling: PDF → structured markdown, preserving headers/tables/figure placeholders
      ▼
[3] Figure Captioning     ──  caption.py        ──→ captions injected into the markdown above
      │   VLM (LLaVA via Ollama). Built and functional, but its output is currently EXCLUDED from
      │   the live corpus (config.yaml → chunking.include_figure_captions: false) — confirmed VLM
      │   fabrications found during gold-set construction (decisions.md D14/D30/D33/D34/D35).
      ▼
[4] Chunking               ──  chunk.py         ──→ data/chunks/chunks.jsonl
      │   header-aware split (H1/H2/H3), paragraph→line→sentence cascading fallback for oversized
      │   sections, junk-section filtering, metadata baked into every chunk
      ▼
[5] Indexing                ──  embed.py         ──→ index/faiss.index, index/bm25.pkl, index/*.json
      │   dense (FAISS IndexFlatIP, MedCPT asymmetric dual-encoder) + sparse (BM25Okapi)
      ▼
[6] Retrieval                ──  retrieve.py      ──→ ranked, source-attributed passages
          dense / hybrid (weighted fusion) / RRF, configurable diversity cap per document

  ─────────────────────────  planned, not yet built  ─────────────────────────
[7] Retrieval Quality Gate   ──  cross-encoder reranker + CRAG-style relevance filter
[8] Answer Generation        ──  quantized local LLM (GGUF), cites the passages that survive [7]
[9] Output Verification      ──  NLI faithfulness/relevance check → confidence score or abstention
```

`src/verify.py` runs a 4-check sanity suite (chunk quality, embedding sanity, dense retrieval smoke test,
BM25+RRF comparison) against whatever index is currently built — the fastest way to confirm the pipeline is in
a consistent, working state after any change.

---

## Setup

```bash
git clone <repo>
cd "Oncology Agent"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com) — used only for the one-time
# metadata-extraction step, never at query time.
cp .secrets/secrets.yaml.example .secrets/secrets.yaml
# Edit .secrets/secrets.yaml and paste your key
```

---

## Running the pipeline

```bash
# 1. Metadata extraction — all PDFs in data/raw/, or a single one
python -m src.meta_builder
python -m src.meta_builder --pdf data/raw/my_paper.pdf
python -m src.meta_builder --no-llm          # CrossRef + Entrez only, no API key needed

# 2. Document parsing (Docling)
python -m src.extract                        # all PDFs
python -m src.extract --pdf data/raw/my_paper.pdf --force   # re-parse one, ignoring cached output

# 3. Figure captioning (optional — output currently excluded from the live corpus, see above)
python -m src.caption                        # all documents
python -m src.caption --doc-id my_doc_id     # resume/target specific documents

# 4. Chunking
python -m src.chunk

# 5. Indexing
python -m src.embed                          # both dense (FAISS) and sparse (BM25)
python -m src.embed --dense-only             # rebuild FAISS only, e.g. after an embedding-model change
python -m src.embed --index-type hnsw        # override config.yaml's index type for a one-off comparison

# 6. Sanity check the built index
python -m src.verify

# 7. Query retrieval directly
python -m src.retrieve --query "What is the ORR for T-DXd in HER2-low breast cancer?" --mode hybrid --top-k 5
```

Each stage reads the previous stage's output from disk, so any stage can be re-run in isolation after an
upstream fix — there is currently no automated staleness check between `chunks.jsonl` and the built indexes
(see `state.md` Technical Debt), so re-run `embed.py` after any `chunk.py` change.

### Adding new papers

1. Drop the PDF into `data/raw/`.
2. Optionally add a cleaner folder name in `DOC_ID_MAP` (duplicated in both `src/meta_builder.py` and
   `src/extract.py` — keep both in sync).
3. Optionally add hardcoded ground truth in `OFFLINE_GROUND_TRUTH` (`src/meta_builder.py`) for papers where
   guaranteed-correct clinical fields matter more than LLM inference.
4. Run the pipeline stages above in order, ending with `embed.py` to rebuild the indexes.

```python
DOC_ID_MAP = {
    "my-new-pdf-filename-stem": "clean_doc_id_2024",
}
OFFLINE_GROUND_TRUTH = {
    "clean_doc_id_2024": {
        "source_type":    "rct",
        "cancer_subtype": ["HR+", "HER2-"],
        "drug_focus":     ["drug_a", "drug_b"],
    }
}
```

---

## Retrieval configuration

Set in `config.yaml → retrieval`, tuned by direct measurement against `eval/gold.jsonl` (not defaults or
reputation — see `decisions.md` D42-D57 for every sweep behind these numbers):

| Setting | Current value | Notes |
|---|---|---|
| `mode` | `hybrid` | `dense` \| `hybrid` (weighted score fusion) \| `rrf` (reciprocal rank fusion) |
| `candidate_pool` | 50 | results each of dense/sparse pulls before fusion |
| `hybrid_alpha` | 0.5 | weight on dense score in hybrid mode; sparse gets `1 - hybrid_alpha` |
| `rrf_k` | 60 | RRF rank-damping constant |
| `max_per_doc` | 5 | diversity cap — prevents the largest document (NCCN, ~78% of the corpus) from dominating results |
| `confidence_threshold` | 0.647 | dense-mode cosine cutoff; diagnostic only (`src/verify.py`), not wired into abstention logic yet |

Embedding model: `config.yaml → embedding` — currently MedCPT (`ncbi/MedCPT-Query-Encoder` +
`ncbi/MedCPT-Article-Encoder`), an asymmetric dual-encoder chosen after a full parameter-sweep comparison
against BGE-base, BGE-M3, and PubMedBERT. All four candidates remain independently rebuildable under
`eval/compare-embedding-models/`.

---

## Evaluation

`eval/gold.jsonl` — 109 hand-written, hand-verified queries against the actual corpus (factual lookups,
exact-term, definitional, comparative, clinician-style, multi-hop, plus 10 deliberately unanswerable queries),
each paired with the exact chunk(s) that answer it.

```bash
python -m eval.eval_retrieval                          # full comparison, all modes
python -m eval.eval_retrieval --mode hybrid --by-type   # one mode, broken down by query type
python -m eval.eval_retrieval --sweep hybrid_alpha      # parameter sweep
python -m eval.eval_retrieval --embedding-model medcpt --index-dir eval/compare-embedding-models/medcpt
python -m eval.threshold_calibration                    # 4-tier confidence-threshold calibration
```

---

## meta.json schema

```json
{
  "doc_id":            "monaleesa2_subanalysis_2018",
  "title":             "Ribociclib plus letrozole ...",
  "pub_year":          2018,
  "pub_month":         2,
  "source_type":       "rct",
  "journal":           "Breast Cancer Research and Treatment",
  "doi":               "10.1007/s10549-017-4518-8",
  "guideline_version": null,
  "cancer_type":       "Breast",
  "cancer_subtype":    ["HR+", "HER2-"],
  "population":        "Postmenopausal women with HR+, HER2- advanced breast cancer ...",
  "line_of_therapy":   "first_line",
  "drug_focus":        ["ribociclib", "letrozole"],
  "drug_class":        "CDK4/6 inhibitor",
  "notes":             "MONALEESA-2 sub-analysis ...",
  "_needs_review":     false,
  "_llm_confidence":   1.0,
  "_conflicts":        []
}
```

| Field | Source | Notes |
|---|---|---|
| `source_type` | LLM → PubMed pub_types | `rct` `meta_analysis` `review` `guideline` `mechanism` `retrospective` |
| `cancer_subtype` | LLM → offline GT | always `list[str]` — e.g. `["HR+", "HER2-"]` |
| `drug_focus` | LLM → offline GT | always `list[str]` — captures combination therapies |
| `_needs_review` | pipeline | `false` = verified and locked; `true` = needs human check |
| `_conflicts` | defensive merge | fields where offline and LLM disagreed |

Offline ground truth always wins over the LLM; disagreements are logged in `_conflicts`, never silently
overwritten. Once `_needs_review: false` is set, a re-run will never overwrite that file.

---

## LLM backends (metadata extraction only)

| Backend | Daily limit | Setup |
|---|---|---|
| Groq `llama-3.1-8b-instant` | 14,400 req/day free | Key in `.secrets/secrets.yaml` |
| Groq `llama-3.3-70b-versatile` | 1,000 req/day free | Same key, change `model` in `config.yaml` |
| Ollama (local) | Unlimited | `brew install ollama && ollama pull llama3.1` |

```yaml
api:
  llm_backend: "ollama"   # or "groq"
```

This is the only step in the pipeline that calls an external API — it runs once per document, not at query
time, so it doesn't compromise the offline design of retrieval itself.

---

## Known limitations

- **Figure captions are excluded from the live corpus.** The captioning pipeline (`caption.py`) runs and
  produces output, but confirmed VLM fabrications (invented drug names, invented statistics, one instance
  describing an entirely different disease) were found across 4 documents during gold-set construction. See
  `decisions.md` D14/D30/D33/D34/D35.
- **Two NCCN source PDFs' license terms restrict AI/tool use** (~78% of the current corpus). Development
  continues, but this must be resolved before any public or graded submission — see `decisions.md` D36.
- **No answer generation yet.** The system returns ranked passages, not a composed answer. Stages 7-9 above are
  approved future work, not descoped.
- **No staleness check** between `chunks.jsonl` and the built indexes — re-run `embed.py` after any `chunk.py`
  change.

---

## GitHub hygiene

```
✅ config.yaml                     safe — no secrets
✅ requirements.txt                safe
✅ .gitignore                      safe
✅ .secrets/secrets.yaml.example   safe — placeholder only
✅ src/*.py, eval/*.py             safe
✅ decisions.md, state.md          project record — should be committed
❌ .secrets/secrets.yaml           gitignored — real API key lives here
❌ data/                           gitignored — raw PDFs too large, extracted output regenerable
❌ index/                          gitignored — binary FAISS/BM25, regenerable via embed.py
❌ venv/                           gitignored
❌ technical-report/                gitignored — personal report drafting space
```

---

## Project structure

```
Oncology Agent/
├── config.yaml                        # pipeline config (no secrets)
├── requirements.txt
├── README.md
├── decisions.md                       # design-decision log — read this first
├── state.md                           # current project snapshot
├── .gitignore
├── .secrets/
│   ├── secrets.yaml                   # your API keys (gitignored)
│   └── secrets.yaml.example
├── src/
│   ├── meta_builder.py                # [1] metadata extraction
│   ├── extract.py                     # [2] PDF → structured markdown (Docling)
│   ├── caption.py                     # [3] figure captioning (VLM) — built, excluded from live corpus
│   ├── chunk.py                       # [4] header-aware chunking
│   ├── embed.py                       # [5] FAISS + BM25 index build
│   ├── retrieve.py                    # [6] hybrid retrieval (dense / hybrid / rrf)
│   ├── verify.py                      # 4-check sanity suite for the built index
│   └── compare_index_types.py         # IndexFlatIP vs IndexHNSWFlat measurement tool
├── eval/
│   ├── gold.jsonl                     # 109-entry hand-verified retrieval eval set
│   ├── eval_retrieval.py              # retrieval metrics + parameter sweeps
│   ├── threshold_calibration.py       # confidence-threshold calibration
│   ├── browse_chunks.py, review_gold.py   # gold-set construction helpers
│   └── compare-embedding-models/      # bge_base/, pubmedbert/, bge_m3/, medcpt/ — each independently rebuildable
├── data/
│   ├── raw/                           # source PDFs (gitignored)
│   ├── extracted/<doc_id>/            # parsed markdown, figures, meta.json
│   └── chunks/chunks.jsonl            # final indexed chunks
└── index/                             # faiss.index, bm25.pkl, id_map.json, chunk_map.json (gitignored)
```

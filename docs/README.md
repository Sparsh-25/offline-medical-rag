# Offline Oncology RAG Agent

**An offline, no-hallucination Retrieval-Augmented Generation system for breast cancer clinical literature.**

Designed for air-gapped / CPU-first deployment with a university GPU upgrade path.  
Every answer is citation-grounded, NLI-verified, and source-typed before reaching the LLM.

---

## Project status

| Phase | Status | Notes |
|---|---|---|
| Phase 0 — Ingestion & Indexing | 🟡 In progress | Indexes built; 5 known bugs unfixed (see below) |
| Phase 1 — Hybrid Retrieval + LLM | ⬜ Not started | |
| Phase 2 — CRAG + Reranking + NLI | ⬜ Not started | |
| Phase 3 — Adaptive Routing | ⬜ Not started | |
| Phase 4 — Compression + Quantization | ⬜ Not started | |
| Phase 5 — Self-Verification | ⬜ Not started | |
| Phase 6 — E5-mistral + Production | ⬜ Not started | |

---

## Quick start

```bash
git clone https://github.com/Sparsh-25/offline-medical-rag
cd offline-medical-rag
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add API keys (Groq for metadata only — no key needed for local inference)
cp .secrets/secrets.yaml.example .secrets/secrets.yaml
# Edit .secrets/secrets.yaml and paste your Groq key

# Drop PDFs into data/raw/ then run in order:
python -m src.meta_builder   # Step 1 — extract clinical metadata
python -m src.extract        # Step 2 — Docling PDF → markdown + figures
python -m src.caption        # Step 3 — LLaVA figure captioning (needs Ollama)
python -m src.chunk          # Step 4 — header-aware chunking
python -m src.embed          # Step 5 — FAISS + BM25 indexes
python -m src.verify         # Step 6 — sanity checks
```

---

## Repository layout

```
offline-medical-rag/
├── README.md
├── config.yaml                        # all pipeline settings (no secrets)
├── requirements.txt
├── .gitignore
├── .secrets/
│   ├── secrets.yaml                   # real API keys — gitignored
│   └── secrets.yaml.example           # safe template to commit
├── src/
│   ├── meta_builder.py                # Step 1 — PDF → meta.json (CrossRef + PubMed + LLM)
│   ├── extract.py                     # Step 2 — Docling PDF → markdown + figure PNGs
│   ├── caption.py                     # Step 3 — LLaVA figure captioning via Ollama
│   ├── chunk.py                       # Step 4 — header-aware chunker + metadata merge
│   ├── embed.py                       # Step 5 — BGE-base FAISS HNSW + BM25 index builder
│   └── verify.py                      # Step 6 — 4-check sanity suite
├── docs/
│   ├── architecture.md                # full system design + data flow
│   ├── roadmap.md                     # phased build plan with gates
│   └── phase0.md                      # Phase 0 decisions, bugs, fixes
├── data/
│   ├── raw/                           # drop PDFs here — gitignored
│   ├── extracted/
│   │   └── <doc_id>/
│   │       ├── <doc_id>.md            # structured markdown post-captioning
│   │       ├── meta.json              # clinical metadata
│   │       ├── extraction_log.json    # figure/table inventory
│   │       └── figures/               # filtered figure PNGs
│   └── chunks/
│       └── chunks.jsonl               # final chunk store — gitignored
└── index/
    ├── faiss.index                    # dense HNSW index — gitignored
    ├── bm25.pkl                       # sparse BM25 index — gitignored
    ├── id_map.json                    # FAISS int position → chunk_id
    └── chunk_map.json                 # chunk_id → full chunk object
```

---

## Tier 0 corpus (current — 6 documents, breast cancer)

| doc_id | Type | Year | Status |
|---|---|---|---|
| `nccn_breast_v2_2026` | guideline | **None ← bug** | pub_year missing |
| `asco_her2_testing_guideline_2023` | review | 2023 | ✓ |
| `monaleesa2_subanalysis_2018` | rct | 2018 | ✓ |
| `frontiers_molbiosci_multiomics_2022` | review | 2022 | ✓ |
| `frontiers_pubhealth_tdxd_cea_2023` | review | 2023 | ✓ |
| `sci_reports_nact_hr_pos_2025` | retrospective | 2025 | ✓ |

**Planned replacements for Tier 0 (before Phase 1):**
- Replace `frontiers_molbiosci_multiomics_2022` with a landmark RCT (DESTINY-Breast04 or MONARCH-3)
- Add Waks & Winer 2019 JAMA biology review
- Add CDK4/6 inhibitor meta-analysis (Cochrane or PMC)

---

## Known bugs — fix before Phase 1

```
CRITICAL  [1] nccn_breast_v2_2026/meta.json: set pub_year: 2026
               477 chunks have recency_weight=0.7 (neutral) instead of 1.0 (most recent)
               Fix: edit meta.json, re-run chunk.py + embed.py

CRITICAL  [2] Junk section filter missing in chunk.py
               AUTHOR CONTRIBUTIONS / REFERENCES chunks contaminate top-3 retrieval results
               Fix: add JUNK_SECTION_PATTERNS filter in chunk_document()

IMPORTANT [3] section_h1 is None for all chunks
               Every chunk shows None › Section Name in prompt context
               Fix: fall back to document title from meta when h1 is None

IMPORTANT [4] 51 chunks exceed 512 tokens (max observed: 1818)
               BGE-base silently truncates these — tail content is not embedded
               Fix: investigate NCCN sections with no double-newline breaks;
               add single-newline subsplit fallback

NOTED     [5] Smoke test queries in verify.py reference docs not in corpus
               (waks, destiny) — 3 of 6 queries will always fail regardless of
               retrieval quality. Update test_queries to match actual doc_ids.
```

---

## Figure captioning status

| Document | Figures saved | Caption status |
|---|---|---|
| `frontiers_pubhealth_tdxd_cea_2023` | 5 | ✓ 5/5 success (bar charts, forest plots) |
| `nccn_breast_v2_2026` | 26 | ✗ 26/26 failed (complex clinical flowcharts) |
| All others | varies | ✗ Failed — ValidationErrors on garbage JSON from LLaVA:7b |

**Root cause:** LLaVA:7b cannot handle complex oncology figures. Fallback text (`Figure from oncology document: <stem>`) is written to markdown so placeholders are replaced, but the captions carry no clinical signal.

**Fix path:**
- CPU now: Switch to `moondream-2b-int8` via Ollama — simpler schema, more reliable JSON, 3–8s/image on CPU
- GPU (university): Switch to `llava:13b` or `BioViL-T` — set `vlm.model` in `config.yaml`, no code change needed

---

## LLM backends

| Backend | Used for | Limit | Config |
|---|---|---|---|
| Groq `llama-3.1-8b-instant` | metadata extraction only | 14,400 req/day free | `.secrets/secrets.yaml` |
| Ollama `llava:7b` | figure captioning (current, failing) | Unlimited | `config.yaml → vlm.model` |
| Ollama (pending) | answer generation Phase 1 | Unlimited | TBD |

---

## Config reference (`config.yaml`)

```yaml
paths:           # input/output paths
embedding:       # model: BAAI/bge-base-en-v1.5, device: cpu
chunking:        # min: 80 tokens, max: 512 tokens
recency:         # baseline_year: 2015 (weight 0.5), peak_year: 2024 (weight 1.0)
vlm:             # model: llava, timeout: 300s, base_url: Ollama
api:             # entrez_email, llm_backend (groq|ollama), groq/ollama settings
extraction:      # figure_min_width: 250px, figure_min_height: 200px
```

---

## GitHub hygiene

```
✅ committed    config.yaml, requirements.txt, README.md, docs/, src/
✅ committed    data/extracted/*/meta.json  (no PII, safe to commit)
✅ committed    .secrets/secrets.yaml.example
❌ gitignored   .secrets/secrets.yaml        (real API keys)
❌ gitignored   data/raw/*.pdf               (too large)
❌ gitignored   data/chunks/chunks.jsonl     (binary-adjacent, regenerable)
❌ gitignored   index/                       (binary FAISS/BM25)
❌ gitignored   venv/
```

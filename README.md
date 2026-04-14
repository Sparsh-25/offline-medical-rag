# Oncology RAG — Metadata Ingestion Pipeline

**Offline medical literature ingestion for an Oncology RAG system.**  
Extracts clinical metadata from PDFs using CrossRef, PubMed, and an LLM (Groq/Ollama), writing structured `meta.json` files used by the retrieval pipeline.

---

## How it works

```
data/raw/*.pdf
      │
      ▼
  [GT]  OFFLINE_GROUND_TRUTH  ──── hardcoded, highest authority (known papers)
  [0]   PDF text + DOI regex  ──── always runs
  [1]   CrossRef API          ──── title, journal, pub_year, pub_month
  [2]   PubMed/Entrez API     ──── abstract, MeSH terms, publication type
  [3]   LLM (Groq or Ollama)  ──── source_type, population, line_of_therapy, drugs
      │
      ▼
 Defensive Merge  ──── Offline values NEVER overwritten by LLM
      │                Conflicts logged in _conflicts + _needs_review = True
      ▼
data/extracted/<doc_id>/meta.json
```

### State-aware re-runs (idempotent)

| Existing file state | Behaviour |
|---|---|
| No file yet | Full extraction runs |
| `_needs_review: false` | File returned as-is — all phases skipped |
| `_needs_review: true` | Existing values used as base; pipeline re-runs to fill nulls |

Once you set `_needs_review: false` in a `meta.json`, it will **never be overwritten** by a re-run.

---

## Adding new papers

1. Drop the PDF into `data/raw/`
2. Run `python -m src.meta_builder`

The pipeline auto-detects all PDFs. For a new unknown paper it will:
- Extract the DOI from the PDF text
- Fetch title, journal, year from CrossRef
- Look up the abstract on PubMed
- Send the abstract to Groq to fill clinical fields

**Optional — add a cleaner folder name** in `DOC_ID_MAP` (`src/meta_builder.py`):
```python
DOC_ID_MAP = {
    "my-new-pdf-filename-stem": "clean_doc_id_2024",
}
```

**Optional — add hardcoded ground truth** for high-stakes known papers in `OFFLINE_GROUND_TRUTH`:
```python
OFFLINE_GROUND_TRUTH = {
    "clean_doc_id_2024": {
        "source_type":    "rct",
        "cancer_subtype": ["HR+", "HER2-"],
        "drug_focus":     ["drug_a", "drug_b"],
    }
}
```
Ground truth values always win over the LLM. Use this for papers where guaranteed correctness is required.

---

## Setup

```bash
git clone <repo>
cd "Oncology Agent"
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add your Groq API key (free at console.groq.com)
cp .secrets/secrets.yaml.example .secrets/secrets.yaml
# Edit .secrets/secrets.yaml and paste your key
```

---

## Running

```bash
# Full pipeline — all PDFs in data/raw/
python -m src.meta_builder

# Single PDF
python -m src.meta_builder --pdf data/raw/my_paper.pdf

# Skip LLM (CrossRef + Entrez only — no API key needed)
python -m src.meta_builder --no-llm
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

---

## LLM backends

| Backend | Daily limit | Setup |
|---|---|---|
| Groq `llama-3.1-8b-instant` | 14,400 req/day free | Key in `.secrets/secrets.yaml` |
| Groq `llama-3.3-70b-versatile` | 1,000 req/day free | Same key, change `model` in `config.yaml` |
| Ollama (local, M1) | Unlimited | `brew install ollama && ollama pull llama3.1` |

Switch backend in `config.yaml`:
```yaml
api:
  llm_backend: "ollama"   # or "groq"
```

---

## GitHub hygiene

```
✅ config.yaml                     safe — no secrets
✅ requirements.txt                safe
✅ .gitignore                      safe
✅ .secrets/secrets.yaml.example   safe — placeholder only
✅ src/meta_builder.py             safe
✅ data/extracted/*/meta.json      safe — no PII
❌ .secrets/secrets.yaml           gitignored — real API key lives here
❌ data/raw/*.pdf                  gitignored — too large
❌ venv/                           gitignored
❌ index/                          gitignored — binary FAISS/BM25
```

---

## Project structure

```
Oncology Agent/
├── config.yaml                  # pipeline config (no secrets)
├── requirements.txt
├── README.md
├── .gitignore
├── .secrets/
│   ├── secrets.yaml             # your API keys (gitignored)
│   └── secrets.yaml.example     # safe template
├── src/
│   ├── meta_builder.py          # metadata extraction pipeline
│   └── extract.py               # PDF chunking (next phase)
├── data/
│   ├── raw/                     # drop PDFs here (gitignored)
│   └── extracted/
│       └── <doc_id>/
│           └── meta.json
└── index/                       # FAISS + BM25 indexes (gitignored)
```
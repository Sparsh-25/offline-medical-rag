# Phase 0 — Ingestion & Indexing

## What Phase 0 is

Phase 0 builds the foundation every later phase depends on:
- Clean markdown from each document (Docling extraction)
- Clinical metadata per document (meta_builder)
- Figure captions embedded as alt-text (caption.py)
- Header-aware chunks with full metadata baked in (chunk.py)
- Dense FAISS HNSW + sparse BM25 indexes ready for retrieval (embed.py)
- Verified indexes with no known data quality issues (verify.py)

Phase 0 is not complete until verify.py passes all 4 checks cleanly.

---

## Current state (as of first verify.py run)

```
Total chunks:     610
Documents:        6
Dense index:      FAISS HNSW, 610 vectors, dim=768
Sparse index:     BM25Okapi, 610 documents
Embedding model:  BAAI/bge-base-en-v1.5 (CPU)
```

### verify.py results summary

| Check | Status | Notes |
|---|---|---|
| 1 — Chunk quality | 🔴 Fail | 477 chunks missing pub_year; 51 over 512 tokens; junk chunks present |
| 2 — Embedding sanity | ✅ Pass | Norms 1.0000, same-doc (0.6695) > cross-doc (0.6476) |
| 3 — Dense smoke test | 🟡 Partial | 2/6 correct — but 3 queries reference docs not in corpus |
| 4 — BM25 + RRF | ✅ Pass | Exact-term query BM25 wins correctly; RRF fuses sensibly |

---

## Active design decisions

### Why header-aware chunking (not sliding window)

Clinical documents have strong section structure — Results, Methods, Discussion, Dosing, Staging. A chunk that crosses section boundaries carries mixed intent and produces incoherent embeddings. Splitting only at H1/H2/H3 preserves clinical coherence: a Results chunk knows it is from Results, a Dosing chunk knows it is from Dosing. This context is injected as `section_h1/h2/h3` into every chunk and surfaced in the LLM prompt.

Sliding window with overlap is used in general-purpose RAG where documents have no structure. Oncology PDFs from PMC and NCCN always have structure — using it is strictly better.

### Why metadata is baked into chunks at build time

At query time, FAISS returns integer positions. Those map to `chunk_id` via `id_map.json`, which maps to the full chunk object in `chunk_map.json`. The full chunk object already contains `pub_year`, `source_type`, `recency_weight`, `drug_focus` — everything the retriever and prompt assembler need.

Alternative: store only `chunk_id` in FAISS, look up `meta.json` at query time. This requires file I/O per retrieved chunk at query time. With 20 retrieved chunks per query it is negligible for 6 documents, but becomes a bottleneck at Tier 2 (500 docs). Baking metadata in is the correct choice for any production-oriented system.

### Why both dense and sparse in Phase 0 (not just dense)

RRF fusion (Phase 1) requires both indexes to already exist. Building sparse separately in Phase 1 would mean rebuilding the retrieval interface twice. The cost of building BM25 in Phase 0 is negligible (~1 second for 610 chunks). It is always built alongside FAISS.

### Why FAISS HNSW instead of IndexFlatIP

`IndexFlatIP` is exact search — O(n) per query. At 610 chunks it is imperceptibly fast. At 5000 chunks (Tier 1) it is still fast. `IndexHNSWFlat` is approximate but scales to millions of vectors with sub-linear query time.

The reason to use HNSW now: it is the same code at all corpus sizes. Switching from FlatIP to HNSW later would require rebuilding the index with different parameters. Using HNSW from the start means no migration cost.

Trade-off: HNSW with M=32 uses ~2× the RAM of FlatIP. At 610 vectors × 768 dim × 4 bytes ≈ 1.9MB — irrelevant at this scale.

### Why Groq for metadata extraction (not local LLM)

meta_builder.py runs once per document, requires internet, and produces a file that is then manually verified and locked (`_needs_review: false`). Using Groq (free tier, 14,400 req/day) is faster than waiting for a local 7B model on CPU and produces better extraction quality. The metadata pipeline is not on the inference critical path — it does not need to be offline.

At Tier 1/2, the same LLM extraction runs via Ollama locally (configured in config.yaml), because batch processing of 50–500 documents would exhaust the Groq free tier.

### Why the NCCN guideline dominates the corpus (477/610 chunks)

NCCN guidelines are long structured documents covering all breast cancer subtypes, staging, treatment algorithms, and references. 477 chunks from one document is expected and correct — the NCCN is the primary reference for clinical decision making. It will continue to dominate Tier 0.

Mitigation already designed: `max_per_doc=2` cap in Phase 1 `retrieve.py` prevents NCCN from filling all top-k slots at query time. This cap is applied post-RRF, not during indexing.

---

## Caption failure impact on current chunks

caption.py ran with llava:7b and failed on 27/32 figures across the corpus.

What this means for your chunks:
- figure_caption chunks contain fallback text like 
  "Figure from oncology document: figure-003-page-004"
- This text has zero clinical signal — it will never retrieve correctly
- BGE-base embeds it but it matches nothing a user would query
- You have only 2 real figure_caption chunks out of 610 total

This is acceptable for Phase 0 prototype because:
- The prose and table chunks (608 of 610) are unaffected
- Figure captions are supplementary — core clinical content 
  is in prose sections
- You fix captioning before Phase 1, then re-run the full pipeline

## Caption fix — what to do before Phase 1

### Option A: Moondream on CPU (do this now, today)

Switch model in config.yaml:
  vlm:
    model: "moondream"
    timeout: 60

Install: ollama pull moondream

Problem: caption.py expects LLaVA JSON output via Pydantic schema.
Moondream returns plain text, not JSON. So you need a small code change
in caption.py — add a moondream-specific path that skips JSON parsing
and just uses the plain text as the summary directly.

In caption_image() in caption.py, add at the top:

  if "moondream" in VLM_MODEL.lower():
      # Moondream returns plain text, not JSON
      # Use a simple prompt, get text back, wrap in minimal VisualExtraction
      simple_prompt = (
          "Describe this oncology figure in 2-3 sentences. "
          "State the figure type, what clinical finding it shows, "
          "and any key numbers visible like hazard ratios or p-values."
      )
      payload = {
          "model": VLM_MODEL,
          "prompt": simple_prompt,
          "images": [encoded],
          "stream": False,
      }
      response = requests.post(
          f"{VLM_BASE_URL}/api/generate",
          json=payload,
          timeout=VLM_TIMEOUT,
      )
      response.raise_for_status()
      summary = response.json().get("response", "").strip()
      if not summary:
          return None
      return VisualExtraction(
          figure_id=figure_id,
          archetypes=["Other"],
          summary=summary,
          clinical_relevance=None,
      )

Expected results with Moondream:
- frontiers_pubhealth figures (bar charts, forest plots): good captions
- NCCN flowcharts: basic description, not structured — acceptable
- Speed: 3-8 seconds per image on CPU
- No ValidationErrors

### Option B: llava:13b on university GPU (do when GPU access available)

No code change needed. Just change config.yaml:
  vlm:
    model: "llava:13b"

Submit caption.py as a GPU job. Expected results:
- NCCN flowcharts: much better — 13b handles complex diagrams
- All figure types: reliable JSON output, full Pydantic schema works
- Speed: 5-15 seconds per image on GPU

### After fixing captioning

Re-run the full pipeline in order:
  python src/caption.py   ← re-captions all figures
  python src/chunk.py     ← rebuilds chunks with real captions
  python src/embed.py     ← rebuilds indexes
  python src/verify.py    ← confirm figure_caption count improved

Check 1 should show figure_caption chunks > 2 after this.

---

## Fix instructions (in order)

### Fix 1 — NCCN pub_year (5 minutes)

```bash
# Edit the file
nano data/extracted/nccn_breast_v2_2026/meta.json

# Change:
"pub_year": null,
# To:
"pub_year": 2026,

# Then set reviewed:
"_needs_review": false,

# Re-run (order matters):
python src/chunk.py
python src/embed.py
python src/verify.py
```

Expected result: Check 1 shows `Missing pub_year: 0 ✓`

---

### Fix 2 — Junk section filter (15 minutes)

Add to `src/chunk.py` before the `chunk_document()` function:

```python
import re

JUNK_SECTION_PATTERNS = [
    r"^author contributions?",
    r"^references?$",
    r"^acknowledgements?",
    r"^conflict(s)? of interest",
    r"^funding",
    r"^supplementary",
    r"^data availability",
    r"^ethics (statement|approval)",
    r"^abbreviations?$",
    r"^supporting information",
]

def is_junk_section(heading: str) -> bool:
    if not heading:
        return False
    h = heading.lower().strip()
    return any(re.match(p, h) for p in JUNK_SECTION_PATTERNS)
```

In `chunk_document()`, in the section loop, add this check after updating the header breadcrumb:

```python
for sec in sections:
    level, heading, body = sec["level"], sec["heading"], sec["body"]

    # Update header breadcrumb
    if level == 1:
        h1, h2, h3 = heading, None, None
    elif level == 2:
        h2, h3 = heading, None
    elif level == 3:
        h3 = heading

    # NEW: skip junk sections
    if is_junk_section(heading):
        print(f"    Skip junk section: {heading!r}")
        continue

    if not body:
        continue
    ...
```

Re-run `chunk.py` → `embed.py` → `verify.py`.

Expected result: AUTHOR CONTRIBUTIONS chunks no longer appear in Check 3 top-3 results.

---

### Fix 3 — section_h1 fallback (5 minutes)

In `chunk_document()` in `src/chunk.py`, after building the `Chunk` object and before appending it:

```python
chunk = Chunk(
    ...
    section_h1=h1,
    ...
)

# NEW: fall back to document title when top-level header is missing
if chunk.section_h1 is None:
    chunk.section_h1 = meta.get("title", doc_id)

chunks.append(chunk)
chunk_index += 1
```

Re-run `chunk.py` → `embed.py` → `verify.py`.

Expected result: Check 3 shows `Title of Document › Section Name` instead of `None › Section Name`.

---

### Fix 4 — Oversized chunks (30 minutes)

First, investigate which documents are producing oversized chunks:

```bash
python - <<'EOF'
import json
chunks = [json.loads(l) for l in open("data/chunks/chunks.jsonl")]
big = [c for c in chunks if c["token_count"] > 512]
print(f"Total oversized: {len(big)}")
from collections import Counter
by_doc = Counter(c["doc_id"] for c in big)
print("By doc:", dict(by_doc))
for c in sorted(big, key=lambda x: -x["token_count"])[:5]:
    print(f"\n{c['chunk_id']} — {c['token_count']} tokens")
    print(f"  H2: {c['section_h2']}")
    print(f"  First 300 chars: {c['chunk_text'][:300]!r}")
EOF
```

If the oversized chunks are from NCCN and contain single-newline-separated content (not double-newline), update `subsplit()` in `chunk.py`:

```python
def subsplit(body: str, max_tokens: int) -> list[str]:
    if rough_tokens(body) <= max_tokens:
        return [body]

    # Try double-newline split first (paragraph boundary)
    paragraphs = re.split(r'\n\n+', body)

    # If that didn't help (e.g. NCCN uses single \n), fall back to single \n
    if len(paragraphs) == 1:
        paragraphs = re.split(r'\n', body)

    chunks = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n" + para).strip() if current else para
        if rough_tokens(candidate) > max_tokens and current:
            chunks.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())

    # Last resort: hard character split if a single line is still over limit
    final = []
    for chunk in chunks:
        if rough_tokens(chunk) > max_tokens:
            # Split at sentence boundary (period + space + capital)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', chunk)
            cur = ""
            for sent in sentences:
                candidate = (cur + " " + sent).strip() if cur else sent
                if rough_tokens(candidate) > max_tokens and cur:
                    final.append(cur.strip())
                    cur = sent
                else:
                    cur = candidate
            if cur.strip():
                final.append(cur.strip())
        else:
            final.append(chunk)
    return final
```

Re-run `chunk.py` → `embed.py` → `verify.py`.

Expected result: Check 1 shows `Over 512 tokens: 0 ✓`.

---

### Fix 5 — Update smoke test queries (5 minutes)

In `src/verify.py`, update `test_queries` in `check_dense_retrieval()` to match your actual corpus:

```python
test_queries = [
    ("What is the median progression-free survival for ribociclib plus letrozole?",
     "monaleesa"),
    ("First-line treatment recommendation for HR positive HER2 negative metastatic breast cancer",
     "nccn"),
    ("HER2 testing IHC scoring criteria for breast cancer",
     "asco"),
    ("trastuzumab deruxtecan cost-effectiveness analysis breast cancer",
     "frontiers_pubhealth"),
    ("multiomics biomarker breast cancer treatment response",
     "frontiers_molbiosci"),
    ("neoadjuvant chemotherapy HR positive breast cancer pathological complete response",
     "sci_reports"),
]
```

---

## Caption model comparison for CPU

| Model | Size | Speed (CPU) | JSON reliability | Quality |
|---|---|---|---|---|
| `llava:7b` (current) | ~4GB | 60–300s/img | Poor — complex schema | Fails on flowcharts |
| `moondream` (recommended) | 1.8GB | 3–8s/img | N/A (plain text output) | Good for bar charts, KM curves |
| `llava-phi3` | ~2.5GB | 15–40s/img | Medium | Better than 7b |
| `llava:13b` (GPU) | ~8GB | 5–15s/img (GPU) | Good | Good on all figure types |
| `BioViL-T` (GPU) | varies | GPU only | N/A | Best for radiology/IHC |

**Recommended CPU path:** Switch to `moondream` now. Update `config.yaml → vlm.model: "moondream"`. Re-run `caption.py`. The Pydantic VisualExtraction schema will need to be bypassed — Moondream returns plain text, not JSON. Simplest approach: in `caption_image()`, if the model is `moondream`, skip the JSON parsing and return a `VisualExtraction` with only `summary` populated.

**GPU path (university cluster):** Keep current LLaVA architecture, switch to `llava:13b`. The entire `caption.py` pipeline — Pydantic schema, sanitiser, metadata builder — works unchanged. Just change `vlm.model`.

---

## Known failure cases

| Scenario | Current behavior | Phase it gets fixed |
|---|---|---|
| NCCN clinical flowcharts | LLaVA:7b times out or hallucinates | Phase 0 fix (GPU: llava:13b) |
| Queries about docs not in corpus | Bot may confabulate from partial context | Phase 2 (CRAG abstain) |
| Very long NCCN sections | Truncated silently by BGE-base | Fix 4 above |
| Conflicting evidence across trials | LLM picks whichever was retrieved first | Phase 5 (NLI cross-check) |
| Drug name misspellings in query | Dense fails; BM25 also misses | Phase 2 (query expansion, future) |

---

## Benchmark status

| Benchmark | Status | Target |
|---|---|---|
| RAGCare-QA MRR@10 | Not yet run — Phase 1 prerequisite | > 0.55 (Phase 2 gate) |
| RAGCare-QA faithfulness | Not yet run | > 0.70 (Phase 2 gate) |
| Phase 0 smoke test (6 queries) | 2/6 correct (3 docs not in corpus) | ≥ 4/6 (Phase 0 gate) |
| medQA (for LLM quality baseline) | Not yet run — pending LLM selection | Q4 within 4pts of Q8 |

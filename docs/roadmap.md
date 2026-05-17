# Roadmap — Offline Oncology RAG Agent

## Design philosophy

Each phase adds exactly one layer of capability. Nothing is skipped.
Every phase has a measurable gate — the system must pass it before the next phase begins.
Bugs from Phase 0 are fixed before Phase 1 code is written.

---

## Phase 0 — Ingestion & Indexing

**Goal:** Clean, verified indexes from 6 gold-standard breast cancer documents.

### Completed
- [x] `meta_builder.py` — CrossRef + PubMed + Groq LLM → `meta.json` with defensive merge
- [x] `extract.py` — Docling PDF → markdown + figure PNGs (size filter, content dedup)
- [x] `caption.py` — LLaVA:7b via Ollama → Pydantic VisualExtraction → markdown injection
- [x] `chunk.py` — Header-aware chunking, metadata merge, subsplitter
- [x] `embed.py` — FAISS HNSW (dense) + BM25Okapi (sparse) + id_map + chunk_map
- [x] `verify.py` — 4-check sanity suite (chunk quality, embedding sanity, dense smoke, RRF)
- [x] First verify.py run completed — output analyzed

### Bugs to fix before Phase 1 (in priority order)

**[CRITICAL-1]** `nccn_breast_v2_2026/meta.json` — set `pub_year: 2026`
- Impact: 477/610 chunks have `recency_weight=0.7` instead of `1.0`
- Fix: edit `meta.json`, re-run `chunk.py` + `embed.py` + `verify.py`

**[CRITICAL-2]** Junk section filter missing in `chunk.py`
- Impact: AUTHOR CONTRIBUTIONS / REFERENCES chunks appearing in top-3 retrieval
- Fix: add to `chunk_document()`:
```python
JUNK_SECTION_PATTERNS = [
    r"^author contributions", r"^references$",
    r"^acknowledgements", r"^conflict of interest",
    r"^funding", r"^supplementary", r"^data availability",
]
def is_junk_section(heading: str) -> bool:
    h = heading.lower().strip()
    return any(re.match(p, h) for p in JUNK_SECTION_PATTERNS)
```

**[IMPORTANT-3]** `section_h1` is None for all chunks
- Impact: every chunk prompt shows `None › Section Name`
- Fix: in `chunk_document()`, after building Chunk: `if chunk.section_h1 is None: chunk.section_h1 = meta.get("title", doc_id)`

**[IMPORTANT-4]** 51 chunks exceed 512 tokens (max: 1818)
- Impact: BGE-base silently truncates — tail of oversized chunks not embedded
- Investigation: `python -c "import json; chunks=[json.loads(l) for l in open('data/chunks/chunks.jsonl')]; big=[c for c in chunks if c['token_count']>512]; [print(c['chunk_id'], c['token_count'], c['chunk_text'][:100]) for c in sorted(big, key=lambda x:-x['token_count'])[:5]]"`
- Likely cause: NCCN sections use single `\n` not `\n\n` — subsplitter never fires
- Fix: add `\n` fallback in `subsplit()` when `\n\n` split produces no reduction

**[NOTED-5]** Smoke test queries reference docs not in corpus
- Impact: 3/6 queries always fail regardless of retrieval quality
- Fix: update `test_queries` in `verify.py` to match actual `doc_id` values

### Figure captioning fix (separate from above)

Current: LLaVA:7b fails on 27/32 figures. Fallback text has no clinical signal.

CPU fix (do now): Switch to Moondream-2b-int8 via Ollama
```yaml
# config.yaml
vlm:
  model: "moondream"   # ollama pull moondream
  timeout: 60
```
Moondream uses a simpler output schema, does not return complex JSON, and runs reliably on CPU at 3–8s/image. Trade-off: less structured output than LLaVA. Caption quality is sufficient for Phase 0 — the alt-text is embeddable and clinically descriptive.

GPU fix (Phase 1 / university cluster): Switch to `llava:13b` — no code change needed, just change `vlm.model` in config.

After fixing captioning, re-run `caption.py` → `chunk.py` → `embed.py` → `verify.py` in order.

### Phase 0 gate (all must pass before Phase 1)

- [ ] verify.py Check 1: zero chunks with missing pub_year or source_type
- [ ] verify.py Check 1: zero chunks over 512 tokens
- [ ] verify.py Check 1: zero junk section chunks
- [ ] verify.py Check 2: same-doc cosine > cross-doc cosine
- [ ] verify.py Check 3: ≥ 4/6 smoke test queries return correct doc at rank-1
- [ ] verify.py Check 4: RRF results make clinical sense on both test queries

---

## Phase 1 — Hybrid Retrieval + LLM wire-up

**Goal:** End-to-end pipeline. User types a question, bot returns a sourced answer.

### Components to build

**`src/retrieve.py`** — single function every later phase calls
```python
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    # 1. dense_search(query, top_k=20) via FAISS HNSW
    # 2. sparse_search(query, top_k=20) via BM25
    # 3. rrf(dense, sparse, k=60) → fused ranking
    # 4. max_per_doc=2 cap (prevents NCCN dominating at 78% of corpus)
    # 5. return top_k chunks with full metadata
```

**`src/prompt.py`** — prompt assembler
```
System: You are an oncology clinical assistant...
        Cite every claim with [SOURCE N].
        When sources conflict, prefer recency_weight closer to 1.0.
        If context is insufficient, say "not enough evidence in available documents."

[SOURCE 1 — recency: 1.0 — type: guideline — NCCN Breast v2.2026 — Section: Systemic Therapy]
{chunk_text}

[SOURCE 2 — recency: 0.65 — type: rct — MONALEESA-2, NEJM 2016 — Section: Results]
{chunk_text}

Question: {user_query}
Answer:
```

**`src/answer.py`** — LLM runner via llama-cpp-python
- Model: Mistral-Small-3.2 or LLaMA-3-8B-Instruct
- Format: Q8 GGUF (not Q4 yet — need clean baseline before quantization)
- Install: `pip install llama-cpp-python`
- Download: `huggingface-cli download bartowski/Mistral-Small-3.2-24B-Instruct-2506-GGUF --include "*.Q8*"`

**`src/cli.py`** — simple interactive CLI
```bash
python src/cli.py
> What is the first-line treatment for HR+/HER2- metastatic breast cancer?
[retrieving...]
[SOURCE 1 — recency: 1.0 — type: guideline — NCCN Breast v2.2026]
CDK4/6 inhibitor plus endocrine therapy is the preferred first-line regimen...
```

### Phase 1 gate
- [ ] Bot correctly answers 4/5 manually written factoid queries against the corpus
- [ ] Every answer includes at least one `[SOURCE N]` citation
- [ ] No answer fabricates a drug name or statistic not present in retrieved chunks
- [ ] End-to-end latency < 60s on CPU (retrieval + prompt assembly + Q8 LLM)

---

## Phase 2 — CRAG + Cross-encoder Reranking + NLI

**Goal:** Retrieval quality gates — stop bad chunks from reaching the LLM.

### Components to build

**Cross-encoder reranking:**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~66MB)
- Position: after RRF, before LLM
- Retrieve top-20 → rerank → pass top-5
- Typical gain: 8–15% MRR improvement

**CRAG scorer:**
- Score each retrieved chunk against the query using cross-encoder
- Threshold: if max score < 0.3 → flag as "insufficient context" → bot abstains
- Abstain response: "The available documents do not contain enough evidence to answer this reliably. Retrieved context: [show what was found]."

**NLI faithfulness check (soft — for logging only in Phase 2):**
- Model: `cross-encoder/nli-deberta-v3-small` (~80MB)
- Input: `(answer_sentence, retrieved_chunk)` pairs
- Labels: entail | neutral | contradict
- In Phase 2: log contradictions, do not block. Blocking happens in Phase 5.

### Phase 2 gate
- [ ] RAGCare-QA MRR@10 > 0.55
- [ ] RAGCare-QA faithfulness > 0.70
- [ ] Abstain rate on out-of-corpus questions > 80% (bot says "not enough evidence" correctly)

---

## Phase 3 — Adaptive Routing

**Goal:** Route queries to the right retrieval strategy based on query type.

### Components to build

**DistilBERT query classifier:**
- Fine-tune on labeled oncology queries
- Classes: `factoid` | `clinical_reasoning` | `comparative` | `ambiguous`
- Training data: label 200–500 queries from RAGCare-QA manually
- Routes:
  - `factoid` → dense-only (faster)
  - `comparative` → hybrid + wider top-k
  - `ambiguous` → hybrid + lower CRAG threshold

**RAGCare-QA benchmark harness:**
- Run after every routing change
- Metrics: MRR@10, Recall@5, faithfulness
- Gate: must improve vs Phase 2 baseline before advancing

### Phase 3 gate
- [ ] DistilBERT classifier accuracy > 0.80 on held-out query set
- [ ] RAGCare-QA MRR@10 improves by ≥ 2 points vs Phase 2

---

## Phase 4 — Context Compression + Quantization

**Goal:** Reduce compute and context window without quality loss.

### Components to build

**ACC-RAG context compression:**
- Score each sentence in retrieved context by relevance to query
- Drop sentences below threshold before LLM prompt assembly
- Target: 40–60% context reduction
- EXIT: early-exit scoring on easy sentences after fewer transformer layers

**4-bit GGUF quantization:**
- Method: SpinQuant (rotation + GPTQ calibration) — NOT naive GPTQ

> ⚠️ **SpinQuant Risk Flag:**  
> SpinQuant requires access to the base model weights and a calibration dataset (typically C4 or a domain subset). It is not a post-hoc tool you run on an existing GGUF — it requires modifying the model during quantization. This means:  
> - You need the full float16/bfloat16 checkpoint (~14–48GB depending on model)  
> - You need GPU compute for the calibration pass (A100 recommended)  
> - University GPU cluster access is required  
>   
> **Simpler alternative:** Use `llama.cpp` Q4_K_M directly (GGUF download from HuggingFace). Quality is slightly lower than SpinQuant but the process is trivial. Validate by running the Q4 model on a held-out medQA subset and checking accuracy drops < 3–4 points vs Q8.  
>   
> **Recommendation:** Start with Q4_K_M GGUF download. Only invest in SpinQuant if the quality drop is unacceptable.

### Phase 4 gate
- [ ] Q4 model medQA accuracy within 4 points of Q8 baseline
- [ ] ACC-RAG achieves ≥ 40% context reduction with < 2 point MRR drop
- [ ] End-to-end latency < 30s on CPU with Q4 + compression

---

## Phase 5 — Self-Verification Hardening

**Goal:** Hard NLI gate — contradicted claims are blocked, not just logged.

### Components to build

**Abstain policy (hard):**
- If NLI entailment score < 0.6 across all retrieved chunks for the generated answer → return structured refusal
- Refusal format: "Insufficient evidence in available documents. The most relevant passage found was: [show chunk]. Confidence: [score]."

**Citation attachment:**
- Every sentence in output maps to a `chunk_id`
- Surface in CLI: show source passage on demand

**Cross-check:**
- For answers involving statistics (HR, p-value, median survival) — verify the exact number appears in a retrieved chunk
- If the number is not in any chunk → flag as hallucinated → abstain

### Phase 5 gate
- [ ] Hallucination rate on fabricated-statistic test set < 5%
- [ ] Citation coverage > 90% (every claim has a source)
- [ ] User-facing "not enough evidence" rate is clinically appropriate (not over-triggering)

---

## Phase 6 — E5-mistral + Production Hardening

**Goal:** Upgrade embedding model and expand corpus to Tier 1 (30–50 docs).

### Components to build

**E5-mistral-7b-instruct upgrade:**
- Requires ~14GB VRAM for inference — university GPU
- Re-index all chunks after upgrading (BGE and E5 embeddings are not compatible)
- Gate: only migrate if retrieval recall is the measurable bottleneck

**Tier 1 corpus expansion:**
- Entrez API bulk pull + LLM metadata extraction (automated)
- Quality filter: RCT or meta-analysis only, journal impact > 5, 2015–2025
- Still requires human review for low-confidence LLM extractions

**Continuous eval harness:**
- RAGCare-QA + custom oncology test set
- Automated regression on every significant code change
- Track: faithfulness, relevance, latency per query type

### Phase 6 gate
- [ ] RAGCare-QA MRR@10 > 0.70 on Tier 1 corpus
- [ ] Full pipeline latency < 45s on CPU (or < 10s on GPU)

---

## Removed from roadmap / descoped

| Item | Reason |
|---|---|
| BioViL-T for figure captioning | Requires significant setup; LLaVA:13b on GPU is simpler and sufficient |
| SpinQuant as primary quantization | High GPU compute cost; Q4_K_M GGUF is good enough for prototype |
| Web fallback in CRAG | Offline-first requirement — no internet at inference time |
| Streaming responses | Not needed for prototype; add in Phase 6 if needed |

---

## Corpus expansion plan

| Tier | Size | Method | When |
|---|---|---|---|
| Tier 0 | 6 docs | Manual collection + manual meta.json | Now (Phase 0) |
| Tier 1 | 30–50 docs | Entrez API + LLM extraction + human spot-check | After Phase 2 gate passes |
| Tier 2 | 200–500 docs | Entrez bulk + LLM extraction + quarantine queue | After Phase 4 gate passes, on GPU |

**Tier 1 gate:** RAGCare-QA MRR@10 must improve vs Tier 0 baseline before expanding further.
**Rule:** Never expand corpus before the pipeline handles current corpus cleanly.

# Pre-Phase-2 Plan — real, ordered, actual

_The one rule: nothing in Phase 2 (reranking, CRAG, NLI) gets built until Phase 1 runs end-to-end AND you have baseline numbers from a working eval harness. Optimising against no baseline is how you waste a month._

Where you are today: `src/` has 6 files (`meta_builder, extract, caption, chunk, embed, verify`). That is the ingest+index pipeline only. There is no retrieval fusion, no LLM, no eval. Everything below is what stands between you and a defensible Phase 2 start.

---

## PART A — Close out Phase 0 (fixes + corrections)

### A1. The bugs already on your TODO (do these first, they're cheap)
- **NCCN `pub_year`** → set `2026`, `_needs_review: false` in `data/extracted/nccn_breast_v2_2026/meta.json`. Fixes 477 chunks stuck at recency 0.7.
- **Junk section filter** in `chunk.py` — `JUNK_SECTION_PATTERNS` + `is_junk_section()`, skip AUTHOR CONTRIBUTIONS / REFERENCES / FUNDING etc.
- **`section_h1` fallback** to document title when H1 is `None`.
- **Oversized-chunk subsplit** — `\n` fallback + sentence-boundary last resort in `subsplit()`.
- **`verify.py` smoke queries** — replace the ones referencing docs not in the corpus (waks, destiny) with real `doc_id`s.
- **Caption pivot** — switch `config.yaml → vlm.model: "moondream"`, add the plain-text path in `caption.py`, `ollama pull moondream`. Do NOT keep investing in the LLaVA JSON schema.

### A2. Corrections to those fixes (these matter — the TODO version is not enough)
- **Token counter is wrong and will keep truncating silently.** `rough_tokens = len(text.split())*1.3` counts words; BGE's 512 limit is WordPiece *subwords*. Oncology terms explode into many subwords, so a "500-token" chunk can be 700+ real tokens and still get cut by BGE. Fix: load the real tokenizer and count actual tokens.
  ```python
  from transformers import AutoTokenizer
  _tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
  def real_tokens(text: str) -> int:
      return len(_tok.encode(text, add_special_tokens=False))
  ```
  Use `real_tokens` everywhere `rough_tokens` is used, and set the subsplit ceiling to ~480 (leave headroom for BGE's special tokens).
- **HNSW is the wrong index at your scale and it's untuned.** `embed.py` sets `efConstruction` but never `efSearch` (default 16 → recall loss), and at 610 vectors approximate search buys you nothing. Switch to exact:
  ```python
  index = faiss.IndexFlatIP(dim)   # exact, faster than HNSW at this scale, zero tuning
  ```
  (Revisit HNSW only when the corpus is in the thousands. Keep it a one-line swap.)

### A3. Corpus (recommended, not strictly required)
- 6 docs with NCCN = 78% of chunks makes every eval number NCCN-shaped. Pull **~6–9 more docs forward** (target ~12–15 total): 2–3 landmark RCTs (e.g. DESTINY-Breast04, MONARCH-3), a CDK4/6 meta-analysis, the Waks & Winer review. Reuse `meta_builder.py`. This is the difference between eval numbers that mean something and numbers dominated by one document.

### A4. Re-run and pass the Phase 0 gate
Order: `caption.py → chunk.py → embed.py → verify.py`. Gate = all four checks green:
zero missing `pub_year`/`source_type`, zero chunks over the real-token ceiling, zero junk chunks, same-doc cosine > cross-doc, ≥4/6 smoke queries rank-1 correct, RRF sane on both test queries, figure_caption chunk count > 2.

---

## PART B — Build Phase 1 (the actual RAG loop)

Four new files. This is what turns an index into an agent.

### B1. `src/retrieve.py`
```python
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    # dense_search(query, 20) via FAISS  → [(chunk_id, score)]
    # sparse_search(query, 20) via BM25
    # rrf(dense, sparse, k=60)           → fused ranking
    # max_per_doc=2 cap AFTER rrf        → stops NCCN filling every slot
    # return top_k full chunk dicts (metadata already baked in)
```
- Load `faiss.index`, `bm25.pkl`, `id_map.json`, `chunk_map.json` once at import, not per-query.
- BM25 tokenisation must match embed.py (same lowercase/whitespace) or sparse recall drops.

### B2. `src/prompt.py`
- Assembles the source-headed prompt with recency + type + section, exactly as in `architecture.md`.
- **Add a context-budget guard the TODO is missing:** sum real tokens of selected chunks; if over the model's window (minus room for the answer), drop lowest-ranked chunks. Otherwise long NCCN chunks silently overflow the LLM context.

### B3. `src/answer.py`
- `llama-cpp-python` runner. **Correction to your model choice:** Mistral-Small-3.2-**24B** at Q8 is ~25GB and will take *minutes* per query on laptop CPU — it directly contradicts your own <60s latency gate. Real dev path:
  - **Dev/default:** Llama-3.1-8B-Instruct or Qwen2.5-7B-Instruct at **Q4_K_M / Q5_K_M** (~5GB, seconds/query on CPU).
  - Keep the "clean Q8 baseline before quantization" idea, but apply it to the **8B** (Q8 of an 8B is ~8GB and runnable). Decouple "which model" from "quantization experiment" — a 24B needs Q4 just to run, so it can't be your Q8 baseline anyway.
- Settings for factual QA: `temperature=0`, low `top_p`, cap `max_tokens`.

### B4. `src/cli.py`
- Interactive loop: query → retrieve → prompt → answer → print answer + resolved sources.
- **Add citation validation now (don't wait for Phase 5):** parse `[SOURCE N]` from the answer, confirm each N maps to a chunk you actually retrieved; flag any that doesn't. Cheap, and it catches the most common hallucination immediately.
- **Add a pre-LLM abstain floor:** if top RRF score is trivially low / no chunk clears a minimal threshold, return "not enough evidence in available documents" before calling the LLM. Full CRAG is Phase 2; this stub prevents confident answers from empty retrieval.
- Log every query's retrieved chunk_ids + scores to a JSONL — you need this for the eval harness and for debugging.

### B5. Phase 1 gate
4/5 hand-written factoid queries answered correctly with ≥1 valid `[SOURCE N]`; no fabricated drug/stat; end-to-end < 60s CPU on the 8B model.

---

## PART C — Evaluation harness (the actual missing piece — build BEFORE Phase 2)

Without this, every Phase 2/3 gate number is meaningless. RAGCare-QA does **not** fit a 6-doc breast corpus (it's broad-specialty one-choice MCQ scored on accuracy, not MRR) — use it only as an optional secondary sanity check, not your primary gate.

### C1. Build a gold set: 30–50 Q/A grounded in YOUR docs
One JSONL, `eval/gold.jsonl`, each line:
```json
{
  "qid": "q001",
  "question": "Median PFS for ribociclib + letrozole first-line HR+/HER2- MBC?",
  "relevant_chunk_ids": ["monaleesa2_subanalysis_2018_chunk_0007"],
  "reference_answer": "≈25.3 months (MONALEESA-2).",
  "type": "factoid",
  "in_corpus": true
}
```
- ~30–40 answerable questions spread across your docs (not just NCCN), tagged factoid / clinical / comparative.
- ~8–10 deliberately **out-of-corpus** questions (`in_corpus: false`) to measure abstention.
- You find `relevant_chunk_ids` by reading your own `chunks.jsonl` — tedious but it IS the project's ground truth.

### C2. `eval/eval_retrieval.py` — retrieval quality
For each in-corpus query, run `retrieve(query, top_k=10)` and compute against `relevant_chunk_ids`:
- **Recall@5**, **Recall@10**
- **MRR** = mean of 1/(rank of first relevant chunk)
- **Hit@1**, **Hit@3**
Report overall + per query-type. This is your retrieval baseline; Phase 2 reranking must beat it or it doesn't ship.

### C3. `eval/eval_answers.py` — answer quality
Run the full Phase 1 pipeline per query and score:
- **Citation validity** (automatic): every `[SOURCE N]` resolves to a retrieved chunk. Target ~100%.
- **Abstention rate** on out-of-corpus questions: bot should say "not enough evidence". Target high.
- **Groundedness / correctness**: for the ~30 in-corpus questions, judge whether the answer is supported and matches the reference. Pre-Phase-2 you have no NLI, so use one of:
  - manual rubric (0/1 supported, 0/1 correct) — 30 questions is doable by hand, and defensible for a capstone; or
  - an LLM judge via Groq (you already have the key) scoring support/correctness — cheaper to repeat, note it's not offline.
- **Latency** per query.

### C4. Freeze the baseline
Run C2 + C3 on Phase 1, save the numbers to `eval/baseline_phase1.json`. **These become the reference the Phase 2 gate is measured against** (e.g. "rerank must lift MRR by ≥X, faithfulness by ≥Y vs this file"). Rewrite the Phase 2 gate in `roadmap.md` to reference this baseline instead of RAGCare-QA absolute thresholds.

---

## PART D — Consolidated "missing / to correct" checklist

Missing entirely (must add): `retrieve.py`, `prompt.py`, `answer.py`, `cli.py`, the gold eval set, `eval_retrieval.py`, `eval_answers.py`, a downloaded GGUF model, context-budget guard, citation validation, pre-LLM abstain stub, retrieval logging.

To correct in existing code: token counter (word-proxy → real tokenizer), FAISS index (HNSW → FlatIP or set efSearch), model choice (24B-Q8 → 8B first), smoke-test queries, caption model (LLaVA → moondream), NCCN meta.json.

To reframe in docs: drop "zero hallucination by construction" → "grounded, citation-first, abstains on insufficient evidence"; re-point Phase 2 gate at the Phase 1 baseline, not RAGCare-QA absolutes.

---

## Ordered sequence (do it in this order)

1. A1 + A2 fixes → re-run pipeline → Phase 0 gate green.
2. (Optional but recommended) A3 corpus expansion → re-run → re-gate.
3. Download GGUF (8B Q4_K_M) + `pip install llama-cpp-python`.
4. B1 `retrieve.py` → smoke-test retrieval by hand.
5. B2 `prompt.py` + B3 `answer.py` + B4 `cli.py` → first end-to-end answer.
6. B5 Phase 1 gate (5 factoid queries).
7. C1 gold set (this is the slow part — budget real time).
8. C2 + C3 eval scripts → C4 freeze baseline.
9. Only now: start Phase 2.

## Rough effort (solo, part-time)
A (fixes+corrections): ~1 week. A3 corpus: ~3–5 days. B (Phase 1 build + model): ~2–3 weeks. C (gold set + eval scripts + baseline): ~2–3 weeks — the gold set dominates. Total before Phase 2: **~6–8 weeks.**

## Definition of done (the real Phase 2 entry gate)
- Phase 0 verify.py all green on the current corpus.
- `python src/cli.py` answers a typed clinical question with valid citations in < 60s CPU.
- `eval/gold.jsonl` exists with ≥30 answerable + ≥8 out-of-corpus questions.
- `eval/baseline_phase1.json` exists with real Recall@5 / MRR / citation-validity / abstention / latency numbers.
- Phase 2 gate in roadmap.md rewritten to "beat baseline_phase1.json", not RAGCare-QA absolutes.

If all five are true, you're genuinely ready for Phase 2. If any is false, you're not — no matter how much Phase 2 code you write.

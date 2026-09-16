"""
answer.py — Stage 8 answer generation for the Oncology RAG pipeline.

Pipeline position:
    retrieve.py -> prompt.py -> [answer.py]

Given a query, this:
  1. retrieves the top chunks (retrieve.py — MedCPT + BM25 hybrid),
  2. builds the grounded RAG prompt (prompt.py — system prompt v1.2),
  3. runs the local LLM (Qwen3-32B Q6_K via llama-cpp-python — see decisions.md D59/D60),
  4. returns a grounded answer plus the sources it was given.

All settings come from config.yaml -> llm. The model runs wherever llama-cpp-python and
the GGUF file live (the GPU box); the import of llama_cpp is deferred to load time so
this module still imports on a machine without it (e.g. for testing prompt assembly).

Usage:
    python -m src.answer --query "first-line treatment for HR+/HER2- metastatic breast cancer"
"""

from __future__ import annotations

# Must be set before torch / tokenizers load (via retrieve.py), or macOS can segfault.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import re
from pathlib import Path

import yaml

from src.retrieve import retrieve, MAX_PER_DOC, TOP_K
from src.prompt import build_messages
from src.rerank import load_reranker, rerank, cap_top_k

_ROOT = Path(__file__).resolve().parent.parent
cfg = yaml.safe_load((_ROOT / "config.yaml").read_text())
LLM_CFG = cfg["llm"]
RERANK_CFG = cfg.get("reranker", {})

# The model is big and slow to load, so we load it once and reuse it.
_llm = None
# Same for the reranker (loaded only if enabled).
_reranker = None


def load_reranker_once():
    """Load the configured reranker once, then reuse on later calls."""
    global _reranker
    if _reranker is None:
        _reranker = load_reranker(RERANK_CFG["model"])
    return _reranker


def retrieve_top_k(query: str, top_k: int) -> list[dict]:
    """
    Get the top_k chunks for a query. If the reranker is enabled (config), pull a
    larger uncapped candidate pool, reorder it with the cross-encoder, then apply the
    per-doc cap and keep top_k — the exact flow the D66 A/B measured (R@5 0.813).
    Otherwise fall back to plain hybrid retrieval.
    """
    if not RERANK_CFG.get("enabled"):
        return retrieve(query, top_k=top_k)

    pool = retrieve(query, top_k=RERANK_CFG.get("pool", 50), max_per_doc=10**6)  # 10**6 = no cap
    reordered = rerank(query, pool, load_reranker_once())
    top = cap_top_k(reordered, max_per_doc=MAX_PER_DOC, top_k=top_k)
    for i, chunk in enumerate(top, start=1):   # renumber rank to the reranked order
        chunk["rank"] = i
    return top


def load_llm():
    """Load the GGUF model once via llama-cpp-python, then reuse on later calls."""
    global _llm
    if _llm is not None:
        return _llm

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is not installed here. Install it on the machine that has "
            "the GPU and the GGUF model (see decisions.md D59 for the build command)."
        )

    model_path = LLM_CFG["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"LLM model not found: {model_path}\nSet llm.model_path in config.yaml."
        )

    print(f"Loading LLM: {model_path}")
    _llm = Llama(
        model_path=model_path,
        n_ctx=LLM_CFG.get("n_ctx", 8192),
        n_gpu_layers=LLM_CFG.get("n_gpu_layers", -1),
        chat_format=LLM_CFG.get("chat_format"),   # None -> use the GGUF's own template
        verbose=False,
    )
    return _llm


NO_EVIDENCE = "The provided sources do not contain enough information to answer this question."


def answer(query: str, top_k: int | None = None) -> dict:
    """
    Retrieve, prompt, and generate a grounded answer for one query.

    Returns a dict: {query, answer, sources}, where `sources` lists the chunks the
    model was given (rank, chunk_id, doc_id, section) for citation/traceability.
    """
    results = retrieve_top_k(query, top_k or TOP_K)
    if not results:
        return {"query": query, "answer": NO_EVIDENCE, "sources": []}

    messages = build_messages(query, results, no_think=LLM_CFG.get("no_think", False))
    llm = load_llm()
    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=LLM_CFG.get("max_tokens", 768),
        temperature=LLM_CFG.get("temperature", 0.0),
    )
    text = out["choices"][0]["message"]["content"].strip()
    # Strip any <think>...</think> reasoning block (Qwen3) — no-op for other models.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    sources = [
        {
            "rank": c["rank"],
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "section": c.get("section_h2") or c.get("section_h1"),
        }
        for c in results
    ]
    return {"query": query, "answer": text, "sources": sources}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded answer for a query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k")
    args = parser.parse_args()

    result = answer(args.query, top_k=args.top_k)
    print(f"\nQ: {result['query']}\n")
    print(result["answer"])
    print("\nSources the answer was grounded in:")
    for s in result["sources"]:
        print(f"  [{s['rank']}] {s['doc_id']}  §{s['section']}")


if __name__ == "__main__":
    main()

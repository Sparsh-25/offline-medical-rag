"""
rerank.py — cross-encoder reranking of retrieved candidates (optional second stage).

A cross-encoder reads the (query, chunk) pair *together* and scores how relevant the
chunk is to the query. Unlike the bi-encoder retriever (which embeds query and chunk
separately), it can weigh their interaction, so it usually orders the top candidates
better. We use it as a second stage: retrieve.py returns a pool of candidates, we
rerank that pool, and keep the best few for the LLM.

All four rerankers we compare are sequence-classification cross-encoders (they output a
single relevance score per pair), so ONE code path loads any of them via plain
transformers. We avoid sentence-transformers on purpose: the GPU box doesn't install it,
to protect its pinned torch build (see decisions.md D59). See decisions.md D66 for why
these four models.

    from src.rerank import load_reranker, rerank
    scorer = load_reranker("medcpt")
    best_five = rerank(query, candidates, scorer)[:5]
"""

from __future__ import annotations

# Short name -> (HuggingFace model id, max input tokens for the query+chunk pair).
# Two medical? No — only MedCPT is a real off-the-shelf medical cross-encoder; the
# other three are general (see D66).
#
# max_length is each model's REAL capacity, not a fixed 512. Our chunks reach ~500
# tokens (p90=499), so a 512-limited model must clip the tail of a long chunk once the
# query is added — we truncate 'only_second' below so the query is always kept intact
# and only the chunk tail is ever cut. bge-v2-m3 supports long inputs, so we give it
# room to read the whole chunk+query with no truncation. Each model is compared at the
# capacity it would actually run with — that is the fair comparison (D66).
RERANKERS = {
    "medcpt":    ("ncbi/MedCPT-Cross-Encoder", 512),           # BERT — hard 512 limit
    "bge-v2-m3": ("BAAI/bge-reranker-v2-m3", 1024),            # supports 8192; 1024 fits chunk+query whole
    "mxbai-v1":  ("mixedbread-ai/mxbai-rerank-base-v1", 512),  # 512-limited architecture
    "bge-base":  ("BAAI/bge-reranker-base", 512),              # XLM-R base — 512 limit
}


def load_reranker(name: str):
    """
    Load a reranker by short name and return a function score(query, texts) that
    gives one relevance number per text (higher = more relevant).

    torch/transformers are imported here, not at module top, so importing this file
    is cheap on a machine that only needs the rerank() helper.
    """
    if name not in RERANKERS:
        raise ValueError(f"unknown reranker {name!r}; choose from {list(RERANKERS)}")

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_id, max_length = RERANKERS[name]
    print(f"Loading reranker: {model_id} (max_length={max_length})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # use_safetensors=True forces the .safetensors weights and skips torch.load, which
    # the box's pinned torch (<2.6) refuses over CVE-2025-32434. All four rerankers ship
    # safetensors, so this loads everywhere without upgrading torch (see decisions.md D59).
    model = AutoModelForSequenceClassification.from_pretrained(model_id, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def score(query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        with torch.no_grad():
            # truncation='only_second' keeps the whole query and clips only the chunk
            # tail if the pair is over max_length (see the RERANKERS note above).
            encoded = tokenizer(pairs, padding=True, truncation="only_second",
                                max_length=max_length, return_tensors="pt").to(device)
            logits = model(**encoded).logits   # shape [n, 1] — one score per pair
        return logits.squeeze(-1).cpu().tolist()

    return score


def rerank(query: str, candidates: list[dict], score, text_key: str = "chunk_text") -> list[dict]:
    """Return the candidate dicts reordered by the reranker's score, most relevant first."""
    scores = score(query, [c[text_key] for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in ranked]

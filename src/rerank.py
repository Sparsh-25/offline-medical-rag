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

# Short name -> HuggingFace model id. Two medical? No — only MedCPT is a real
# off-the-shelf medical cross-encoder; the other three are general (see D66).
RERANKERS = {
    "medcpt":    "ncbi/MedCPT-Cross-Encoder",           # medical, pairs with our MedCPT retriever
    "bge-v2-m3": "BAAI/bge-reranker-v2-m3",             # general, strong
    "mxbai-v1":  "mixedbread-ai/mxbai-rerank-base-v1",  # general
    "bge-base":  "BAAI/bge-reranker-base",              # general, light (fast on Mac CPU)
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

    model_id = RERANKERS[name]
    print(f"Loading reranker: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    def score(query: str, texts: list[str]) -> list[float]:
        pairs = [[query, text] for text in texts]
        with torch.no_grad():
            encoded = tokenizer(pairs, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(device)
            logits = model(**encoded).logits   # shape [n, 1] — one score per pair
        return logits.squeeze(-1).cpu().tolist()

    return score


def rerank(query: str, candidates: list[dict], score, text_key: str = "chunk_text") -> list[dict]:
    """Return the candidate dicts reordered by the reranker's score, most relevant first."""
    scores = score(query, [c[text_key] for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [candidate for candidate, _ in ranked]

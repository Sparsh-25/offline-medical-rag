# Project description (for GitHub repo description field)

Offline no-hallucination RAG system for oncology clinical literature. Hybrid dense+sparse retrieval (BGE-base + BM25/RRF), NLI-verified generation, citation-grounded answers. CPU-first, GPU upgrade path. Breast cancer prototype — NCCN guidelines, RCTs, ASCO guidelines.

---

# Extended description (for README intro, portfolio, or project report)

**Offline Oncology RAG Agent** is a retrieval-augmented generation system designed for clinical literature question-answering in oncology, with a hard requirement of zero hallucination and fully offline operation.

The system ingests breast cancer research documents — NCCN guidelines, landmark RCTs, ASCO guidelines, systematic reviews — and answers clinical queries with source-grounded, NLI-verified responses. Every claim in the output is traceable to a specific chunk from a specific document, with publication year, source type, and recency weight surfaced to the LLM for evidence prioritization.

**Architecture highlights:**
- Metadata-driven retrieval: pub_year, source_type, drug_focus, line_of_therapy baked into every chunk at index build time — no join at query time
- Hybrid retrieval: BGE-base-en-v1.5 dense (FAISS HNSW) + BM25 sparse, fused via Reciprocal Rank Fusion
- Figure understanding: LLaVA vision-language model extracts structured clinical data (hazard ratios, cohort sizes, p-values) from Kaplan-Meier curves, forest plots, and CONSORT diagrams
- No-hallucination pipeline: CRAG scoring gates retrieval quality, cross-encoder reranking improves precision, NLI verification blocks contradicted claims before they reach the user
- Offline-first: all inference runs locally via llama.cpp and Ollama; Groq is used only for one-time metadata extraction

**Build roadmap:** Six phased additions — hybrid retrieval → CRAG + reranking → adaptive DistilBERT routing → ACC-RAG context compression + 4-bit quantization → self-verification with citation attachment → E5-mistral upgrade and corpus expansion to 200–500 documents.

Built as a pre-final year B.E. project in Robotics & AI at Thapar Institute of Engineering and Technology.

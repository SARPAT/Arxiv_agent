# Arxiv Agent

A retrieval-augmented generation (RAG) system for answering questions over a fixed corpus of arXiv papers on NLP and LLM research, combining `sentence-transformers` embeddings with NVIDIA-hosted LLMs for generation.

**Work in progress.** This repository is in early development — a full README with architecture diagram and evaluation results will be added once the core pipeline is built.

**Model choices (verified 2026-08-30):** Embedding uses `BAAI/bge-base-en-v1.5`, self-hosted via `sentence-transformers` (768-dim, zero cost, zero rate limit, zero vendor deprecation risk); generation uses `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM, picked from NVIDIA's own Nemotron lineage over third-party re-hosted checkpoints after two prior picks — `meta/mixtral-8x7b-instruct` and `meta/llama-3.1-8b-instruct` — were deprecated mid-build.

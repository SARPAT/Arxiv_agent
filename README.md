# Arxiv Agent

A retrieval-augmented generation (RAG) system for answering questions over a fixed corpus of arXiv papers on NLP and LLM research, combining `sentence-transformers` embeddings with NVIDIA-hosted LLMs for generation.

**Work in progress.** This repository is in early development — a full README with architecture diagram and evaluation results will be added once the core pipeline is built.

**Model choices (verified 2026-08-30):** Embedding uses `BAAI/bge-base-en-v1.5`, self-hosted via `sentence-transformers` (768-dim, zero cost, zero rate limit, zero vendor deprecation risk); generation uses `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM, picked from NVIDIA's own Nemotron lineage over third-party re-hosted checkpoints after two prior picks — `meta/mixtral-8x7b-instruct` and `meta/llama-3.1-8b-instruct` — were deprecated mid-build.

## Deployment

The backend (`app/api.py`) deploys to [Render](https://render.com) as a
Python web service, configured via the `render.yaml` Blueprint at the repo
root — Render reads it directly, no manual dashboard setup beyond
supplying the three declared environment variables (`NVIDIA_API_KEY`,
`REDIS_URL`, `CORS_ALLOWED_ORIGIN`) it doesn't ship values for. The
frontend (`ui/app.py`) deploys separately as a Gradio Space on
[Hugging Face Spaces](https://huggingface.co/spaces), pointed at the
Render backend via its own `BACKEND_URL` environment variable.

**Embedding is genuinely self-hosted in production, not just in local
development.** The embedding model (`sentence-transformers`, CPU-only)
runs in-process inside the same Render web service that serves `/chat` —
there's no separate embedding API or managed vector service in the loop.
This is a real deployment decision, not just a diagram: it's what keeps
the system free to run (no per-call embedding cost) and is why
Checkpoint 4d's first task was pinning a CPU-only `torch` build before
anything else — a CUDA build wouldn't fit Render's free-tier 512MB RAM
limit at all.

**Cold starts, stated plainly:** Render's free tier spins the service down
after 15 minutes of no traffic. The first request after that has to wait
for a full cold start — roughly 30-60 seconds — before it gets a
response. This is a real, user-visible limitation of running on a free
tier, not a bug; a paid Render plan (or a scheduled keep-alive ping)
removes it, but neither is in scope here.

**What's implemented here vs. what still needs a human:** the retry/timeout
logic, the SSE `error` event, CORS support, and the `render.yaml` Blueprint
are all implemented and tested (see Checkpoint 4d's PR). Actually deploying
— creating a Render account and a Hugging Face account, connecting this
repo, running the Blueprint, creating the Space, and wiring the two
together with real URLs — requires human-created accounts and dashboard
access this environment doesn't have, and hasn't happened yet.

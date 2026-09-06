# Arxiv Agent

A retrieval-augmented generation (RAG) system for answering questions over a fixed corpus of arXiv papers on NLP and LLM research, combining a self-hosted ONNX Runtime embedder with NVIDIA-hosted LLMs for generation.

**Work in progress.** This repository is in early development — a full README with architecture diagram and evaluation results will be added once the core pipeline is built.

**Model choices:** Generation uses `nvidia/nemotron-3.5-lightning-30b-a3b` via NVIDIA NIM (verified 2026-08-30), picked from NVIDIA's own Nemotron lineage over third-party re-hosted checkpoints after two prior picks — `meta/mixtral-8x7b-instruct` and `meta/llama-3.1-8b-instruct` — were deprecated mid-build.

Embedding uses `BAAI/bge-small-en-v1.5` (384-dim, zero cost, zero rate limit, zero vendor deprecation risk), run via **ONNX Runtime** rather than `torch`/`sentence-transformers` (Checkpoint 4f) — see `rag/embedder.py`. This replaced two earlier assumptions that turned out wrong under real deployment constraints: Checkpoint 4e assumed *model weight size* was the dominant memory cost on Render's free tier and swapped `bge-base` for `bge-small` alone; a continuous, uninterrupted memory trace showed baseline memory climbing well past 700MB within 15 seconds of process start, before any request or model load — the real cost was `torch` + `sentence-transformers` + `langchain`'s own import and runtime footprint, not the model file. Checkpoint 4f removes that framework instead of just shrinking the model on top of it. The ONNX model source (`Xenova/bge-small-en-v1.5`) was confirmed live rather than guessed blind; see `rag/embedder.py` for the exact files and the pooling-strategy verification.

## Vector store

Vectors live in a **Qdrant Cloud** collection (Checkpoint 6). Before that
the corpus was a FAISS index built locally and committed to the repo as a
binary, loaded into the web service's memory at startup; that path is gone
entirely — there is no index file, no local index loading, and no
configuration flag selecting between the two. Rollback is `git revert`,
not a toggle.

The collection uses a **named vector `"dense"`** (384-dim, cosine). The
name is deliberate: a collection's vector configuration is immutable after
creation, so an unnamed default vector would have to be deleted and
re-ingested just to add a sparse vector later. Sparse/hybrid retrieval is
not implemented — this naming is the only concession to it.

**Multi-tenancy** is a single collection with a `tenant_id` payload field,
indexed with `is_tenant=True` so Qdrant physically co-locates each
tenant's vectors on disk (its documented recommendation for this pattern).
Every chunk of the shared arXiv corpus is written under `tenant_id:
"public"`. Retrieval takes a *list* of tenant ids, so a future change can
search the shared corpus and a user's own uploads in one query.

Planned user document upload (Checkpoint 7) will be **session-scoped**:
uploads get the session id as their `tenant_id`, which means they do not
persist beyond the session. That is a deliberate limitation of this
design — session-scoped uploads need no account system, no per-user
storage quota, and no deletion/GDPR story — not an oversight.

**Score direction changed with the backend.** FAISS returned an L2
distance where *lower* was a better match; Qdrant returns cosine
similarity where *higher* is better. The embedder L2-normalizes its
output, so the two rank identically and retrieval quality is unchanged —
but any score value recorded before Checkpoint 6 is not comparable with
one recorded after it.

`ingestion/build_index.py` is the migration: re-running it upserts the
corpus into Qdrant using deterministic point ids, so a re-run overwrites
rather than duplicates.

## Deployment

The backend (`app/api.py`) deploys to [Render](https://render.com) as a
Python web service, configured via the `render.yaml` Blueprint at the repo
root — Render reads it directly, no manual dashboard setup beyond
supplying the declared environment variables (`NVIDIA_API_KEY`,
`REDIS_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `CORS_ALLOWED_ORIGIN`) it
doesn't ship values for; `QDRANT_COLLECTION` ships with a default. See
`.env.example` for the same set for local development. The
frontend (`ui/app.py`) deploys separately as a Gradio Space on
[Hugging Face Spaces](https://huggingface.co/spaces), pointed at the
Render backend via its own `BACKEND_URL` environment variable.

**Embedding is genuinely self-hosted in production, not just in local
development.** The embedding model runs in-process (via ONNX Runtime,
CPU-only — see Checkpoint 4f above) inside the same Render web service
that serves `/chat` — there's no separate embedding API in the loop. This
is a real deployment decision, not just a diagram: it's what keeps the
system free to run, at the cost of the service's own memory budget having
to fit the embedder alongside everything else Render's free tier allows
(512MB total) — the reason this project went through two rounds of
memory-driven changes (Checkpoint 4e, then 4f) to fit inside it.

Vector *storage*, unlike embedding, is no longer in-process: Checkpoint 6
moved it to Qdrant Cloud's free tier (see **Vector store** above). That
trades one managed dependency for a chunk of the memory budget the
committed FAISS index used to occupy, and removes a binary from the repo.

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

## Evaluation

| | Checkpoint 3 (`bge-base`) | Checkpoint 4e (`bge-small` + `sentence-transformers`) | Checkpoint 4f (`bge-small` + ONNX) |
|---|---|---|---|
| `recall_at_5` / `recall_at_8` / `mrr` | see `eval/results_checkpoint3.json` | not run — 4e's PR closed before this step | pending |
| `false_accept_rate` / `false_reject_rate` | see `eval/results_checkpoint3.json` | not run | pending |
| `context_survival_rate` | see `eval/results_checkpoint3.json` | not run | pending |

Checkpoint 4e's own re-ingest/recalibrate/re-eval cycle never happened —
its PR was closed before that step, and this checkpoint's index/threshold/
eval are all being regenerated fresh against the ONNX pipeline instead of
building on 4e's. Checkpoint 4f's own numbers require re-running
`ingestion/build_index.py`, `eval/calibrate_threshold.py`, and
`eval/run_eval.py --output-suffix checkpoint4f` for real, which needs
network access to Hugging Face and arXiv that this sandbox does not have.
This table gets filled in once that happens.

Note: the confidence gate was later removed (the pipeline now always
answers and labels provenance via the system prompt), so `false_reject_rate`
is no longer a metric `eval/run_eval.py` reports — the pipeline can't
reject a question anymore. `false_accept_rate` remains and, with the gate
gone, is now the primary guard the eval measures: it catches an
out-of-corpus answer falsely attributed to a real paper.

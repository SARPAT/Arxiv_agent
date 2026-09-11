# Architecture

How the pieces fit, and why they are arranged this way. The README covers
what the project is and how to run it; this covers the decisions.

---

## Request path

`Gradio (HF Space)` → `FastAPI (Render)` → retrieval cache → ONNX embedder
→ Qdrant Cloud → context assembly → NVIDIA NIM → SSE token stream back.

See `docs/architecture.png` for the full diagram, and
`docs/architecture.mmd` for its source.

---

## Vector store

- **One Qdrant Cloud collection** holds everything. There is no second
  store and no local index.
- **Named vector `"dense"`**, 384-dim, cosine. A collection's vector config
  is immutable, so an unnamed default vector would force a delete and
  re-ingest just to add a sparse vector later.
- **Cosine scores are higher-is-better**, roughly 0–1. The FAISS L2
  distance this replaced was lower-is-better and unbounded; embeddings are
  L2-normalized, so ranking is identical but no score recorded before
  Checkpoint 6 is comparable with one recorded after.
- **Every filtered payload field is indexed.** Qdrant Cloud's free tier runs
  in strict mode, which rejects a filter on an unindexed field with a 400
  rather than falling back to a scan. `tenant_id`, `chunk_type` and
  `paper_key` are indexed at `ensure_collection()` time, and
  `count_points()` refuses a filter on anything outside that set.

---

## Multi-tenancy

Every point carries a `tenant_id` payload field, indexed with
`is_tenant=True` so Qdrant physically co-locates a tenant's vectors.

| Tenant | Contents | Lifetime |
|---|---|---|
| `"public"` | The shared arXiv corpus | Permanent, rebuilt by `ingestion/build_index.py` |
| A session id | That session's uploaded PDF | Until the session's Redis keys expire |

`search()` takes a **list** of tenant ids, so one query covers the shared
corpus and a session's own document together. `app.session.tenant_ids_for()`
decides which: `["public", session_id]` once that session has uploaded
something, `["public"]` otherwise.

**That decision is answered from Redis, never from Qdrant.** A per-message
"does this session have uploads?" query would put a network round trip on
the hot path of every message to answer a question that is almost always
"no".

`delete_by_tenant()` refuses `"public"` and the empty string outright. One
wrong argument would wipe the corpus, and the only way back is a full
re-ingestion — so it is a hard guard in the function, not a caller's
responsibility.

---

## User document upload

`POST /upload` → `ingestion.upload.process_upload()` → Qdrant, under the
session's tenant.

- **PDF only**, 10 MB cap, one document per session — a new upload replaces
  the previous one.
- **Text-based PDFs only.** pypdf returns empty text for a scanned page
  without raising, so an explicit guard rejects anything under
  `min_extracted_chars` with a message that says the document looks
  scanned. Without it the upload would "succeed" and then retrieve nothing.
- **The same chunker as the corpus**, imported from
  `ingestion.build_index.build_splitter()` rather than re-declared, so
  uploaded and corpus chunks cannot drift into different sizes inside one
  index.
- **Uniform payload shape.** Uploaded chunks carry every field a corpus
  chunk carries, with empty strings where a PDF cannot supply one
  (`arxiv_id`, `Published`, `Authors`, `Summary`), so nothing downstream
  has to special-case where a chunk came from.
- **Uploaded chunks are not boosted.** They compete on cosine similarity
  like everything else.

**Write ordering is the crash-safety design.** The Redis marker is cleared,
then the session's old points are deleted, then the new ones are written,
then the marker is set. A failure anywhere in between leaves the session
searching the shared corpus alone — never a marker pointing at points that
are half-written or already gone.

Processing is synchronous, on a worker thread under an explicit timeout, so
a pathological PDF returns a 504 instead of occupying a request slot. The
pipeline is a standalone function precisely so moving it to a queue later
changes what calls it, not the logic.

**Embedding is batched, and that is a memory bound rather than a
throughput tweak.** A forward pass allocates attention scores of shape
`(batch, heads, seq_len, seq_len)`, so its peak scales with the batch
size and the *square* of the longest sequence in it. Embedding a whole
document at once therefore scaled with whatever the user uploaded: a
traced 77-chunk upload peaked at **+533MB** over baseline on a 512MB
instance and OOM-killed it. `EMBED_BATCH_SIZE` (default 8) makes that
peak a function of a configured constant instead — the same upload
measured **+110MB**, ~4.8x lower, and slightly faster. It is an env var,
not a constant, so it can be lowered on a live instance without a
redeploy.

Batching is output-invariant: padding is masked, so each sequence's
result does not depend on what else shares its batch. Verified
bit-identical (`0.00e+00` max difference) across batch sizes of 1, 8 and
all-at-once.

`app/api.py` logs this process's RSS once at startup, after the embedder
and Qdrant client are loaded, on a background thread so a cold-start
model download cannot stall the health check. Upload headroom is the
instance limit minus that figure, and no local measurement can substitute
for it — a sandbox's baseline is not Render's. Measured on Render: **282.9 MB**.

`process_upload()` logs one INFO line per request with wall-clock seconds
per stage — `extract`, `delete`, `chunk`, `embed`, `upsert` — emitted on
the failure path as well as the success one. On a failure the stages that
never completed are simply absent, which is what names the one it stopped
inside. `UPLOAD_TIMEOUT_SECONDS` (default 60) bounds the whole thing and
is env-tunable for the same reason `EMBED_BATCH_SIZE` is.

---

## Caching

Two Redis layers, both optional for correctness — with one exception, below.

```
embedding:<embedder>:<query>
retrieval:<backend>:<embedder>:<generation>:<corpus_version>:<tenant_scope>:<query>
```

Each segment exists because its absence caused, or would cause, a real bug:

| Segment | Invalidates on |
|---|---|
| `backend` | FAISS→Qdrant swap — a cached L2 score served to higher-is-better logic |
| `embedder` | A change to the ONNX model or pooling, which changes the vectors |
| `generation` | A system-prompt or model change |
| `corpus_version` | A corpus re-ingest, bumped automatically at the end of one |
| `tenant_scope` | **Which tenants were searched** |

**`tenant_scope` is a correctness requirement, not an optimisation.** Once a
session can upload a document, two sessions asking the same question no
longer deserve the same chunks. Without this segment, session A's private
document would be cached under a key session B's identical question hits —
a cross-session data leak that crashes nothing and looks like a working
feature.

It is a hash of the **sorted, de-duplicated** tenant list actually searched.
That shape matters: every public-only query, from every session, produces
one identical scope and therefore shares one cache entry, so the shared
corpus stays exactly as cacheable as before. Only sessions that have
uploaded something get their own entries.

The key is built and parsed in one place — `_retrieval_key()` and
`parse_retrieval_key()`, both driven by `RETRIEVAL_KEY_SEGMENTS`.
`scripts/spot_check_cache_keys.py` reads through that parser rather than
counting segments itself, because counting segments is what silently broke
it when a segment was last inserted.

---

## Sessions

Redis, keyed `chat:{session_id}:*`, on a rolling one-hour TTL that every
read and write refreshes:

- `:history` — the conversation, so multi-turn context survives between
  requests without the frontend holding it.
- `:upload` — the uploaded document's filename and chunk count, which is
  what `tenant_ids_for()` reads and what `GET /upload/status` reports so the
  frontend's indicator survives a page refresh.

Session-scoped uploads are a deliberate limitation, not an oversight: they
need no account system, no per-user storage quota and no deletion story.
The cost is orphaned points when a session never returns, which
`scripts/cleanup_orphaned_uploads.py` reclaims — dry-run by default,
`--delete` to act, `"public"` excluded unconditionally.

`GET /corpus/info` gives the frontend the corpus's paper titles and the
upload size cap for its welcome message and upload widget, both derived
live from `rag.pipeline.REAL_PAPER_TITLES` (built off `TARGET_PAPERS`) and
`settings.max_upload_bytes` on every call - never a separate hardcoded
count or size the frontend could drift out of sync with.

---

## Generation

One NVIDIA NIM call per turn, with a per-attempt timeout and a
retry that distinguishes transient failures from permanent ones (a
deprecated model is a `410`; retrying it just burns the user's wait).

There is **no confidence gate**. Retrieval's top-k always reaches the model,
and the system prompt carries provenance: ground the answer in the context
when it is relevant, ignore it and answer from clearly-labelled general
knowledge when it is not. A calibrated distance threshold was tried and
removed — on dense-only retrieval the score bands for correct in-corpus
answers and out-of-corpus questions overlapped too much for any single
cutoff to separate them without rejecting legitimate answers.

Sources are built from the **retrieved chunks' metadata**, not parsed out of
the model's text, and the prompt forbids the model writing its own
`Sources:` block. Corpus chunks resolve to canonical paper titles; anything
else falls back to its own `Title`, which is what credits an uploaded
document.

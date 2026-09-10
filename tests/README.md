# Tests

## Running them

```bash
python -m tests.run_all          # everything, with a pass/fail summary
python -m tests.test_qdrant_swap # one module by name
```

Not `pytest tests/` — see **Why these aren't `pytest`-style** below. It
will still *run* (report "0 items"), but that's not the supported path.

No network access and no real Qdrant/Redis/NVIDIA credentials are needed.
Every external client is mocked or faked; `tests/__init__.py` supplies
placeholder env values so `app/config.py`'s settings validate at import
time.

## What's here and what verified it landed correctly

Eight modules, each mocking the layer(s) it needs (a `MagicMock` Qdrant
client, `fakeredis`, a stub embedder/generation call) and none reaching
the network:

| Module | Covers |
|---|---|
| `test_qdrant_swap` | Checkpoint 6 — FAISS→Qdrant: named-vector config, tenant payload index, point-id determinism, metadata preservation, cosine score direction, the retrieval cache key's backend segment |
| `test_gate_removed` | The confidence gate's removal: no abstain path, `similarity_threshold` gone, the SSE `done` frame, the system prompt's provenance rules |
| `test_sources_dedup` | `sources[]` built once from retrieved-chunk metadata, not parsed from model text; general-knowledge answers cite nothing |
| `test_checkpoint7_upload` | Checkpoint 7 — PDF upload: `delete_by_tenant` refusing `"public"`, filename sanitization, upload ordering (delete-before-write), and **the cross-session retrieval-cache-key leak test** |
| `test_cache_scripts` | `scripts/spot_check_cache_keys.py` and `scripts/flush_embedding_cache.py` against a populated fake Redis |
| `test_embedder` | `rag/embedder.py`'s ONNX pooling/normalization math and singleton caching, against a fake `onnxruntime` session |
| `test_plain_overview_chunks` | The synthetic `plain_overview` chunks' exact text and metadata |
| `test_ui_error` | `ui/app.py`'s `chat_fn()` handling an SSE `error` event without hanging |

Run right now, on this branch, all eight pass — see each module's own
docstring for exactly what it checks. `test_checkpoint7_upload`'s leak
test was additionally confirmed to *fail* when the fix it tests is
reverted, not just observed to pass; see its module docstring.

## Why these aren't `pytest`-style

Each module is a standalone script — module-level `assert`s and
`print("PASSED: ...")` lines under a guard, the same shape as this
repo's existing self-tests (`python -m app.cache`, `python -m
rag.pipeline`, `python -m rag.generation`, `python -m eval.utils`). That
convention was already established before this directory existed; these
follow it rather than introducing a second style. `pytest` doesn't
collect `def test_*()` functions from them (there aren't any), so
`pytest tests/` reports "0 items" — not a failure, just nothing in its
collection model to find. `python -m tests.run_all` is what actually runs
them.

## Why each module is its own subprocess

Several modules replace a module-level singleton with a fake
(`vectorstore._client`, `cache._client`, `retrieval.search`, ...) for the
duration of the process. Importing two such modules into one Python
process would let the second one's patch linger and silently affect the
first, depending on import order — the exact kind of cross-test
contamination that makes a suite's pass/fail meaningless. `run_all.py`
runs each as `python -m tests.<module>` in its own subprocess, verified
(see its own history) to actually detect a failure rather than always
reporting green regardless of content.

## What is intentionally not here

Seven files existed as work-in-progress scratch scripts and did not meet
the bar for a committed suite — listed here for the same reason the table
above exists: so "tests exist" is checkable, not asserted.

- **`test_calibrate_curve`, `test_corpus_version_bump`,
  `test_embedder_integration`** — pre-Checkpoint-6 relics. They reference
  `rag.retrieval._get_vectorstore` and FAISS imports that Checkpoint 6
  deleted entirely; confirmed failing against `origin/main` as it stands
  today, not just against this branch, so keeping them would mean
  committing tests that fail from the moment they land.
- **`regression_4c`** — asserts `settings.similarity_threshold` exists.
  That setting was deliberately removed when the confidence gate was
  removed; the assertion is checking for something that was correctly
  deleted.
- **`regression_4d`** — makes a real outbound HTTP call as part of its
  setup, which this sandbox's network policy blocks (`403`). Not a code
  defect; it needs its live call mocked before it belongs in an
  always-runnable suite.
- **`serve_test`, `serve_test_4d`** — not automated tests at all: each
  calls `uvicorn.run()` under `if __name__ == "__main__":` and blocks
  forever serving real HTTP, meant to be started by hand and poked with
  `curl` in a second terminal. `serve_test` is additionally stale - it
  patches `rag.retrieval._vectorstore`, an attribute Checkpoint 6 deleted
  along with the rest of the FAISS singleton.

The Checkpoint 6 and 7 PR descriptions specifically cited "15/21 tests
against a mocked Qdrant client" and the named suites above (gate removal,
doubled-Sources, cache scripts, embedder, overview chunks, UI errors) —
that's this directory's eight, verified by re-running each one before
writing this file. Earlier checkpoints' PRs (4c, 4d, the corpus-version
bump) predate this directory and aren't re-verified here; if any of their
claims need checking, the files above are the honest starting point for
that, not a blanket assurance covering them.

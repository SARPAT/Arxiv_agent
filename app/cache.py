"""Redis-backed caches for the retrieval path.

Three pieces of state live here: the current corpus version (used to
invalidate cached retrieval results whenever the corpus changes), a
query-embedding cache, and a full-retrieval-results cache. All of it is
strictly optional from the request-serving pipeline's correctness
standpoint — a cache miss or any Redis error falls back to live
computation (embedding via the model, or a fresh FAISS search) rather
than raising, so a cache outage only removes the speed benefit for that
one call, never changes the answer or fails the request. The one
exception is ``increment_corpus_version()``, called by
``ingestion/build_index.py`` rather than the request path - see its
docstring for why a failed version bump is deliberately not swallowed
the way everything else here is.

Bumping corpus_version was originally a manual step with nothing in code
tying it to an actual rebuild, which caused two separate debugging
cycles where a freshly rebuilt index kept getting served stale cached
retrieval results under the old version number - fixed by having
``ingestion/build_index.py`` call ``increment_corpus_version()`` itself
as the last step of a successful run, so this can no longer be forgotten.

Both cache keys include ``rag.embedder.cache_identifier()``, the same
invalidation-on-change protection ``corpus_version`` already gives against
document changes. This was found missing the hard way: after Checkpoint
4f swapped the embedding runtime (sentence-transformers -> ONNX Runtime),
``eval/calibrate_threshold.py`` silently kept returning the old runtime's
numbers because it was served stale cached embeddings/retrieval results
computed before the swap - flushing the cache by hand produced the
correct, different numbers.

This originally keyed on ``settings.embedding_model`` instead, which its
own docstring flagged as an incomplete fix: that logical display name was
(deliberately) left unchanged across the sentence-transformers -> ONNX
swap, so it would not have caught that exact incident, and the same gap
would recur for any future change confined to ``rag/embedder.py`` (e.g. a
different ONNX quantization). ``cache_identifier()`` closes this by
hashing that module's actual output-affecting config (the ONNX subpath
and pooling method) instead of relying on a separately-maintained display
string - see its docstring in ``rag/embedder.py``.

The retrieval cache key additionally folds in
``rag.generation.response_cache_identifier()`` - a hash of the generation
system prompt and model name - so a change to either invalidates prior
retrieval entries without a manual flush. This is the same class of gap
one level further downstream: the key encoded the corpus/embedder/query
but nothing about generation, so a prompt or model change (e.g. removing
the confidence gate and rewriting the prompt) left stale entries
reachable under an unchanged key until someone flushed by hand. The
embedding cache key does NOT carry this - a query's raw embedding doesn't
depend on the generation config, only the retrieval result's downstream
use does.

This module knows nothing about ``rag/``'s ``Document`` type on purpose:
callers pass and receive plain JSON-safe data (embedding vectors as
``list[float]``, retrieved chunks as ``{"page_content": ..., "metadata":
...}`` dicts), and ``rag/retrieval.py`` owns converting to and from its
own types.
"""

import hashlib
import json
import logging
import re

import redis

from app.config import settings
from app.json_utils import json_dumps_safe
from rag.embedder import cache_identifier
from rag.generation import response_cache_identifier

logger = logging.getLogger(__name__)

EMBEDDING_TTL_SECONDS = 86400
RETRIEVAL_TTL_SECONDS = 86400

CORPUS_VERSION_KEY = "corpus:version"

# Same construction pattern as app/session.py: one client, built once at
# import time, relying on redis-py's own connection pooling rather than
# opening a new connection per request.
_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _normalize_query(query: str) -> str:
    """Lowercase, strip, and collapse internal whitespace runs to a
    single space, so queries that only differ in casing or incidental
    whitespace share a cache entry."""
    return re.sub(r"\s+", " ", query.strip().lower())


def _query_hash(query: str) -> str:
    """Stable hash of the normalized query, used as the cache key suffix
    for both the embedding and retrieval caches."""
    return hashlib.sha256(_normalize_query(query).encode("utf-8")).hexdigest()


def get_corpus_version() -> int:
    """Return the current corpus version, defaulting to 1.

    The first read (whenever the key doesn't exist yet) also writes 1 to
    Redis, so the version is explicit going forward rather than an
    implicit default forever. Every cache read/write goes through this
    (or ``increment_corpus_version()`` below), so bumping the version is
    the only mechanism needed to invalidate stale retrieval results after
    a corpus change - see ``increment_corpus_version()``.
    """
    try:
        raw = _client.get(CORPUS_VERSION_KEY)
        if raw is None:
            _client.set(CORPUS_VERSION_KEY, "1")
            return 1
        return int(raw)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable reading corpus:version")
        return 1


def increment_corpus_version() -> int:
    """Atomically increment and return the new ``corpus:version``.

    Called by ``ingestion/build_index.py`` once a rebuild has fully
    succeeded (the new index already saved to disk) - the structural fix
    for two separate debugging cycles where a rebuilt index kept getting
    served stale cached retrieval results under the old corpus_version,
    because bumping it was a manual step someone had to remember (and,
    twice, forgot). With this, a rebuild can no longer silently leave old
    cache entries reachable under an unchanged version number.

    Deliberately does NOT degrade silently on a Redis error, unlike every
    other function in this module: ``get_corpus_version()``'s
    fallback-to-1 is safe because a request-time read gets a fresh chance
    to succeed on the very next call, but a failed bump here has no such
    retry - the caller (ingestion) must find out it didn't happen, since
    silently swallowing exactly this error is the bug this function
    exists to close. So ``redis.exceptions.RedisError`` is left to
    propagate rather than caught.

    Uses Redis ``INCR`` (atomic - race-free against a concurrent
    ``get_corpus_version()`` call from a live request) rather than a
    read-then-write, and correctly starts a nonexistent key at 1 - the
    same default ``get_corpus_version()`` uses for a fresh corpus - so
    this needs no special-casing for "first ever ingestion run".

    Only the retrieval cache is keyed on corpus_version (see
    ``get_cached_retrieval()``/``set_cached_retrieval()``) - the
    embedding cache is correctly left untouched, since a query's raw
    embedding doesn't depend on what's in the corpus, only the search
    result does.
    """
    return _client.incr(CORPUS_VERSION_KEY)


def get_cached_embedding(query: str) -> list[float] | None:
    """Return the cached embedding vector for ``query``, or ``None`` on a
    cache miss or Redis error."""
    key = f"embedding:{cache_identifier()}:{_query_hash(query)}"
    try:
        raw = _client.get(key)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable reading embedding cache (key=%s)", key)
        return None
    if raw is None:
        logger.info("embedding cache miss (key=%s)", key)
        return None
    logger.info("embedding cache hit (key=%s)", key)
    return json.loads(raw)


def set_cached_embedding(query: str, vector: list[float]) -> None:
    """Write ``vector`` to the embedding cache for ``query``."""
    key = f"embedding:{cache_identifier()}:{_query_hash(query)}"
    try:
        _client.set(key, json_dumps_safe(vector), ex=EMBEDDING_TTL_SECONDS)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable writing embedding cache (key=%s)", key)


def get_cached_retrieval(query: str, corpus_version: int) -> dict | None:
    """Return the cached ``{"chunks": [...], "scores": [...]}`` retrieval
    result for ``query`` at ``corpus_version``, or ``None`` on a cache
    miss or Redis error.

    The key folds in ``response_cache_identifier()`` (the generation
    prompt + model fingerprint) alongside ``corpus_version`` and the
    embedder identifier, so a prompt/model change invalidates prior
    entries without a manual flush - see ``rag/generation.py``."""
    key = (
        f"retrieval:{cache_identifier()}:{response_cache_identifier()}:"
        f"{corpus_version}:{_query_hash(query)}"
    )
    try:
        raw = _client.get(key)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable reading retrieval cache (key=%s)", key)
        return None
    if raw is None:
        logger.info("retrieval cache miss (key=%s)", key)
        return None
    logger.info("retrieval cache hit (key=%s)", key)
    return json.loads(raw)


def set_cached_retrieval(
    query: str, corpus_version: int, chunks: list[dict], scores: list[float]
) -> None:
    """Write a retrieval result for ``query`` at ``corpus_version`` to the
    retrieval cache. Key construction mirrors ``get_cached_retrieval()`` -
    including ``response_cache_identifier()`` - so a write and its read
    land on the same key only while the generation config is unchanged."""
    key = (
        f"retrieval:{cache_identifier()}:{response_cache_identifier()}:"
        f"{corpus_version}:{_query_hash(query)}"
    )
    value = json_dumps_safe({"chunks": chunks, "scores": scores})
    try:
        _client.set(key, value, ex=RETRIEVAL_TTL_SECONDS)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable writing retrieval cache (key=%s)", key)


if __name__ == "__main__":
    import fakeredis
    import numpy as np

    # A real embedder/FAISS index returns numpy.float32 values, not plain
    # Python floats - json.dumps chokes on those without json_dumps_safe's
    # default=float. Exercise both writers with actual numpy.float32
    # values (not floats that merely look like them) so this class of bug
    # is caught here instead of only against live FAISS output.
    _client = fakeredis.FakeRedis(decode_responses=True)

    set_cached_embedding("test query", [np.float32(0.1), np.float32(0.2)])
    embedding = get_cached_embedding("test query")
    assert embedding == [np.float32(0.1), np.float32(0.2)], embedding
    print("set_cached_embedding()/get_cached_embedding() handled numpy.float32 values without raising.")

    set_cached_retrieval(
        "test query",
        corpus_version=1,
        chunks=[{"page_content": "chunk text", "metadata": {"source": "doc0"}}],
        scores=[np.float32(0.4934097)],
    )
    retrieval = get_cached_retrieval("test query", corpus_version=1)
    assert retrieval["scores"] == [np.float32(0.4934097)], retrieval
    print(
        "set_cached_retrieval()/get_cached_retrieval() handled numpy.float32 "
        "scores without raising."
    )

    # A model/runtime swap must not silently serve a cache entry computed
    # by the old embedder - this exact protection was missing after
    # Checkpoint 4f's sentence-transformers -> ONNX swap. Cache keys now
    # derive from rag.embedder.cache_identifier() rather than
    # settings.embedding_model directly (that string is deliberately left
    # unchanged across runtime swaps - see rag/embedder.py's docstring),
    # so this exercises the actual invalidation path: patch this running
    # module's own cache_identifier name (via sys.modules[__name__], not
    # a fresh `import app.cache` - this script is already running *as*
    # app.cache under the name __main__, so a second import would create
    # a distinct module object with its own globals, silently patching a
    # copy that get_cached_embedding()/etc. above never actually read
    # from) to two values standing in for two different ONNX configs, and
    # confirm they land on different keys rather than colliding.
    import sys

    this_module = sys.modules[__name__]
    original_identifier = this_module.cache_identifier
    try:
        this_module.cache_identifier = lambda: "config-a"
        set_cached_embedding("shared query", [1.0, 2.0])
        set_cached_retrieval(
            "shared query",
            corpus_version=1,
            chunks=[{"page_content": "a", "metadata": {}}],
            scores=[0.1],
        )

        this_module.cache_identifier = lambda: "config-b"
        assert get_cached_embedding("shared query") is None, (
            "a different cache_identifier() must not see config-a's cached embedding"
        )
        assert get_cached_retrieval("shared query", corpus_version=1) is None, (
            "a different cache_identifier() must not see config-a's cached retrieval"
        )
        print(
            "PASSED: a different cache_identifier() changes the cache key - "
            "no collision between different embedder configs."
        )

        this_module.cache_identifier = lambda: "config-a"
        assert get_cached_embedding("shared query") == [1.0, 2.0]
        assert get_cached_retrieval("shared query", corpus_version=1) is not None
        print(
            "PASSED: switching back to the original cache_identifier() still finds "
            "its own cache entry."
        )
    finally:
        this_module.cache_identifier = original_identifier

    # cache_identifier() itself must actually change when the embedder's
    # output-affecting config changes (not just be swappable in a test) -
    # exercise the real function against a patched rag.embedder module
    # constant, the same kind of change (e.g. a different ONNX
    # quantization) that motivated this fix.
    import rag.embedder as embedder_module

    baseline = embedder_module.cache_identifier()
    original_subpath = embedder_module._ONNX_SUBPATH
    try:
        embedder_module._ONNX_SUBPATH = "onnx/model.onnx"  # different quantization
        changed = embedder_module.cache_identifier()
        assert changed != baseline, (
            "cache_identifier() must change when _ONNX_SUBPATH changes"
        )
        assert changed.startswith(embedder_module._HF_REPO + ":"), changed
        print(
            "PASSED: cache_identifier() changes when the embedder's real ONNX "
            "config changes, with no separate value to remember to update."
        )
    finally:
        embedder_module._ONNX_SUBPATH = original_subpath
    assert embedder_module.cache_identifier() == baseline

    # A generation prompt/model change must invalidate the RETRIEVAL cache
    # (the gap that masked the gate-removal change behind stale entries),
    # but must NOT touch the EMBEDDING cache (a query's embedding doesn't
    # depend on generation config). Same sys.modules[__name__] patch
    # rationale as above.
    _client.flushall()
    original_resp = this_module.response_cache_identifier
    try:
        this_module.response_cache_identifier = lambda: "gen-a"
        set_cached_embedding("q2", [3.0, 4.0])
        set_cached_retrieval(
            "q2",
            corpus_version=1,
            chunks=[{"page_content": "c", "metadata": {}}],
            scores=[0.2],
        )

        this_module.response_cache_identifier = lambda: "gen-b"
        assert get_cached_retrieval("q2", corpus_version=1) is None, (
            "a different response_cache_identifier() must not see gen-a's "
            "cached retrieval"
        )
        assert get_cached_embedding("q2") == [3.0, 4.0], (
            "the embedding cache key must NOT depend on generation config - "
            "it should still hit across a prompt/model change"
        )
        print(
            "PASSED: a generation prompt/model change invalidates the retrieval "
            "cache but leaves the embedding cache reachable."
        )

        this_module.response_cache_identifier = lambda: "gen-a"
        assert get_cached_retrieval("q2", corpus_version=1) is not None
        print(
            "PASSED: switching back to the original response_cache_identifier() "
            "still finds its own retrieval entry."
        )
    finally:
        this_module.response_cache_identifier = original_resp

    # response_cache_identifier() itself must actually change when the
    # system prompt or the generation model name changes - exercise the
    # real function against patched rag.generation module state, the exact
    # kind of change (prompt rewrite / model swap) that motivated this.
    import rag.generation as generation_module

    resp_baseline = generation_module.response_cache_identifier()

    original_prompt = generation_module.SYSTEM_PROMPT
    try:
        generation_module.SYSTEM_PROMPT = original_prompt + "\n(extra rule)"
        assert generation_module.response_cache_identifier() != resp_baseline, (
            "response_cache_identifier() must change when SYSTEM_PROMPT changes"
        )
    finally:
        generation_module.SYSTEM_PROMPT = original_prompt
    assert generation_module.response_cache_identifier() == resp_baseline

    original_model = generation_module.settings.generation_model
    try:
        generation_module.settings.generation_model = "some/other-model"
        assert generation_module.response_cache_identifier() != resp_baseline, (
            "response_cache_identifier() must change when the model name changes"
        )
    finally:
        generation_module.settings.generation_model = original_model
    assert generation_module.response_cache_identifier() == resp_baseline
    print(
        "PASSED: response_cache_identifier() changes on a SYSTEM_PROMPT or "
        "generation_model change, and is stable otherwise."
    )

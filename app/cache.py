"""Redis-backed caches for the retrieval path.

Three pieces of state live here: the current corpus version (used to
invalidate cached retrieval results whenever the corpus changes), a
query-embedding cache, and a full-retrieval-results cache. All of it is
strictly optional from the pipeline's correctness standpoint — a cache
miss or any Redis error falls back to live computation (embedding via
the model, or a fresh FAISS search) rather than raising, so a cache
outage only removes the speed benefit for that one call, never changes
the answer or fails the request.

Both cache keys include ``settings.embedding_model``, the same
invalidation-on-change protection ``corpus_version`` already gives against
document changes. This was found missing the hard way: after Checkpoint
4f swapped the embedding runtime (sentence-transformers -> ONNX Runtime),
``eval/calibrate_threshold.py`` silently kept returning the old runtime's
numbers because it was served stale cached embeddings/retrieval results
computed before the swap - flushing the cache by hand produced the
correct, different numbers.

Caveat: this only invalidates on a *model name* change. Checkpoint 4f
deliberately kept ``embedding_model`` as a logical name unchanged across
the sentence-transformers -> ONNX switch (see ``rag/embedder.py``), so by
itself this fix would not have caught that exact swap - only a change
that also changes ``settings.embedding_model``'s string value. Closing
that gap fully would need a second identifier that changes with the
concrete embedding *pipeline/runtime*, not just the model name.

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
    implicit default forever. Nothing increments this yet — there is no
    document upload path in this checkpoint — but a future ingestion step
    can bump it with a single ``INCR corpus:version`` call without
    touching any caching code here, since every cache read already goes
    through this function.
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


def get_cached_embedding(query: str) -> list[float] | None:
    """Return the cached embedding vector for ``query``, or ``None`` on a
    cache miss or Redis error."""
    key = f"embedding:{settings.embedding_model}:{_query_hash(query)}"
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
    key = f"embedding:{settings.embedding_model}:{_query_hash(query)}"
    try:
        _client.set(key, json_dumps_safe(vector), ex=EMBEDDING_TTL_SECONDS)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable writing embedding cache (key=%s)", key)


def get_cached_retrieval(query: str, corpus_version: int) -> dict | None:
    """Return the cached ``{"chunks": [...], "scores": [...]}`` retrieval
    result for ``query`` at ``corpus_version``, or ``None`` on a cache
    miss or Redis error."""
    key = f"retrieval:{settings.embedding_model}:{corpus_version}:{_query_hash(query)}"
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
    retrieval cache."""
    key = f"retrieval:{settings.embedding_model}:{corpus_version}:{_query_hash(query)}"
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
    # Checkpoint 4f's sentence-transformers -> ONNX swap. Change
    # settings.embedding_model between two calls with the same query text
    # and confirm they land on different keys rather than colliding.
    original_model = settings.embedding_model
    try:
        settings.embedding_model = "model-a"
        set_cached_embedding("shared query", [1.0, 2.0])
        set_cached_retrieval(
            "shared query",
            corpus_version=1,
            chunks=[{"page_content": "a", "metadata": {}}],
            scores=[0.1],
        )

        settings.embedding_model = "model-b"
        assert get_cached_embedding("shared query") is None, (
            "a different embedding_model must not see model-a's cached embedding"
        )
        assert get_cached_retrieval("shared query", corpus_version=1) is None, (
            "a different embedding_model must not see model-a's cached retrieval"
        )
        print(
            "PASSED: changing settings.embedding_model changes the cache key - "
            "no collision between different models/runtimes."
        )

        settings.embedding_model = "model-a"
        assert get_cached_embedding("shared query") == [1.0, 2.0]
        assert get_cached_retrieval("shared query", corpus_version=1) is not None
        print(
            "PASSED: switching back to the original embedding_model still finds "
            "its own cache entry."
        )
    finally:
        settings.embedding_model = original_model

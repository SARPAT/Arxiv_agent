"""Redis-backed conversation history store.

Each session's history is stored at the key ``chat:{session_id}:history``
as a JSON-encoded list of ``{"role": "user"|"assistant", "content": "..."}``
dicts. The TTL is a rolling 1 hour: every ``append_turn`` call rewrites
the key with ``SET ... EX 3600`` (a single atomic command, not a separate
``EXPIRE`` call), so an active conversation's history never expires
mid-use while an abandoned one is cleaned up automatically. A single
module-level client is reused for every call, relying on redis-py's own
connection pooling rather than opening a new connection per request.

Any Redis error on read or write is caught, logged, and degrades to an
empty or unpersisted history instead of raising — a session-store hiccup
should not fail a request whose answer has already been generated and
streamed to the user.
"""

import json
import logging

import redis

from app.config import settings
from rag.vectorstore import PUBLIC_TENANT_ID

logger = logging.getLogger(__name__)

TTL_SECONDS = 3600

_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"chat:{session_id}:history"


def _upload_key(session_id: str) -> str:
    return f"chat:{session_id}:upload"


def get_history(session_id: str) -> list[dict[str, str]]:
    """Return the stored conversation history for ``session_id``.

    Returns an empty list both when this session has no history yet and
    when Redis itself is unreachable — the caller has no way to tell
    those two cases apart, which is intentional: either way, the correct
    behavior is to proceed with no prior context rather than fail.
    """
    try:
        raw = _client.get(_key(session_id))
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable reading history for %s", session_id)
        return []
    return json.loads(raw) if raw is not None else []


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Record one user/assistant turn onto ``session_id``'s history and
    reset its TTL to ``TTL_SECONDS`` from now.

    Reads through ``get_history`` (so a read failure here already
    degrades to starting from ``[]``, the same as any other call to it).
    If the write itself fails, the failure is logged and swallowed rather
    than raised: the answer this turn is recording was already streamed
    to the user, so failing to persist it for the next turn is a
    degraded experience, not a request failure.
    """
    history = get_history(session_id)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})
    try:
        _client.set(_key(session_id), json.dumps(history), ex=TTL_SECONDS)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable writing history for %s", session_id)


def set_upload(session_id: str, filename: str, chunk_count: int) -> None:
    """Record that ``session_id`` has an uploaded document in Qdrant.

    This marker is what ``tenant_ids_for()`` reads on every message, so
    retrieval never has to ask Qdrant whether a session has uploads. It
    carries the same rolling TTL as the history key.

    Unlike everything else in this module, a Redis failure here is NOT
    swallowed - same reasoning as ``app/cache.py``'s
    ``increment_corpus_version()``. The points are already in Qdrant by
    the time this runs; losing the marker would leave a document the user
    was told was ingested that no query will ever search, which is worse
    than a failed upload the user can simply retry.
    """
    _client.set(
        _upload_key(session_id),
        json.dumps({"filename": filename, "chunk_count": chunk_count}),
        ex=TTL_SECONDS,
    )


def get_upload(session_id: str) -> dict | None:
    """Return ``{"filename", "chunk_count"}`` for this session's uploaded
    document, or ``None`` if it has none (or Redis is unreachable).

    Reading refreshes the TTL, so an active conversation's upload cannot
    expire out from under it while its history is still being extended -
    the same rolling-window behaviour ``append_turn`` gives history, and
    the reason ``GETEX`` is used rather than a plain ``GET``.
    """
    if not session_id:
        return None
    try:
        raw = _client.getex(_upload_key(session_id), ex=TTL_SECONDS)
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable reading upload state for %s", session_id)
        return None
    return json.loads(raw) if raw is not None else None


def clear_upload(session_id: str) -> None:
    """Forget this session's uploaded document.

    Called before the Qdrant points are replaced, so that a failure partway
    through an upload leaves the session with no marker rather than one
    pointing at a document that is no longer there - retrieval then
    correctly falls back to the shared corpus alone."""
    try:
        _client.delete(_upload_key(session_id))
    except redis.exceptions.RedisError:
        logger.warning("Redis unavailable clearing upload state for %s", session_id)


def tenant_ids_for(session_id: str) -> list[str]:
    """The Qdrant tenants a query from ``session_id`` should search.

    ``["public", session_id]`` once that session has uploaded a document,
    ``["public"]`` otherwise. Answered from the Redis marker above rather
    than by querying Qdrant, so the common case (no upload) costs nothing
    per message beyond a single Redis read that also refreshes the TTL."""
    if get_upload(session_id):
        return [PUBLIC_TENANT_ID, session_id]
    return [PUBLIC_TENANT_ID]

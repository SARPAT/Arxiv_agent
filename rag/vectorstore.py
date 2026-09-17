"""Qdrant Cloud vector store — the only module that talks to Qdrant.

Checkpoint 6 replaced the local FAISS index (a committed binary loaded
into the process at startup) with a Qdrant Cloud collection. Both
ingestion (``ingestion/build_index.py``) and retrieval
(``rag/retrieval.py``) go through this module; nothing else constructs a
``QdrantClient``, and no connection value is hardcoded anywhere - it all
comes from ``app.config.settings``.

Layout, all locked decisions:

- **One collection, many tenants.** Every point carries a ``tenant_id``
  payload field, indexed with ``is_tenant=True`` so Qdrant physically
  co-locates a tenant's vectors on disk (its documented recommendation
  for the single-collection multi-tenancy pattern). The current corpus is
  all ``"public"``; Checkpoint 7's user uploads will use the session id,
  which is why ``search()`` takes a *list* of tenant ids and not one
  string - it can then query ``["public", session_id]`` in one call.
- **Every filtered payload field must be indexed.** Qdrant Cloud's free
  tier runs in strict mode, which rejects a filter on an unindexed field
  with a 400 instead of falling back to a scan. So ``tenant_id`` plus
  everything in ``FILTERABLE_PAYLOAD_FIELDS`` gets an index at
  ``ensure_collection()`` time, and ``count_points()`` refuses a filter on
  anything outside that set.
- **Named vector ``"dense"``**, not an unnamed default. A collection's
  vector configuration is immutable after creation, so creating it
  unnamed would mean deleting the collection and re-ingesting just to add
  a second (e.g. sparse) vector later. Sparse/hybrid retrieval is out of
  scope here; this naming is the only concession made to it.
- **Cosine distance.** The embedder already L2-normalizes its output, so
  cosine and the old L2 distance rank identically - but the *number* and
  its direction change: **higher is now better** (roughly 0-1), where
  FAISS's L2 distance was lower-is-better and unbounded. Every score
  comparison in the codebase was inverted for this; see the PR.
"""

import hashlib
import logging
import threading
import time
import uuid

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from rag.embedder import get_embedder

logger = logging.getLogger(__name__)

# Batch size for upserts. Small enough to keep each request well inside
# Qdrant Cloud's payload limits, large enough that a ~334-chunk corpus is
# a handful of round trips rather than hundreds.
UPSERT_BATCH_SIZE = 100

# The tenant field gets its own index type (``is_tenant=True``), so it is
# named separately from the plain keyword-indexed fields below.
TENANT_FIELD = "tenant_id"

# The tenant holding the shared arXiv corpus. Named once here, in the
# module that owns tenancy, and imported by ingestion and retrieval - the
# literal was previously written out in three places, and a typo in any
# one of them would have been a silently empty search or, worse, a corpus
# written under a tenant nothing queries.
PUBLIC_TENANT_ID = "public"

# Payload fields that anything filters on, beyond the tenant. Qdrant Cloud
# free-tier clusters run in strict mode, which **rejects a filter on any
# unindexed payload field** with a 400 rather than falling back to a scan -
# so every field used in a filter must be indexed here. Found the hard way:
# the ingestion summary filters by chunk_type and paper_key via
# count_points(), and on a fresh cluster it failed with
#   400: Index required but not found for "chunk_type" ... [keyword]
# after the upsert itself had already succeeded.
FILTERABLE_PAYLOAD_FIELDS = ("chunk_type", "paper_key")

# Every payload field it is legal to filter on. count_points() checks
# against this so a filter on an unindexed field fails immediately, with a
# clear message, instead of as a 400 from the server mid-run.
INDEXED_PAYLOAD_FIELDS = (TENANT_FIELD, *FILTERABLE_PAYLOAD_FIELDS)

# Fixed namespace for deterministic point ids. Any stable UUID works; it
# only has to never change, or every re-ingest would write new points
# alongside the old ones instead of overwriting them.
_POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

_client: QdrantClient | None = None
_client_lock = threading.Lock()
_embedding_dim: int | None = None


def get_client() -> QdrantClient:
    """Return the shared ``QdrantClient``, constructing it on first use.

    A module-level singleton for the same reason the embedder and the
    ChatNVIDIA client are cached: it is a stateless, reusable connection
    pool, not per-user state. Constructing one per request would add a TLS
    handshake to every call.
    """
    global _client
    if _client is None:
        with _client_lock:
            # Double-checked for the same reason rag/embedder.py's
            # get_embedder() is: uvicorn's threadpool plus app/api.py's
            # startup warm-up mean two threads can reach this at once,
            # and a second client is a second connection pool.
            if _client is None:
                _client = QdrantClient(
                    url=settings.qdrant_url, api_key=settings.qdrant_api_key
                )
    return _client


def embedding_dimension() -> int:
    """The embedder's actual output dimension, probed once and cached.

    Read from ``rag/embedder.py`` rather than hardcoded so the collection
    can never be created at a size that has drifted from what the app
    actually queries with - the failure mode that a hardcoded constant in
    ingestion caused in Checkpoint 4e.
    """
    global _embedding_dim
    if _embedding_dim is None:
        _embedding_dim = len(get_embedder().embed_query("dimension probe"))
    return _embedding_dim


def _create_payload_index(client: QdrantClient, field: str, schema) -> None:
    """Create one payload index, tolerating it already existing.

    ``ensure_collection()`` runs on every ingestion and these indexes now
    exist on the live cluster, so "already there" is the normal case, not
    an error. Only an already-exists conflict is swallowed: any other
    ``UnexpectedResponse`` (auth, quota, a bad schema, an unreachable
    cluster) is re-raised, because silently continuing past those is how a
    collection ends up subtly misconfigured and only fails much later, at
    query time.
    """
    try:
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name=field,
            field_schema=schema,
        )
    except UnexpectedResponse as exc:
        already_exists = exc.status_code == 409 or (
            b"already exists" in (exc.content or b"").lower()
        )
        if not already_exists:
            raise


def ensure_collection() -> None:
    """Create the collection and its tenant payload index if absent.

    Idempotent: safe to call on every ingestion run. The collection is
    created with the named vector ``"dense"`` (a dict, not a bare
    ``VectorParams``) at the embedder's reported size, using cosine
    distance.
    """
    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=embedding_dimension(),
                    distance=models.Distance.COSINE,
                )
            },
        )

    # Payload indexes run unconditionally rather than only on the create
    # path, so a collection created without them (or created by hand)
    # still converges to the right shape instead of silently missing one.
    # Which fields are indexed is not cosmetic: strict mode rejects a
    # filter on an unindexed field outright - see FILTERABLE_PAYLOAD_FIELDS.
    #
    # tenant_id keeps its own schema (KeywordIndexParams with
    # is_tenant=True) rather than a plain keyword index - that flag is what
    # makes Qdrant co-locate a tenant's vectors on disk, and it would be
    # lost by downgrading it to PayloadSchemaType.KEYWORD.
    _create_payload_index(
        client,
        TENANT_FIELD,
        models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )
    for field in FILTERABLE_PAYLOAD_FIELDS:
        _create_payload_index(client, field, models.PayloadSchemaType.KEYWORD)


def _tenant_condition(tenant_ids: list[str]) -> models.FieldCondition:
    """Match any of ``tenant_ids`` on the indexed tenant field. Shared by
    every read and write path so they cannot disagree about what "this
    tenant's points" means."""
    return models.FieldCondition(
        key=TENANT_FIELD, match=models.MatchAny(any=tenant_ids)
    )


def point_id(tenant_id: str, metadata: dict, text: str) -> str:
    """Deterministic UUID5 for one chunk, so re-ingesting overwrites
    rather than appends.

    Derived from the tenant plus the chunk's stable identity
    (``paper_key`` + ``chunk_type`` + a hash of the text). ``tenant_id``
    is part of the key so that the same text uploaded by two tenants
    stays two distinct points instead of one clobbering the other -
    which matters as soon as Checkpoint 7 adds per-session uploads.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    name = (
        f"{tenant_id}:{metadata.get('paper_key')}:"
        f"{metadata.get('chunk_type')}:{content_hash}"
    )
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, name))


def upsert_chunks(
    chunks: list, tenant_id: str, timings: dict[str, float] | None = None
) -> int:
    """Embed and upsert ``chunks`` under ``tenant_id``. Returns the count.

    ``chunks`` are langchain ``Document``s (what the ingestion pipeline
    already produces). Every metadata field they carry is copied into the
    payload verbatim alongside the chunk text and the tenant id - losing
    one silently would break, for example, ``eval/run_eval.py``'s
    ``paper_key`` matching.

    ``timings``, when given, accumulates seconds spent under ``"embed"``
    and ``"upsert"``. Those two are only separable in here - from a
    caller's side they are one opaque call - and telling them apart is the
    whole point when an upload is timing out: local CPU work and a network
    round trip to Qdrant want completely different fixes. Optional and
    additive so existing callers (``ingestion/build_index.py``) are
    untouched.
    """
    client = get_client()
    embedder = get_embedder()
    total = 0

    def _record(stage: str, seconds: float) -> None:
        if timings is not None:
            timings[stage] = timings.get(stage, 0.0) + seconds

    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch = chunks[start : start + UPSERT_BATCH_SIZE]
        t0 = time.perf_counter()
        vectors = embedder.embed_documents([chunk.page_content for chunk in batch])
        _record("embed", time.perf_counter() - t0)

        points = []
        for chunk, vector in zip(batch, vectors):
            payload = dict(chunk.metadata)
            payload["text"] = chunk.page_content
            payload["tenant_id"] = tenant_id
            points.append(
                models.PointStruct(
                    id=point_id(tenant_id, chunk.metadata, chunk.page_content),
                    vector={"dense": vector},
                    payload=payload,
                )
            )

        t0 = time.perf_counter()
        client.upsert(collection_name=settings.qdrant_collection, points=points)
        _record("upsert", time.perf_counter() - t0)
        total += len(points)

    return total


def search(
    query_vector: list[float], k: int, tenant_ids: list[str]
) -> list[tuple[dict, float]]:
    """Return the ``k`` best matches across ``tenant_ids``, best first.

    Each result is a ``({"page_content": ..., "metadata": {...}}, score)``
    pair - the same chunk shape ``app/cache.py`` stores and
    ``rag/retrieval.py`` converts to ``Document``s, so this swap doesn't
    ripple past retrieval.

    **The score is cosine similarity: higher is better**, unlike the FAISS
    L2 distance this replaced. Qdrant returns results already ordered
    best-first, so callers must not re-sort ascending.
    """
    response = get_client().query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        using="dense",
        limit=k,
        query_filter=models.Filter(must=[_tenant_condition(tenant_ids)]),
        with_payload=True,
    )

    results = []
    for point in response.points:
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        # Everything else in the payload (including tenant_id) is metadata,
        # which keeps every field the FAISS documents carried.
        results.append(({"page_content": text, "metadata": payload}, point.score))
    return results


def count_points(
    tenant_ids: list[str] | None = None,
    field: str | None = None,
    value: str | None = None,
) -> int:
    """Exact point count, optionally narrowed to tenants and/or one
    payload field value. Used for the ingestion verification summary and
    ``scripts/verify_qdrant.py``'s breakdowns.

    ``field`` must be one of ``INDEXED_PAYLOAD_FIELDS``: strict mode
    rejects a filter on an unindexed field with a 400, so this fails fast
    and locally rather than part-way through a run against the cluster.
    """
    if field is not None and field not in INDEXED_PAYLOAD_FIELDS:
        raise ValueError(
            f"cannot filter on unindexed payload field {field!r}; "
            f"indexed fields are {list(INDEXED_PAYLOAD_FIELDS)}. Add it to "
            "FILTERABLE_PAYLOAD_FIELDS so ensure_collection() indexes it."
        )

    conditions = []
    if tenant_ids:
        conditions.append(_tenant_condition(tenant_ids))
    if field is not None:
        conditions.append(
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
        )

    return get_client().count(
        collection_name=settings.qdrant_collection,
        count_filter=models.Filter(must=conditions) if conditions else None,
        exact=True,
    ).count


def delete_by_tenant(tenant_id: str) -> int:
    """Delete every point belonging to ``tenant_id``. Returns how many.

    **Refuses the public tenant, unconditionally.** One wrong argument
    here would wipe the shared corpus, and the only way back is a full
    Colab re-ingestion - so this is a hard guard rather than a caller's
    responsibility. An empty tenant id is refused for the same reason: it
    is what a missing session id looks like, and it must never be treated
    as "delete whatever matches".

    The count is read before the delete because Qdrant's delete reports an
    operation status, not a number of points removed.
    """
    if not tenant_id or tenant_id == PUBLIC_TENANT_ID:
        raise ValueError(
            f"refusing to delete points for tenant {tenant_id!r}: the shared "
            "corpus is not deletable through this function, and an empty "
            "tenant id is never a valid target"
        )

    deleted = count_points(tenant_ids=[tenant_id])
    if deleted:
        get_client().delete(
            collection_name=settings.qdrant_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[_tenant_condition([tenant_id])])
            ),
            wait=True,
        )
    return deleted


def list_tenants(limit: int = 1000) -> list[str]:
    """Every distinct ``tenant_id`` currently in the collection.

    Uses Qdrant's facet aggregation (available in the pinned client) with
    ``exact=True`` rather than scrolling every point, so the cost is a
    single aggregate request instead of one pass over the corpus. Only
    ``scripts/cleanup_orphaned_uploads.py`` needs this; ``limit`` caps how
    many distinct tenants come back, and hitting that cap is logged rather
    than silently truncating a cleanup run's view of what exists.
    """
    hits = get_client().facet(
        collection_name=settings.qdrant_collection,
        key=TENANT_FIELD,
        limit=limit,
        exact=True,
    ).hits
    if len(hits) >= limit:
        logger.warning(
            "list_tenants() hit its limit of %d distinct tenants; there may be "
            "more that this call cannot see - re-run with a higher limit",
            limit,
        )
    return [hit.value for hit in hits]

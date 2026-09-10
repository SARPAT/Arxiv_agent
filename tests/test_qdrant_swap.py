"""Verify the FAISS -> Qdrant swap (Checkpoint 6) with a mocked Qdrant
client and mocked embedder - no network. Covers: named-vector collection
config, tenant payload index with is_tenant, deterministic/idempotent
point ids, full metadata preservation, tenant-filtered search, the
higher-is-better score direction reaching retrieval best-first, the
retrieval cache key's backend segment, and the inverted calibration
comparisons."""
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from qdrant_client import models

import rag.vectorstore as vs
from app.config import settings

DIM = 384


class FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * DIM

    def embed_documents(self, texts):
        return [[0.1] * DIM for _ in texts]


# --- 1. ensure_collection(): named vector "dense", COSINE, size from embedder ---
client = MagicMock()
client.collection_exists.return_value = False
vs._embedding_dim = None
with patch.object(vs, "get_client", return_value=client), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    vs.ensure_collection()

kwargs = client.create_collection.call_args.kwargs
vectors_config = kwargs["vectors_config"]
assert isinstance(vectors_config, dict), "vectors_config must be a dict (named vector)"
assert set(vectors_config) == {"dense"}, vectors_config
assert vectors_config["dense"].size == DIM, "size must come from the embedder"
assert vectors_config["dense"].distance == models.Distance.COSINE
assert kwargs["collection_name"] == settings.qdrant_collection
print("PASSED: collection created with named vector 'dense', COSINE, embedder-derived size.")

# --- 1b. ALL THREE payload indexes created, with the right schema each ---
# Strict mode on free-tier clusters rejects filtering on an unindexed
# payload field, so chunk_type/paper_key must be indexed too - the real
# failure was "400: Index required but not found for 'chunk_type'".
by_field = {
    c.kwargs["field_name"]: c.kwargs["field_schema"]
    for c in client.create_payload_index.call_args_list
}
assert set(by_field) == {"tenant_id", "chunk_type", "paper_key"}, by_field

# tenant_id keeps its own schema - NOT downgraded to a plain keyword index.
tenant_schema = by_field["tenant_id"]
assert isinstance(tenant_schema, models.KeywordIndexParams), type(tenant_schema)
assert tenant_schema.is_tenant is True, "is_tenant must be True"
assert tenant_schema.type == models.KeywordIndexType.KEYWORD

# The other two are plain keyword indexes.
for field in ("chunk_type", "paper_key"):
    assert by_field[field] == models.PayloadSchemaType.KEYWORD, (field, by_field[field])
print("PASSED: all three payload indexes created; tenant_id keeps is_tenant=True, others plain KEYWORD.")

# The indexed set must cover exactly what count_points is allowed to filter on.
assert set(vs.INDEXED_PAYLOAD_FIELDS) == {"tenant_id", "chunk_type", "paper_key"}
assert vs.TENANT_FIELD == "tenant_id"
assert set(vs.FILTERABLE_PAYLOAD_FIELDS) == {"chunk_type", "paper_key"}
print("PASSED: field-name constants are defined in one place and cover the indexed set.")

# Idempotent: existing collection -> no create_collection, but all indexes re-set.
client2 = MagicMock()
client2.collection_exists.return_value = True
with patch.object(vs, "get_client", return_value=client2), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    vs.ensure_collection()
assert client2.create_collection.call_count == 0, "must not recreate an existing collection"
assert client2.create_payload_index.call_count == 3, client2.create_payload_index.call_count
print("PASSED: ensure_collection() is idempotent on an existing collection (3 indexes re-set).")

# --- 1c. A re-run where the indexes ALREADY EXIST must not raise ---
from qdrant_client.http.exceptions import UnexpectedResponse

def _conflict(*args, **kwargs):
    raise UnexpectedResponse(
        status_code=409,
        reason_phrase="Conflict",
        content=b'{"status":{"error":"Index already exists"}}',
        headers={},
    )

client3a = MagicMock()
client3a.collection_exists.return_value = True
client3a.create_payload_index.side_effect = _conflict
with patch.object(vs, "get_client", return_value=client3a), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    vs.ensure_collection()  # must not raise
assert client3a.create_payload_index.call_count == 3
print("PASSED: re-running against a cluster where the indexes already exist does not raise.")

# ...but an UNRELATED error is NOT swallowed (no bare except).
def _forbidden(*args, **kwargs):
    raise UnexpectedResponse(
        status_code=403,
        reason_phrase="Forbidden",
        content=b'{"status":{"error":"Access denied"}}',
        headers={},
    )

client3b = MagicMock()
client3b.collection_exists.return_value = True
client3b.create_payload_index.side_effect = _forbidden
with patch.object(vs, "get_client", return_value=client3b), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    try:
        vs.ensure_collection()
        raise AssertionError("a 403 must propagate, not be swallowed")
    except UnexpectedResponse as exc:
        assert exc.status_code == 403
print("PASSED: an unrelated UnexpectedResponse (403) propagates instead of being swallowed.")

# --- 1d. count_points() refuses a filter on an unindexed field, locally ---
client3c = MagicMock()
with patch.object(vs, "get_client", return_value=client3c):
    for ok_field in ("chunk_type", "paper_key"):
        vs.count_points(field=ok_field, value="x")  # must not raise
    try:
        vs.count_points(field="Title", value="Attention Is All You Need")
        raise AssertionError("filtering an unindexed field must raise")
    except ValueError as exc:
        assert "unindexed payload field" in str(exc) and "Title" in str(exc), str(exc)
print("PASSED: count_points() rejects an unindexed-field filter before hitting the cluster.")

# --- 2. Deterministic point ids: stable across runs, distinct across tenants ---
meta = {"paper_key": "attention", "chunk_type": "body"}
a1 = vs.point_id("public", meta, "some chunk text")
a2 = vs.point_id("public", meta, "some chunk text")
b = vs.point_id("session-xyz", meta, "some chunk text")
c = vs.point_id("public", meta, "different text")
assert a1 == a2, "same tenant+chunk must yield the same id (idempotent re-ingest)"
assert a1 != b, "different tenants must not collide"
assert a1 != c, "different content must not collide"
import uuid as _uuid
_uuid.UUID(a1)  # must be a valid UUID for Qdrant
print("PASSED: point ids are deterministic, tenant-scoped, content-scoped, valid UUIDs.")

# --- 3. upsert_chunks: preserves EVERY metadata field + text + tenant_id ---
# The 7 fields audited off the committed FAISS index.
full_meta = {
    "paper_key": "attention",
    "chunk_type": "body",
    "arxiv_id": "1706.03762",
    "Title": "Attention Is All You Need",
    "Published": "2023-08-02",
    "Authors": "Ashish Vaswani, ...",
    "Summary": "The dominant sequence transduction models ...",
}
chunks = [Document(page_content="chunk text here", metadata=dict(full_meta))]
client3 = MagicMock()
with patch.object(vs, "get_client", return_value=client3), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    n = vs.upsert_chunks(chunks, tenant_id="public")
assert n == 1
point = client3.upsert.call_args.kwargs["points"][0]
assert set(point.vector) == {"dense"}, "vector must be under the named key 'dense'"
assert len(point.vector["dense"]) == DIM
for field, value in full_meta.items():
    assert point.payload[field] == value, f"metadata field {field} lost"
assert point.payload["text"] == "chunk text here"
assert point.payload["tenant_id"] == "public"
print("PASSED: upsert preserves all 7 audited metadata fields + text + tenant_id, vector named 'dense'.")

# Batching: 250 chunks -> 3 upsert calls at batch size 100.
many = [Document(page_content=f"c{i}", metadata=dict(full_meta)) for i in range(250)]
client4 = MagicMock()
with patch.object(vs, "get_client", return_value=client4), patch.object(
    vs, "get_embedder", return_value=FakeEmbedder()
):
    assert vs.upsert_chunks(many, tenant_id="public") == 250
assert client4.upsert.call_count == 3, client4.upsert.call_count
print("PASSED: upsert batches at 100 (250 chunks -> 3 calls).")

# --- 4. search(): tenant filter is MatchAny over a LIST; returns best-first ---
def fake_point(score, paper_key, text):
    p = MagicMock()
    p.score = score
    p.payload = {"text": text, "paper_key": paper_key, "chunk_type": "body",
                 "Title": "T", "tenant_id": "public"}
    return p

client5 = MagicMock()
client5.query_points.return_value = MagicMock(
    points=[fake_point(0.91, "attention", "best"), fake_point(0.72, "bert", "second")]
)
with patch.object(vs, "get_client", return_value=client5):
    results = vs.search([0.1] * DIM, k=2, tenant_ids=["public", "session-1"])

qkwargs = client5.query_points.call_args.kwargs
assert qkwargs["using"] == "dense", "must query the named vector"
assert qkwargs["limit"] == 2
assert qkwargs["with_payload"] is True
cond = qkwargs["query_filter"].must[0]
assert cond.key == "tenant_id"
assert cond.match.any == ["public", "session-1"], "tenant_ids must pass through as a list"
# Shape + score direction
(chunk0, score0), (chunk1, score1) = results
assert chunk0["page_content"] == "best" and chunk0["metadata"]["paper_key"] == "attention"
assert "text" not in chunk0["metadata"], "text must move to page_content, not stay in metadata"
assert score0 == 0.91 and score1 == 0.72
assert score0 > score1, "higher-is-better, best first"
print("PASSED: search() filters by tenant list, queries 'dense', returns best-first higher-is-better.")

# --- 5. retrieval.retrieve(): passes tenant ["public"], preserves best-first order ---
import fakeredis
import app.cache as cache_mod

cache_mod._client = fakeredis.FakeRedis(decode_responses=True)
import rag.retrieval as retrieval

captured = {}
def fake_search(vector, k, tenant_ids):
    captured["tenant_ids"] = tenant_ids
    captured["k"] = k
    return [
        ({"page_content": "best", "metadata": {"paper_key": "attention"}}, 0.91),
        ({"page_content": "second", "metadata": {"paper_key": "bert"}}, 0.72),
    ]

with patch.object(retrieval, "search", fake_search), patch.object(
    retrieval, "get_embedder", return_value=FakeEmbedder()
):
    out = retrieval.retrieve("what is attention?", k=2)

assert captured["tenant_ids"] == ["public"], captured["tenant_ids"]
assert [d.page_content for d, _ in out] == ["best", "second"], "order must be preserved"
assert out[0][1] > out[1][1], "best-first must survive retrieve()"
assert out[0][0].metadata["paper_key"] == "attention"
print("PASSED: retrieve() searches tenant ['public'] and preserves Qdrant's best-first order.")

# Warm cache returns the same order/scores (round-trips through Redis JSON).
with patch.object(retrieval, "search", fake_search), patch.object(
    retrieval, "get_embedder", return_value=FakeEmbedder()
):
    warm = retrieval.retrieve("what is attention?", k=2)
assert [d.page_content for d, _ in warm] == ["best", "second"]
assert warm[0][1] == 0.91
print("PASSED: cached retrieval round-trips order and higher-is-better scores intact.")

# --- 6. Retrieval cache key carries the qdrant backend segment; embedding key doesn't ---
from app.cache import RETRIEVAL_BACKEND
assert RETRIEVAL_BACKEND == "qdrant"
keys = cache_mod._client.keys("*")
retrieval_keys = [k for k in keys if k.startswith("retrieval:")]
embedding_keys = [k for k in keys if k.startswith("embedding:")]
assert retrieval_keys and all(k.startswith("retrieval:qdrant:") for k in retrieval_keys), retrieval_keys
assert embedding_keys and not any(":qdrant:" in k for k in embedding_keys), embedding_keys
print("PASSED: retrieval key = 'retrieval:qdrant:...'; embedding key unchanged (no backend segment).")

# --- 7. calibrate_threshold: comparisons inverted for higher-is-better ---
import eval.calibrate_threshold as calib

# in-corpus (label 1) score high, out-of-corpus (label 0) score low -> perfectly
# separable under higher-is-better. AUC must be 1.0, not 0.0 (which is what a
# missed inversion would produce).
labels = [1, 1, 0, 0]
scores = [0.9, 0.85, 0.3, 0.2]
pts = calib.roc_points(labels, scores)
auc = calib.roc_auc(pts)
assert auc == 1.0, f"expected AUC 1.0 under higher-is-better, got {auc}"
t, fpr, tpr = calib.select_threshold(pts, min_specificity=1.0)
assert tpr == 1.0 and fpr == 0.0
assert 0.3 < t <= 0.85, f"threshold should sit between the classes, got {t}"
recs = [
    {"label": 1, "style": "formal", "subtype": None, "score": 0.9},
    {"label": 0, "style": None, "subtype": "unrelated", "score": 0.2},
]
assert calib.sensitivity_by_style(recs, 0.5)["formal"] == 1.0, "in-corpus above threshold = proceed"
assert calib.specificity_by_subtype(recs, 0.5)["unrelated"] == 1.0, "out-of-corpus below threshold = abstain"
print("PASSED: calibrate_threshold inverted correctly (AUC 1.0, >= proceeds, < abstains).")

print("\nALL CHECKPOINT 6 QDRANT SWAP TESTS PASSED")

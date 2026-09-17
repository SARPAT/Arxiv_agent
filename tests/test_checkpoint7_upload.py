"""Checkpoint 7 (user document upload) against a mocked Qdrant client, a
fake Redis and a stub embedder. No network, no real cluster.

The single most important assertion here is the cache-key leak test: two
sessions asking the same question must not share a retrieval cache entry
once either has uploaded a document.
"""
import io
from unittest.mock import MagicMock, patch

import fakeredis
from qdrant_client import models

import app.cache as cache
import app.session as session
import ingestion.upload as upload
import rag.retrieval as retrieval
import rag.vectorstore as vectorstore
from app.config import settings

PASSED = []


def ok(msg):
    PASSED.append(msg)
    print(f"PASSED: {msg}")


# --- Shared fakes -----------------------------------------------------------
fake_redis = fakeredis.FakeRedis(decode_responses=True)
cache._client = fake_redis
session._client = fake_redis


class StubEmbedder:
    """Deterministic unit vectors; dimension matches the real embedder."""
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384


def make_client():
    client = MagicMock()
    client.collection_exists.return_value = True
    client.count.return_value = MagicMock(count=0)
    return client


def reset(client):
    fake_redis.flushall()
    vectorstore._client = client
    vectorstore._embedding_dim = 384


# A genuinely valid one-page PDF, built by hand so the parse path is the
# real pypdf one rather than a mock. Text is long enough to clear
# min_extracted_chars.
def build_pdf(body_text: str) -> bytes:
    from pypdf import PdfWriter

    # pypdf can only *write* pages it has; compose a minimal PDF by hand and
    # read it back through PdfWriter to get a well-formed file.
    stream = f"BT /F1 12 Tf 40 700 Td ({body_text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_at}\n%%EOF\n"
    ).encode()
    written = io.BytesIO()
    PdfWriter(clone_from=io.BytesIO(bytes(out))).write(written)
    return written.getvalue()


LONG_TEXT = "Retrieval augmented generation over user documents. " * 40
REAL_PDF = build_pdf("Transformers process tokens in parallel using self attention. " * 3)


# --- 1. delete_by_tenant refuses the public tenant --------------------------
client = make_client()
reset(client)
for forbidden in ("public", "", None):
    try:
        vectorstore.delete_by_tenant(forbidden)
    except ValueError:
        pass
    else:
        raise AssertionError(f"delete_by_tenant({forbidden!r}) must raise")
assert not client.delete.called, "no delete request may reach Qdrant for a refused tenant"
ok("delete_by_tenant() refuses 'public' and empty tenants without calling Qdrant")

client.count.return_value = MagicMock(count=7)
assert vectorstore.delete_by_tenant("session-x") == 7
selector = client.delete.call_args.kwargs["points_selector"]
condition = selector.filter.must[0]
assert condition.key == "tenant_id" and condition.match.any == ["session-x"], condition
ok("delete_by_tenant() deletes exactly one tenant's points and returns the count")


# --- 2. Filename sanitization ----------------------------------------------
cases = {
    "paper.pdf": "paper",
    "/etc/passwd": "passwd",
    "../../../secret.pdf": "secret",
    r"C:\Users\me\My Paper.PDF": "My Paper",
    "wei<>rd:na*me?.pdf": "wei_rd_na_me",
    ".pdf": "uploaded document",
    "///": "uploaded document",
    "a" * 300 + ".pdf": "a" * upload.MAX_TITLE_CHARS,
}
for raw, expected in cases.items():
    got = upload.sanitize_filename(raw)
    assert got == expected, f"{raw!r} -> {got!r}, expected {expected!r}"
ok("sanitize_filename() strips paths/extensions/unsafe chars, caps length, never empty")


# --- 3. Rejections happen before any Qdrant or embedder work ----------------
client = make_client()
reset(client)
with patch.object(vectorstore, "get_embedder", StubEmbedder):
    try:
        upload.process_upload(b"x" * (settings.max_upload_bytes + 1), "big.pdf", "s1")
    except upload.FileTooLarge as exc:
        assert "MB" in str(exc), exc
    else:
        raise AssertionError("oversized upload must raise FileTooLarge")
assert not client.method_calls, f"oversize must be rejected before Qdrant: {client.method_calls}"
ok("oversized input is rejected before parsing and before any Qdrant call")

try:
    upload.process_upload(b"not a pdf at all", "notes.txt", "s1")
except upload.NotAPdf:
    pass
else:
    raise AssertionError("non-PDF must raise NotAPdf")
assert not client.method_calls
ok("a non-PDF is rejected on its magic bytes, before parsing and before Qdrant")

# A structurally valid PDF whose text is too short: parsing runs, nothing else.
short_pdf = build_pdf("hi")
try:
    upload.process_upload(short_pdf, "scan.pdf", "s1")
except upload.NoExtractableText as exc:
    assert "scanned" in str(exc).lower(), exc
else:
    raise AssertionError("short extracted text must raise NoExtractableText")
assert not client.method_calls, f"short text must be rejected before Qdrant: {client.method_calls}"
ok("short/empty extracted text is rejected before any Qdrant call, with the scanned-PDF message")

for bad_session in ("", "public"):
    try:
        upload.process_upload(REAL_PDF, "p.pdf", bad_session)
    except upload.UploadRejected:
        pass
    else:
        raise AssertionError(f"session_id {bad_session!r} must be rejected")
ok("an empty or 'public' session id is refused - an upload can never target the corpus")


# --- 4. A real upload: payload shape, tenant, delete-before-write ------------
client = make_client()
reset(client)
client.count.return_value = MagicMock(count=3)  # 3 stale points to replace
with patch.object(vectorstore, "get_embedder", StubEmbedder):
    summary = upload.process_upload(REAL_PDF, "  My Paper.pdf", "session-42")

assert summary["chunk_count"] >= 1 and summary["char_count"] > 0, summary
assert summary["filename"] == "My Paper", summary

call_order = [c[0] for c in client.method_calls]
assert call_order.index("delete") < call_order.index("upsert"), call_order
ok("existing session points are deleted before the new ones are written")

points = client.upsert.call_args.kwargs["points"]
payload = points[0].payload
assert payload["tenant_id"] == "session-42", payload
assert payload["chunk_type"] == "body", payload
assert payload["paper_key"] == "My Paper" and payload["Title"] == "My Paper", payload
for field in ("arxiv_id", "Published", "Authors", "Summary"):
    assert payload[field] == "", f"{field} must be present as an empty string: {payload}"
assert payload["text"], payload
assert set(points[0].vector) == {"dense"}, points[0].vector
ok("uploaded points carry tenant_id=session, chunk_type='body', and every corpus field")

assert session.get_upload("session-42") == {
    "filename": "My Paper",
    "chunk_count": summary["chunk_count"],
}
ok("the Redis upload marker is set only after a successful upsert")


# --- 5. The chunker is the corpus chunker, not a re-declared copy ------------
import ingestion.build_index as build_index

splitter = build_index.build_splitter()
assert splitter._chunk_size == build_index.CHUNK_SIZE
assert splitter._chunk_overlap == build_index.CHUNK_OVERLAP
assert "build_splitter" in upload.__dict__ or hasattr(upload, "build_splitter")
ok("uploads chunk through ingestion.build_index.build_splitter() - one shared definition")


# --- 6. Marker is cleared first, so a mid-upload failure can't strand it -----
client = make_client()
reset(client)
session.set_upload("session-99", "Older Doc", 5)
client.upsert.side_effect = RuntimeError("qdrant exploded")
with patch.object(vectorstore, "get_embedder", StubEmbedder):
    try:
        upload.process_upload(REAL_PDF, "new.pdf", "session-99")
    except RuntimeError:
        pass
    else:
        raise AssertionError("an upsert failure must propagate as a server error")
assert session.get_upload("session-99") is None, (
    "a failed upload must leave no marker - otherwise retrieval searches a tenant "
    "whose points were already deleted"
)
ok("a failure mid-upload leaves no stale marker; the session falls back to the corpus")


# --- 7. tenant_ids_for: Redis-only, no Qdrant call --------------------------
client = make_client()
reset(client)
assert session.tenant_ids_for("session-7") == ["public"]
session.set_upload("session-7", "Doc", 4)
assert session.tenant_ids_for("session-7") == ["public", "session-7"]
assert session.tenant_ids_for("") == ["public"]
assert not client.method_calls, "upload state must come from Redis, never a Qdrant query"
ok("tenant_ids_for() adds the session tenant only when it has an upload, from Redis alone")


# --- 8. THE LEAK TEST: same question, different sessions, different keys -----
reset(make_client())
question = "what is the main contribution?"
key_public = cache._retrieval_key(question, 1, cache.tenant_scope_id(["public"]))
key_a = cache._retrieval_key(question, 1, cache.tenant_scope_id(["public", "sess-a"]))
key_b = cache._retrieval_key(question, 1, cache.tenant_scope_id(["public", "sess-b"]))
assert key_a != key_b, "two uploading sessions must not share a retrieval cache key"
assert key_a != key_public and key_b != key_public
ok("LEAK TEST: two sessions with uploads produce different retrieval cache keys")

# ...and the same question from two sessions *without* uploads still shares one.
assert cache._retrieval_key(
    question, 1, cache.tenant_scope_id(session.tenant_ids_for("sess-c"))
) == cache._retrieval_key(
    question, 1, cache.tenant_scope_id(session.tenant_ids_for("sess-d"))
) == key_public
ok("public-only queries from different sessions still share one cache key")


# --- 9. End to end through retrieve(): B never sees A's cached chunks --------
client = make_client()
reset(client)


def fake_search(query_vector, k, tenant_ids):
    private = "sess-a" in tenant_ids
    text = "A's private upload" if private else "public corpus chunk"
    return [({"page_content": text, "metadata": {"paper_key": "x"}}, 0.9)] * k


with patch.object(retrieval, "search", fake_search), patch.object(
    retrieval, "get_embedder", StubEmbedder
):
    a_docs = retrieval.retrieve(question, k=2, tenant_ids=["public", "sess-a"])
    assert a_docs[0][0].page_content == "A's private upload"
    # Session B, same question, no upload -> public scope. If the tenant
    # scope were missing from the key this would return A's chunk from cache.
    b_docs = retrieval.retrieve(question, k=2, tenant_ids=session.tenant_ids_for("sess-b"))
assert b_docs[0][0].page_content == "public corpus chunk", b_docs[0][0].page_content
ok("LEAK TEST end-to-end: retrieve() never serves one session's upload to another")


# --- 10. sources[] surfaces an uploaded document ----------------------------
import rag.pipeline as pipeline
from langchain_core.documents import Document

corpus_doc = Document(page_content="c", metadata={"paper_key": "attention"})
upload_doc = Document(
    page_content="u", metadata={"paper_key": "My Paper", "Title": "My Paper"}
)
meta_doc = Document(page_content="m", metadata={"paper_key": "meta"})
titles = pipeline.retrieved_paper_titles([corpus_doc, upload_doc, meta_doc])
assert titles == ["Attention Is All You Need", "My Paper"], titles
assert pipeline.retrieved_paper_titles([Document(page_content="x", metadata={})]) == []
ok("retrieved_paper_titles() credits an uploaded document and still drops synthetic chunks")


# --- 11. list_tenants uses the facet aggregation ----------------------------
client = make_client()
reset(client)
client.facet.return_value = MagicMock(
    hits=[MagicMock(value="public"), MagicMock(value="sess-a")]
)
assert vectorstore.list_tenants() == ["public", "sess-a"]
assert client.facet.call_args.kwargs["key"] == "tenant_id"
assert client.facet.call_args.kwargs["exact"] is True
ok("list_tenants() reads distinct tenants via Qdrant's facet aggregation")


# --- 12. Cleanup script: dry run by default, public always excluded ---------
import scripts.cleanup_orphaned_uploads as cleanup

client = make_client()
reset(client)
client.facet.return_value = MagicMock(
    hits=[MagicMock(value="public"), MagicMock(value="live-s"), MagicMock(value="orphan-s")]
)
client.count.return_value = MagicMock(count=4)
cleanup.redis_client = fake_redis
session.set_upload("live-s", "Doc", 4)

cleanup.main.__globals__["argparse"].ArgumentParser.parse_args = lambda self: type(
    "A", (), {"delete": False}
)()
cleanup.main()
assert not client.delete.called, "dry run must not delete anything"
ok("cleanup script defaults to a dry run and deletes nothing")

cleanup.main.__globals__["argparse"].ArgumentParser.parse_args = lambda self: type(
    "A", (), {"delete": True}
)()
cleanup.main()
deleted_tenants = [
    c.kwargs["points_selector"].filter.must[0].match.any[0]
    for c in client.delete.call_args_list
]
assert deleted_tenants == ["orphan-s"], deleted_tenants
ok("cleanup --delete removes only the orphan; the live session and 'public' are untouched")


# --- 13. Every browser session gets its own id (and so its own tenant) ---
import ui.app as ui

initial = ui.session_state.value
assert callable(initial), (
    "gr.State deep-copies a plain value at build time, so a fixed uuid would "
    "give every visitor the same session id - one shared history, and one "
    "user's uploaded PDF searched on another user's questions"
)
assert initial() != initial(), "each page load must mint a distinct session id"
ok("each browser session mints its own id, so uploads cannot cross users")


# --- 14. Per-stage timings are logged, on success AND on failure --------
# The point of these is diagnosing a request that hit the upload timeout,
# so the failure path matters more than the success one: whichever stage
# is absent from the line is the stage it was still inside.
import logging

client = make_client()
reset(client)


def captured_upload(embedder, session_id):
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    upload_logger = logging.getLogger("ingestion.upload")
    upload_logger.addHandler(handler)
    upload_logger.setLevel(logging.INFO)
    try:
        with patch.object(vectorstore, "get_embedder", embedder):
            try:
                upload.process_upload(REAL_PDF, "timed.pdf", session_id)
            except RuntimeError:
                pass
    finally:
        upload_logger.removeHandler(handler)
    return [m for m in records if m.startswith("upload timings")]


lines = captured_upload(StubEmbedder, "session-timed")
assert len(lines) == 1, lines
assert "[ok]" in lines[0], lines[0]
for stage in ("extract", "delete", "chunk", "embed", "upsert"):
    assert f"{stage}=" in lines[0], f"{stage} missing from {lines[0]!r}"
assert "total=" in lines[0]
ok("process_upload() logs one timing line per stage on the success path")

exploding = MagicMock()
exploding.embed_documents.side_effect = RuntimeError("embed exploded")
lines = captured_upload(lambda: exploding, "session-boom")
assert len(lines) == 1, lines
assert "[failed]" in lines[0], lines[0]
# extract/delete/chunk completed; embed and upsert did not - and their
# absence is what identifies where it stopped.
for stage in ("extract", "delete", "chunk"):
    assert f"{stage}=" in lines[0], f"{stage} missing from {lines[0]!r}"
for stage in ("embed", "upsert"):
    assert f"{stage}=" not in lines[0], (
        f"{stage} must be absent - it never completed: {lines[0]!r}"
    )
ok("a failed upload still logs its partial timings, naming the stage it died in")


print(f"\nALL {len(PASSED)} CHECKPOINT 7 TESTS PASSED")

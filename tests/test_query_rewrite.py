"""Verify follow-up questions are resolved before retrieval (rag/query_rewrite.py
and its wiring in rag/pipeline.py's retrieve_context).

The production failure this guards: a session uploaded cache_blend.pdf,
asked about it, then asked "who are the author's of this paper". History
reached the generator but not the retriever, so the raw follow-up was
embedded, retrieved the RAG paper's chunks, and the answer named the RAG
paper's authors.

The model call itself is faked - NVIDIA's API is unreachable here - so
these check everything around it: that retrieval receives the rewritten
query and the right tenants, that no call is made when there's nothing to
resolve, and that every way the call can fail falls back to the raw
message rather than failing the turn."""
from unittest.mock import patch

import rag.pipeline as pipeline
import rag.query_rewrite as qr


class _Reply:
    def __init__(self, content):
        self.content = content


class _FakeClient:
    """Records the messages it was sent and returns a canned reply (or
    raises it, if it's an exception)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return _Reply(self.reply)


def _no_client():
    raise AssertionError("plan_search made a model call it had no reason to make")


CACHE_BLEND_HISTORY = [
    {"role": "user", "content": "tell me about the cache blend paper ."},
    {"role": "assistant", "content": "CacheBlend is a system designed to improve KV cache reuse..."},
]
FOLLOW_UP = "who are the author's of this paper"

# --- 1. Nothing to resolve: no history, no upload -> no model call -----------
with patch.object(qr, "_get_client", _no_client):
    assert qr.plan_search("What is attention?") == qr.SearchPlan("What is attention?")
    assert qr.plan_search("What is attention?", history=[]) == qr.SearchPlan("What is attention?")
print("PASSED: a first message with no upload is searched verbatim, with no model call.")

# --- 2. A follow-up is rewritten, and the prompt carries the history ---------
client = _FakeClient('{"query": "Who are the authors of the CacheBlend paper?", "upload_only": true}')
with patch.object(qr, "_get_client", lambda: client):
    plan = qr.plan_search(FOLLOW_UP, CACHE_BLEND_HISTORY, upload_filename="cache_blend")
assert plan == qr.SearchPlan("Who are the authors of the CacheBlend paper?", upload_only=True), plan
sent = client.calls[0]
assert "cache blend paper" in sent[1]["content"] and FOLLOW_UP in sent[1]["content"], sent
assert '"cache_blend"' in sent[0]["content"], "the upload's name must reach the rewrite prompt"
print("PASSED: a follow-up is rewritten from the conversation, which the prompt actually carries.")

# --- 3. Output tolerance and every fallback path ------------------------------
for label, reply, expected in [
    ("code-fenced JSON", '```json\n{"query": "CacheBlend authors", "upload_only": false}\n```',
     qr.SearchPlan("CacheBlend authors")),
    ("prose around JSON", 'Here you go: {"query": "CacheBlend authors", "upload_only": false}',
     qr.SearchPlan("CacheBlend authors")),
    ("non-boolean upload_only", '{"query": "CacheBlend authors", "upload_only": "yes"}',
     qr.SearchPlan("CacheBlend authors")),
    ("not JSON", "Who are the authors of CacheBlend?", qr.SearchPlan(FOLLOW_UP)),
    ("broken JSON", '{"query": "CacheBlend authors",', qr.SearchPlan(FOLLOW_UP)),
    ("empty query", '{"query": "  ", "upload_only": false}', qr.SearchPlan(FOLLOW_UP)),
    ("missing query", '{"upload_only": true}', qr.SearchPlan(FOLLOW_UP)),
    ("runaway query", '{"query": "' + "x" * 600 + '"}', qr.SearchPlan(FOLLOW_UP)),
    ("API error", RuntimeError("[503] Service Unavailable"), qr.SearchPlan(FOLLOW_UP)),
    ("timeout", TimeoutError("read timed out"), qr.SearchPlan(FOLLOW_UP)),
]:
    with patch.object(qr, "_get_client", lambda: _FakeClient(reply)):
        got = qr.plan_search(FOLLOW_UP, CACHE_BLEND_HISTORY, upload_filename="cache_blend")
    assert got == expected, (label, got)

# A client that can't even be constructed (bad key, unreachable API) is
# the same best-effort failure, not an exception out of the turn.
def _broken_client():
    raise ConnectionError("integrate.api.nvidia.com unreachable")

with patch.object(qr, "_get_client", _broken_client):
    assert qr.plan_search(FOLLOW_UP, CACHE_BLEND_HISTORY) == qr.SearchPlan(FOLLOW_UP)
print("PASSED: fenced/wrapped JSON is accepted; bad output, API errors, timeouts and an "
      "unbuildable client all fall back to the raw message.")

# --- 4. upload_only can't narrow scope without an upload to narrow to --------
with patch.object(qr, "_get_client", lambda: _FakeClient('{"query": "BERT authors", "upload_only": true}')):
    assert qr.plan_search("who wrote it?", CACHE_BLEND_HISTORY) == qr.SearchPlan("BERT authors")
print("PASSED: upload_only is ignored when the session has no upload.")

# --- 5. The prompt stays bounded however long the session gets ---------------
long_history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i} " + "y" * 5000}
    for i in range(20)
]
client = _FakeClient('{"query": "q", "upload_only": false}')
with patch.object(qr, "_get_client", lambda: client):
    qr.plan_search("and then?", long_history)
transcript = client.calls[0][1]["content"]
assert "turn-13 " not in transcript and "turn-14 " in transcript and "turn-19 " in transcript, (
    "only the last 6 history messages belong in the prompt"
)
assert len(transcript) < 6 * 700, len(transcript)
print("PASSED: only the last 6 messages reach the prompt, each capped in length.")

# --- 6. retrieve_context: the rewritten query and scope reach retrieval ------
# This is the production bug end to end: history must reach the retriever.
calls = []


def _fake_retrieve(query, k, tenant_ids):
    calls.append({"query": query, "tenant_ids": tenant_ids})
    return [(pipeline.Document(page_content="x", metadata={}), 0.9)]


with patch.object(pipeline, "retrieve", _fake_retrieve):
    for label, upload, reply, expected_query, expected_tenants in [
        ("upload-only follow-up", {"filename": "cache_blend", "chunk_count": 89},
         '{"query": "Who are the authors of the CacheBlend paper?", "upload_only": true}',
         "Who are the authors of the CacheBlend paper?", ["sess-1"]),
        ("comparison with the corpus", {"filename": "cache_blend", "chunk_count": 89},
         '{"query": "CacheBlend vs RAG", "upload_only": false}',
         "CacheBlend vs RAG", ["public", "sess-1"]),
        ("no upload", None,
         '{"query": "Who are the authors of BERT?", "upload_only": true}',
         "Who are the authors of BERT?", ["public"]),
    ]:
        calls.clear()
        with patch.object(pipeline, "get_upload", lambda sid, u=upload: u), \
                patch.object(qr, "_get_client", lambda r=reply: _FakeClient(r)):
            pipeline.retrieve_context(FOLLOW_UP, "sess-1", CACHE_BLEND_HISTORY)
        assert calls == [{"query": expected_query, "tenant_ids": expected_tenants}], (label, calls)

    # The eval harness path: no session, no history -> verbatim, public, no call.
    calls.clear()
    with patch.object(pipeline, "get_upload", lambda sid: None), \
            patch.object(qr, "_get_client", _no_client):
        pipeline.retrieve_context("What is multi-head attention?")
    assert calls == [{"query": "What is multi-head attention?", "tenant_ids": ["public"]}], calls
print("PASSED: retrieve_context searches the rewritten query, narrowing to the upload only "
      "when asked to; the eval harness's no-session path is unchanged.")

# --- 7. Both pipeline entry points hand history to retrieval -----------------
seen = []


def _recording_retrieve_context(query, session_id="", history=None):
    seen.append(history)
    return [pipeline.Document(page_content="x", metadata={})], 0.9


with patch.object(pipeline, "retrieve_context", _recording_retrieve_context), \
        patch.object(pipeline, "generate", lambda q, c, history=None: "answer"), \
        patch.object(pipeline, "generate_stream", lambda q, c, history=None: iter(["answer"])):
    pipeline.run_pipeline(FOLLOW_UP, history=CACHE_BLEND_HISTORY, session_id="s")
    list(pipeline.run_pipeline_stream(FOLLOW_UP, history=CACHE_BLEND_HISTORY, session_id="s"))
assert seen == [CACHE_BLEND_HISTORY, CACHE_BLEND_HISTORY], seen
print("PASSED: run_pipeline and run_pipeline_stream both pass history to retrieval, not just "
      "to generation.")

print("\nALL QUERY REWRITE TESTS PASSED")

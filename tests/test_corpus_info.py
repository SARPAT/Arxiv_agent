"""Verify GET /corpus/info (backend) and corpus_info_fn() (frontend):
papers and max_upload_mb are derived dynamically from TARGET_PAPERS and
settings.max_upload_bytes, never hardcoded; the frontend's welcome message
and upload-widget label come from one fetch and stay consistent with each
other; and a cold/unreachable backend falls back to a generic welcome
rather than a stack trace or a blank chat."""
from unittest.mock import patch

import httpx

import app.api as api
import ingestion.build_index as build_index
import rag.pipeline as pipeline
import ui.app as ui
from app.config import settings

# --- 1. /corpus/info: derived from TARGET_PAPERS and settings, not hardcoded ---
info = api.corpus_info()
assert info["papers"] == [p["title"] for p in build_index.TARGET_PAPERS.values()], info
assert info["max_upload_mb"] == settings.max_upload_bytes / (1024 * 1024), info
print("PASSED: /corpus/info's papers and max_upload_mb match their live sources exactly.")

# max_upload_mb must track settings.max_upload_bytes, not a separate constant.
original_bytes = settings.max_upload_bytes
try:
    settings.max_upload_bytes = 5 * 1024 * 1024
    assert api.corpus_info()["max_upload_mb"] == 5.0, api.corpus_info()
finally:
    settings.max_upload_bytes = original_bytes
print("PASSED: max_upload_mb recomputes from settings.max_upload_bytes - no separate hardcoded number.")

# papers must track TARGET_PAPERS' actual size, not a hardcoded count (7).
original_papers = dict(build_index.TARGET_PAPERS)
try:
    build_index.TARGET_PAPERS.clear()
    build_index.TARGET_PAPERS["attention"] = original_papers["attention"]
    pipeline.REAL_PAPER_TITLES = [i["title"] for i in build_index.TARGET_PAPERS.values()]
    api.REAL_PAPER_TITLES = pipeline.REAL_PAPER_TITLES
    shrunk = api.corpus_info()
    assert shrunk["papers"] == [original_papers["attention"]["title"]], shrunk
finally:
    build_index.TARGET_PAPERS.clear()
    build_index.TARGET_PAPERS.update(original_papers)
    pipeline.REAL_PAPER_TITLES = [i["title"] for i in build_index.TARGET_PAPERS.values()]
    api.REAL_PAPER_TITLES = pipeline.REAL_PAPER_TITLES
print("PASSED: papers reflects TARGET_PAPERS' actual size - no hardcoded count of 7.")

# --- 2. corpus_info_fn(): the frontend side of the same endpoint ---


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


with patch.object(
    ui.httpx,
    "get",
    return_value=_FakeResponse(
        200,
        {
            "papers": ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"],
            "max_upload_mb": 10.0,
        },
    ),
):
    chat_init, limit_label = ui.corpus_info_fn()

welcome = chat_init[0]["content"]
assert chat_init[0]["role"] == "assistant", chat_init
assert "Attention Is All You Need" in welcome
assert "BERT: Pre-training of Deep Bidirectional Transformers" in welcome
assert "10 MB" in welcome and "10.0 MB" not in welcome, welcome
assert limit_label == "Max 10 MB", limit_label
print("PASSED: corpus_info_fn() lists every paper and states the real size once, cleanly formatted.")

for label, mock_get in [
    ("a 5xx response", lambda: patch.object(ui.httpx, "get", return_value=_FakeResponse(503, {}))),
    ("a network error", lambda: patch.object(ui.httpx, "get", side_effect=httpx.ConnectError("refused"))),
    (
        "a malformed body",
        lambda: patch.object(ui.httpx, "get", return_value=_FakeResponse(200, {"papers": ["x"]})),
    ),
]:
    with mock_get():
        chat_init, limit_label = ui.corpus_info_fn()
    assert chat_init[0]["role"] == "assistant"
    assert "Welcome" in chat_init[0]["content"], (label, chat_init)
    assert "Attention" not in chat_init[0]["content"], (label, chat_init)
    assert limit_label == "", (label, limit_label)
print(
    "PASSED: corpus_info_fn() falls back to a generic welcome (no paper list, no crash) on "
    "a 5xx, a network error, or a malformed body."
)

# The welcome text and the widget label must move together, off one value -
# not two strings that could independently go stale.
with patch.object(
    ui.httpx,
    "get",
    return_value=_FakeResponse(200, {"papers": ["Attention Is All You Need"], "max_upload_mb": 25}),
):
    chat_init, limit_label = ui.corpus_info_fn()
assert "25 MB" in chat_init[0]["content"] and limit_label == "Max 25 MB", (chat_init, limit_label)
print("PASSED: the chat welcome and the upload-widget label agree, sourced from one fetched value.")

print("\nALL /corpus/info TESTS PASSED")

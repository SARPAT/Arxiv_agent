"""Verify ui/app.py's chat_fn() handles the SSE `error` event: appends the
message, adopts session_id, and terminates normally (doesn't hang) - without
needing a real Gradio server or a real backend."""
from unittest.mock import patch

import ui.app as ui_mod


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


# --- Scenario 1: mid-stream failure - some tokens, then an error frame ---
sse_lines = [
    "event: token",
    'data: {"delta": "partial "}',
    "",
    "event: token",
    'data: {"delta": "answer"}',
    "",
    "event: error",
    'data: {"message": "The AI service is temporarily unavailable. Please try again.", "session_id": "sX"}',
    "",
]

class _CM:
    """httpx.stream() is normally used as a context manager; wrap the
    fake response so patching it drops in transparently."""
    def __init__(self, resp):
        self._resp = resp

    def __enter__(self):
        return self._resp

    def __exit__(self, *a):
        return False


with patch("httpx.stream", side_effect=lambda *a, **k: _CM(_FakeResponse(sse_lines))):
    results = list(ui_mod.chat_fn("What is Attention?", [], ""))

assert results, "chat_fn() must yield at least once"
final_history, final_session_id = results[-1]
assert final_session_id == "sX", final_session_id
assert final_history[-1]["role"] == "assistant"
assert "partial answer" in final_history[-1]["content"], final_history[-1]["content"]
assert (
    "The AI service is temporarily unavailable. Please try again."
    in final_history[-1]["content"]
), final_history[-1]["content"]
print("PASSED: chat_fn() appends the error message after partial content, adopts session_id, terminates without hanging.")

# --- Scenario 2: pre-first-token failure - error is the ONLY content ---
sse_lines_2 = [
    "event: error",
    'data: {"message": "The AI service is temporarily unavailable. Please try again.", "session_id": "sY"}',
    "",
]


with patch("httpx.stream", side_effect=lambda *a, **k: _CM(_FakeResponse(sse_lines_2))):
    results = list(ui_mod.chat_fn("What is Attention?", [], ""))

final_history, final_session_id = results[-1]
assert final_session_id == "sY", final_session_id
assert (
    final_history[-1]["content"]
    == "The AI service is temporarily unavailable. Please try again."
), final_history[-1]["content"]
print("PASSED: chat_fn() shows just the error message (no leading blank/newline) when it's the only content.")

print("\nALL ui/app.py ERROR-EVENT CHECKS PASSED")

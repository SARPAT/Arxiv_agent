"""NVIDIA-hosted answer generation for the RAG pipeline.

Wraps ``ChatNVIDIA`` with the system prompt that governs how the model is
allowed to use retrieved context, and constructs the message list sent to
the model for a single turn. Whether to call this module at all — based
on retrieval confidence — is decided upstream, in ``rag/pipeline.py`` and
``rag/gate.py``; this module always generates when asked.

Every call to the model goes through a timeout (``GENERATION_TIMEOUT_SECONDS``
per attempt) and a fixed-backoff retry (``MAX_RETRIES`` attempts beyond the
first). ``langchain_nvidia_ai_endpoints`` doesn't give retry logic typed
exceptions to work with: a connection-level failure (the request never got
a response at all) surfaces as a real ``requests.exceptions.ConnectionError``
or ``Timeout``, but anything the API itself rejects — a bad request, an
expired model, a rate limit — comes back as a bare ``Exception`` with the
HTTP status folded into the message text (e.g. ``"[410] ..."``), not a
distinguishable exception type. ``_is_retryable`` is what inspects that
text to tell a transient failure from a permanent one.
"""

import logging
import re
import time
from collections.abc import Iterator

import requests
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import settings

logger = logging.getLogger(__name__)

GENERATION_TIMEOUT_SECONDS = 15
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = [1, 2]

GENERATION_ERROR_MESSAGE = (
    "The AI service is temporarily unavailable. Please try again."
)

# Failure modes the API itself reports as permanent - retrying burns
# attempts (and the user's wait) on something that will never succeed.
# 401/403 (bad or missing key), 404 (unknown model), 410 (deprecated
# model, the failure mode that motivated this list - see module docstring).
_NON_RETRYABLE_STATUSES = {401, 403, 404, 410}

_STATUS_PREFIX_RE = re.compile(r"^\[(\d{3})\]")


class GenerationError(Exception):
    """Raised when a generation call fails after exhausting retries, fails
    non-retryably, or fails mid-stream (after some tokens were already
    yielded to the caller, at which point retrying is impossible - there's
    no way to un-send content already streamed to the client)."""


def _is_retryable(exc: Exception) -> bool:
    """Decide whether ``exc`` (raised by a ``ChatNVIDIA`` call) is worth
    retrying.

    Connection-level failures - the request never reached the server, or
    never got a response - are always retryable regardless of message
    content, since ``requests`` raises typed exceptions for those. Anything
    else is a bare ``Exception`` from ``langchain_nvidia_ai_endpoints``
    itself; its message starts with the HTTP status in brackets (e.g.
    ``"[429] ..."``) whenever a real response came back, which is checked
    first as the most reliable signal, falling back to substring matching
    only when that prefix isn't present.
    """
    if isinstance(
        exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ):
        return True

    message = str(exc)
    match = _STATUS_PREFIX_RE.match(message)
    if match:
        status = int(match.group(1))
        if status in _NON_RETRYABLE_STATUSES:
            return False
        return status == 429 or 500 <= status < 600

    lowered = message.lower()
    return "429" in message or "rate limit" in lowered


SYSTEM_PROMPT = """You are a research assistant answering questions about a \
fixed collection of arXiv papers. You will be given retrieved context \
pulled from those papers along with a question. Follow these rules:

1. Treat the retrieved context as your primary source of truth. Prefer it \
over anything else you know whenever it's relevant to the question.
2. If you use general knowledge that is not supported by the retrieved \
context, say so explicitly in the answer (e.g. "based on general \
knowledge, not the provided documents, ...").
3. Never attribute general knowledge to a document, and never invent or \
guess a citation. Only cite a document if its content is actually present \
in the retrieved context you were given.
4. End every response with a "Sources:" block. List the exact document \
titles (as given in the context) that support your answer. If no document \
in the context actually supports the answer, write a single disclaimer \
bullet under "Sources:" saying the answer relies on general knowledge, \
not on the provided documents, instead of listing a title.
"""

# Module-level cache for the ChatNVIDIA client — a stateless, reusable API
# client, not per-user session state. See the equivalent note in
# retrieval.py for why this differs from conversation history.
_client: ChatNVIDIA | None = None


def _get_client() -> ChatNVIDIA:
    """Construct (and cache) the ChatNVIDIA client."""
    global _client
    if _client is None:
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY not found. Set it in a .env file.")
        _client = ChatNVIDIA(
            model=settings.generation_model,
            api_key=settings.nvidia_api_key,
            max_tokens=settings.max_tokens,
            chat_template_kwargs={"enable_thinking": False},
            timeout=GENERATION_TIMEOUT_SECONDS,
        )
    return _client


def _build_messages(
    query: str, context: str, history: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Assemble the message list sent to the model for one turn.

    ``history`` is an optional list of prior ``{"role": ..., "content": ...}``
    messages for multi-turn conversations. It is passed in by the caller on
    every call rather than stored anywhere in this module, so neither this
    function nor the ones that use it has any memory of past turns of its
    own — the caller (``app/session.py``, via ``rag/pipeline.py``) owns
    that state.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }
    )
    return messages


def generate(
    query: str, context: str, history: list[dict[str, str]] | None = None
) -> str:
    """Generate a complete answer to ``query`` given already-assembled
    ``context``, in one blocking call.

    The whole call either succeeds or fails - there's no partial-content
    ambiguity like the streaming path has, so every failed attempt (up to
    ``MAX_RETRIES`` beyond the first) is retried as long as ``_is_retryable``
    says the failure looks transient.
    """
    client = _get_client()
    messages = _build_messages(query, context, history)

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.invoke(messages)
            return response.content
        except Exception as exc:
            if attempt < MAX_RETRIES and _is_retryable(exc):
                logger.warning(
                    "Generation attempt %d failed (%s), retrying in %ss",
                    attempt + 1,
                    exc,
                    RETRY_BACKOFF_SECONDS[attempt],
                )
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                continue
            raise GenerationError(GENERATION_ERROR_MESSAGE) from exc


def generate_stream(
    query: str, context: str, history: list[dict[str, str]] | None = None
) -> Iterator[str]:
    """Generate an answer to ``query``, yielding token deltas as they arrive.

    Each yielded string is one incremental piece of the response text, not
    the full text so far — callers that need the complete response must
    accumulate the yielded deltas themselves (see
    ``rag/pipeline.py``'s ``run_pipeline_stream``).

    Retries (up to ``MAX_RETRIES`` beyond the first attempt, when
    ``_is_retryable`` agrees the failure looks transient) only happen while
    no content has been yielded yet. Once at least one delta has reached
    the caller, it's already been forwarded on toward the client - there's
    no way to un-send it - so any failure past that point raises
    ``GenerationError`` immediately instead of retrying.
    """
    client = _get_client()
    messages = _build_messages(query, context, history)

    attempt = 0
    while True:
        any_yielded = False
        try:
            for chunk in client.stream(messages):
                if chunk.content:
                    any_yielded = True
                    yield chunk.content
            return
        except Exception as exc:
            if any_yielded:
                raise GenerationError(GENERATION_ERROR_MESSAGE) from exc
            if attempt < MAX_RETRIES and _is_retryable(exc):
                logger.warning(
                    "Generation attempt %d failed before any token was sent "
                    "(%s), retrying in %ss",
                    attempt + 1,
                    exc,
                    RETRY_BACKOFF_SECONDS[attempt],
                )
                time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                attempt += 1
                continue
            raise GenerationError(GENERATION_ERROR_MESSAGE) from exc


if __name__ == "__main__":
    # No live NVIDIA API call here - everything below drives the retry
    # logic directly against a fake client, so it runs the same with or
    # without network access.
    from unittest.mock import patch

    class _FakeChunk:
        def __init__(self, content):
            self.content = content

    class _FakeClient:
        """Stands in for ChatNVIDIA. ``invoke_effects``/``stream_effects``
        is a list of callables, one per attempt: each either returns a
        result or raises, so a test can script "fail twice, then
        succeed" without touching real network code."""

        def __init__(self, effects):
            self.effects = list(effects)
            self.calls = 0

        def invoke(self, messages):
            effect = self.effects[self.calls]
            self.calls += 1
            return effect()

        def stream(self, messages):
            effect = self.effects[self.calls]
            self.calls += 1
            return effect()

    def _fake_get_client(client):
        return lambda: client

    def _ok(text="answer"):
        return lambda: _FakeChunk(text)

    def _ok_stream(*chunks):
        return lambda: iter([_FakeChunk(c) for c in chunks])

    def _fail(exc):
        def _raise():
            raise exc

        return _raise

    def _fail_stream(exc, after=()):
        def _raise():
            def _gen():
                for c in after:
                    yield _FakeChunk(c)
                raise exc

            return _gen()

        return _raise

    # --- _is_retryable() ---
    assert _is_retryable(requests.exceptions.ConnectionError("boom")) is True
    assert _is_retryable(requests.exceptions.Timeout("boom")) is True
    assert _is_retryable(Exception("[410] Gone")) is False
    assert _is_retryable(Exception("[404] Not Found")) is False
    assert _is_retryable(Exception("[401] Unauthorized")) is False
    assert _is_retryable(Exception("[403] Forbidden")) is False
    assert _is_retryable(Exception("[429] Too Many Requests")) is True
    assert _is_retryable(Exception("[500] Internal Server Error")) is True
    assert _is_retryable(Exception("[503] Service Unavailable")) is True
    assert _is_retryable(Exception("rate limit exceeded, try later")) is True
    assert _is_retryable(Exception("something odd happened")) is False
    print("PASSED: _is_retryable() classifies status-coded and connection-level failures correctly.")

    with patch("time.sleep") as mock_sleep:
        # generate(): retryable failure, retryable failure, then success.
        fake = _FakeClient([
            _fail(Exception("[500] Internal Server Error")),
            _fail(requests.exceptions.ConnectionError("boom")),
            _ok("final answer"),
        ])
        with patch(f"{__name__}._get_client", _fake_get_client(fake)):
            result = generate("q", "ctx")
        assert result == "final answer", result
        assert mock_sleep.call_args_list == [((1,),), ((2,),)], mock_sleep.call_args_list
        print("PASSED: generate() retries retryable failures with 1s/2s backoff, then returns on success.")

    with patch("time.sleep") as mock_sleep:
        # generate(): non-retryable failure fails fast, no sleep at all.
        fake = _FakeClient([_fail(Exception("[410] Gone"))])
        with patch(f"{__name__}._get_client", _fake_get_client(fake)):
            try:
                generate("q", "ctx")
                raise AssertionError("expected GenerationError")
            except GenerationError as exc:
                assert str(exc) == GENERATION_ERROR_MESSAGE
        assert mock_sleep.call_count == 0
        assert fake.calls == 1, "non-retryable failure must not retry"
        print("PASSED: generate() fails fast on a non-retryable status, no retry attempted.")

    with patch("time.sleep") as mock_sleep:
        # generate(): all 3 attempts retryable and all fail -> GenerationError.
        fake = _FakeClient([
            _fail(Exception("[500] a")),
            _fail(Exception("[500] b")),
            _fail(Exception("[500] c")),
        ])
        with patch(f"{__name__}._get_client", _fake_get_client(fake)):
            try:
                generate("q", "ctx")
                raise AssertionError("expected GenerationError")
            except GenerationError:
                pass
        assert fake.calls == 3, "expected exactly 3 total attempts"
        assert mock_sleep.call_args_list == [((1,),), ((2,),)]
        print("PASSED: generate() exhausts retries (3 total attempts) then raises GenerationError.")

    with patch("time.sleep") as mock_sleep:
        # generate_stream(): pre-first-token retryable failure, then success.
        fake = _FakeClient([
            _fail_stream(Exception("[503] Service Unavailable")),
            _ok_stream("hello ", "world"),
        ])
        with patch(f"{__name__}._get_client", _fake_get_client(fake)):
            chunks = list(generate_stream("q", "ctx"))
        assert chunks == ["hello ", "world"], chunks
        assert mock_sleep.call_args_list == [((1,),)]
        print("PASSED: generate_stream() retries a pre-first-token failure, then streams normally.")

    with patch("time.sleep") as mock_sleep:
        # generate_stream(): failure AFTER a chunk was already yielded must
        # raise immediately, mid-iteration, with no retry - this is the
        # core "can't un-send a token" constraint from the checkpoint spec.
        fake = _FakeClient([
            _fail_stream(Exception("[500] mid-stream"), after=("first chunk",)),
        ])
        with patch(f"{__name__}._get_client", _fake_get_client(fake)):
            gen = generate_stream("q", "ctx")
            first = next(gen)
            assert first == "first chunk"
            try:
                next(gen)
                raise AssertionError("expected GenerationError")
            except GenerationError as exc:
                assert str(exc) == GENERATION_ERROR_MESSAGE
        assert mock_sleep.call_count == 0, "must not retry after any token was yielded"
        assert fake.calls == 1
        print("PASSED: generate_stream() raises immediately (no retry) on a mid-stream failure.")

    print("\nALL rag/generation.py SELF-TESTS PASSED")

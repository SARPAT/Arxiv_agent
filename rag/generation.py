"""NVIDIA-hosted answer generation for the RAG pipeline.

Wraps ``ChatNVIDIA`` with the system prompt that governs how the model is
allowed to use retrieved context, and constructs the message list sent to
the model for a single turn. Whether to call this module at all — based
on retrieval confidence — is decided upstream, in ``rag/pipeline.py`` and
``rag/gate.py``; this module always generates when asked.
"""

from collections.abc import Iterator

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import settings

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
    ``context``, in one blocking call."""
    client = _get_client()
    messages = _build_messages(query, context, history)
    response = client.invoke(messages)
    return response.content


def generate_stream(
    query: str, context: str, history: list[dict[str, str]] | None = None
) -> Iterator[str]:
    """Generate an answer to ``query``, yielding token deltas as they arrive.

    Each yielded string is one incremental piece of the response text, not
    the full text so far — callers that need the complete response must
    accumulate the yielded deltas themselves (see
    ``rag/pipeline.py``'s ``run_pipeline_stream``).
    """
    client = _get_client()
    messages = _build_messages(query, context, history)
    for chunk in client.stream(messages):
        if chunk.content:
            yield chunk.content

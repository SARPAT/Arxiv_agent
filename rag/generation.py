"""NVIDIA-hosted answer generation for the RAG pipeline.

Wraps ``ChatNVIDIA`` with the system prompt that governs how the model is
allowed to use retrieved context, and constructs the message list sent to
the model for a single turn.
"""

import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

GENERATION_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
MAX_TOKENS = 512

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
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not found. Set it in a .env file.")
        _client = ChatNVIDIA(
            model=GENERATION_MODEL,
            api_key=api_key,
            max_tokens=MAX_TOKENS,
            chat_template_kwargs={"enable_thinking": False},
        )
    return _client


def generate(
    query: str, context: str, history: list[dict[str, str]] | None = None
) -> str:
    """Generate an answer to ``query`` given already-assembled ``context``.

    ``history`` is an optional list of prior ``{"role": ..., "content": ...}``
    messages for multi-turn conversations. It is passed in by the caller on
    every call rather than stored anywhere in this module, so this function
    has no memory of past turns of its own — the caller (ultimately, a
    per-session request handler once one exists) owns that state.

    Known limitation: no retrieval confidence check before generation —
    low-relevance context is still sent to the model. Addressed in a
    follow-up.

    This function does not check whether ``context`` is actually relevant
    to ``query`` before generating. Whatever context the caller assembled —
    even the near-random nearest neighbors retrieval returns for a
    genuinely out-of-corpus question — gets sent straight to the model with
    no gate in front of it. The system prompt's rules are the only thing
    standing between an irrelevant context and a false attribution.
    """
    client = _get_client()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append(
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        }
    )
    response = client.invoke(messages)
    return response.content

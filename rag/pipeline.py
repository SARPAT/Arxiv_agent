"""Orchestrates the retrieve -> gate -> assemble context -> generate RAG pipeline.

This is the single entry point the rest of the app (and the eval harness)
should call — it wires ``retrieval.py``, ``gate.py``, and ``generation.py``
together and owns no state of its own beyond what's passed in as arguments.
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from app.config import settings
from ingestion.build_index import TARGET_PAPERS
from rag.gate import should_abstain
from rag.generation import GenerationError, generate, generate_stream
from rag.retrieval import retrieve

logger = logging.getLogger(__name__)

ABSTAIN_RESPONSE = "I don't have information on that in my corpus.\n\nSources: none"

REAL_PAPER_TITLES = [info["title"] for info in TARGET_PAPERS.values()]

# LongContextReorder is a stateless transformer (it holds no data between
# calls), so one shared instance is safe here for the same reason the
# vectorstore and ChatNVIDIA client are cached in retrieval.py/generation.py.
_long_reorder = LongContextReorder()


def _format_chunk(doc: Document) -> str:
    """Render one chunk as ``[Source: <title>]\\n<content>``.

    This is the unit both budget selection and the final joined context
    operate on, so the two always agree on exactly how many characters a
    given chunk costs.
    """
    title = doc.metadata.get("Title", doc.metadata.get("paper_key", "Unknown"))
    return f"[Source: {title}]\n{doc.page_content}"


def _select_within_budget(docs: list[Document], max_chars: int) -> list[Document]:
    """Return the prefix of ``docs``, in the order given, that fits within
    ``max_chars``.

    This runs on retrieval's original best-first order, before any
    reordering for presentation. Selection happening first means the
    top-ranked document is only ever dropped if it alone exceeds the whole
    budget — never because a lower-ranked document got presented ahead of
    it by a later reordering step.
    """
    selected = []
    total_chars = 0
    for doc in docs:
        chunk_text = _format_chunk(doc)
        if total_chars + len(chunk_text) > max_chars:
            break
        selected.append(doc)
        total_chars += len(chunk_text)
    return selected


def docs2str(docs: list[Document], max_chars: int) -> str:
    """Join document chunks into a single context string, truncated to
    ``max_chars``.

    Chunks are appended in the order given until the next one would push
    the total past ``max_chars``. Called on an already budget-selected
    list (see ``assemble_context()``), this truncation is a safety net
    rather than the actual selection step.
    """
    parts = []
    total_chars = 0
    for doc in docs:
        chunk_text = _format_chunk(doc)
        if total_chars + len(chunk_text) > max_chars:
            break
        parts.append(chunk_text)
        total_chars += len(chunk_text)
    return "\n\n".join(parts)


def assemble_context(
    docs: list[Document], max_chars: int = settings.max_context_chars
) -> str:
    """Turn retrieved documents into the context string sent to the model.

    Selects which documents fit the character budget first, in retrieval's
    original best-first order, then reorders only the surviving documents
    with ``LongContextReorder`` for presentation — moving the most
    relevant ones to the start and end of the list, which helps LLMs
    attend to them across a long context. Selecting before reordering
    means truncation can only ever drop a document that didn't make the
    relevance cut in the first place, regardless of where the reordering
    step later moves the documents that did.
    """
    within_budget = _select_within_budget(docs, max_chars)
    reordered = _long_reorder.transform_documents(within_budget)
    return docs2str(reordered, max_chars=max_chars)


def extract_sources_block(response_text: str) -> str:
    """Return the text of the response's "Sources:" block, or "" if absent.

    Matches "Sources:" wherever it occurs in the text, with any amount or
    kind of whitespace before or after it — there's no line-start anchor,
    so a preceding newline is never required, which matters because
    accumulated streamed text isn't guaranteed to have one. Markdown
    emphasis directly wrapping the word ("**Sources:**", "*Sources:*") is
    stripped first so it doesn't need special-casing separately.
    """
    normalized = re.sub(
        r"[*_]{1,2}(Sources:)[*_]{1,2}", r"\1", response_text, flags=re.IGNORECASE
    )
    match = re.search(r"Sources:\s*(.*)", normalized, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_cited_sources(response_text: str) -> list[str]:
    """Return the real paper titles ``response_text``'s "Sources:" block
    actually cites.

    Matches by case-insensitive substring against the corpus's known
    titles — the same check ``eval/run_eval.py`` uses to detect false
    attribution, reused here to surface which titles were cited to API
    callers (see ``app/api.py``'s ``done`` event).
    """
    sources_text = extract_sources_block(response_text).lower()
    return [title for title in REAL_PAPER_TITLES if title.lower() in sources_text]


@dataclass
class PipelineResult:
    """Everything one call to ``run_pipeline`` produced, for callers (like
    the eval harness) that need more than just the final answer text."""

    answer: str
    abstained: bool
    top1_score: float
    context: str
    docs: list[Document]


def retrieve_and_gate(query: str) -> tuple[list[Document], float, bool]:
    """Retrieve the top ``settings.retrieval_k`` chunks and apply the
    confidence gate.

    Returns ``(docs, top1_score, abstained)``. ``top1_score`` is the
    closest chunk's raw distance — it's the same value regardless of how
    many total documents are requested, since it's always whichever single
    result is nearest.
    """
    retrieved = retrieve(query, k=settings.retrieval_k)
    docs = [doc for doc, _score in retrieved]
    top1_score = retrieved[0][1]
    return docs, top1_score, should_abstain(top1_score)


def run_pipeline(
    query: str, history: list[dict[str, str]] | None = None
) -> PipelineResult:
    """Run one full turn of the RAG pipeline, including the confidence gate.

    If the gate abstains, context assembly and generation are both
    skipped — no model call is made, and ``ABSTAIN_RESPONSE`` is returned
    as-is.
    """
    docs, top1_score, abstained = retrieve_and_gate(query)
    if abstained:
        return PipelineResult(
            answer=ABSTAIN_RESPONSE,
            abstained=True,
            top1_score=top1_score,
            context="",
            docs=docs,
        )

    context = assemble_context(docs)
    answer = generate(query, context, history=history)
    return PipelineResult(
        answer=answer,
        abstained=False,
        top1_score=top1_score,
        context=context,
        docs=docs,
    )


def run_pipeline_stream(
    query: str, history: list[dict[str, str]] | None = None
) -> Iterator[dict]:
    """Run one full turn of the RAG pipeline, streaming the answer.

    Yields a sequence of small event dicts:

    - ``{"type": "token", "delta": <str>}`` for each piece of generated text
    - exactly one ``{"type": "done", "sources": [...], "abstained": <bool>,
      "top1_score": <float>}`` once the answer (or the abstain response)
      is complete, **or** exactly one ``{"type": "error", "message": <str>}``
      instead of "done" if generation fails (see ``rag/generation.py``'s
      retry/timeout handling) — either before any token was sent, or
      mid-stream after some already were. Callers must not treat a
      "token"-then-nothing-else sequence as a silent success: an "error"
      event always follows a failed generation, "done" always follows a
      successful one, and exactly one of the two terminates every call
      that reaches generation at all.

    The abstain path yields ``ABSTAIN_RESPONSE`` as a single "token" event
    followed by the same "done" shape a real answer produces, so callers
    don't need a separate case for it.
    """
    docs, top1_score, abstained = retrieve_and_gate(query)

    if abstained:
        yield {"type": "token", "delta": ABSTAIN_RESPONSE}
        yield {
            "type": "done",
            "sources": [],
            "abstained": True,
            "top1_score": top1_score,
        }
        return

    context = assemble_context(docs)
    accumulated = []
    try:
        for delta in generate_stream(query, context, history=history):
            accumulated.append(delta)
            yield {"type": "token", "delta": delta}
    except GenerationError as exc:
        yield {"type": "error", "message": str(exc)}
        return

    full_text = "".join(accumulated)
    sources = extract_cited_sources(full_text)
    if not sources:
        # Reached only on the non-abstain path, where a citation is expected.
        # `repr()` rather than the plain string, so any invisible or unusual
        # characters that a live streamed response contains are visible in
        # the log line instead of silently blending into normal whitespace.
        logger.warning(
            "extract_cited_sources() found no citations in a non-abstained "
            "response; raw text: %r",
            full_text,
        )
    yield {
        "type": "done",
        "sources": sources,
        "abstained": False,
        "top1_score": top1_score,
    }


def answer_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    """Run one turn of the RAG pipeline and return just the answer text.

    ``history`` is optional prior conversation turns, forwarded straight to
    ``generation.generate()``. It is accepted as a parameter here — never
    stored as module or global state — which is what lets ``app/api.py``
    own per-session history (via ``app/session.py``) and call this function
    safely for many concurrent users without their histories bleeding into
    each other.
    """
    return run_pipeline(query, history=history).answer


if __name__ == "__main__":
    # Regression check for extract_cited_sources(): a "Sources:" marker
    # with irregular spacing around it (two spaces, no preceding newline)
    # rather than the model's usual newline-separated block format.
    example_text = (
        "The scaling factor prevents the dot products from growing too "
        "large in magnitude, which would otherwise push softmax into "
        "regions with extremely small gradients.  Sources: Attention Is "
        "All You Need"
    )
    cited = extract_cited_sources(example_text)
    assert cited == ["Attention Is All You Need"], cited
    print("extract_cited_sources() regression test passed.")

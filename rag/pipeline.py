"""Orchestrates the retrieve -> assemble context -> generate RAG pipeline.

This is the single entry point the rest of the app (and the eval harness)
should call — it wires ``retrieval.py`` and ``generation.py`` together and
owns no state of its own beyond what's passed in as arguments.

There is no confidence gate: retrieval's top-k always proceeds to
generation, and the system prompt (see ``rag/generation.py``) is what
governs how the model uses that context — grounding answers in it when
relevant, ignoring it and answering from clearly-labeled general
knowledge when it isn't. The gate that used to sit here (a calibrated L2
distance threshold that abstained on low-confidence retrieval) was
removed: on dense-only retrieval the score bands for correct in-corpus
answers and out-of-corpus junk overlapped too much for any single
threshold to separate them without rejecting legitimate answers. The
calibration tooling that measured this (``eval/calibrate_threshold.py``,
``eval/calibration_*.json``) is kept for the eventual hybrid-search
recalibration, just no longer wired into the runtime.
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from app.config import settings
from ingestion.build_index import TARGET_PAPERS
from rag.generation import GenerationError, generate, generate_stream
from rag.retrieval import retrieve

logger = logging.getLogger(__name__)

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
    top1_score: float
    context: str
    docs: list[Document]


def retrieve_context(query: str) -> tuple[list[Document], float]:
    """Retrieve the top ``settings.retrieval_k`` chunks for ``query``.

    Returns ``(docs, top1_score)``. Every query proceeds to generation —
    there is no gate — so this no longer makes an abstain/proceed decision;
    it just retrieves. ``top1_score`` (the closest chunk's raw L2 distance)
    is still returned as telemetry, not as a gate input: it's reported in
    the ``done`` event and by the eval harness, but nothing branches on it.
    """
    retrieved = retrieve(query, k=settings.retrieval_k)
    docs = [doc for doc, _score in retrieved]
    top1_score = retrieved[0][1]
    return docs, top1_score


def run_pipeline(
    query: str, history: list[dict[str, str]] | None = None
) -> PipelineResult:
    """Run one full turn of the RAG pipeline.

    Retrieval's top-k always proceeds to generation; the system prompt (see
    ``rag/generation.py``) is what decides whether to ground the answer in
    the retrieved context or answer from clearly-labeled general knowledge.
    """
    docs, top1_score = retrieve_context(query)
    context = assemble_context(docs)
    answer = generate(query, context, history=history)
    return PipelineResult(
        answer=answer,
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
    - exactly one ``{"type": "done", "sources": [...], "top1_score": <float>}``
      once the answer is complete, **or** exactly one
      ``{"type": "error", "message": <str>}`` instead of "done" if
      generation fails (see ``rag/generation.py``'s retry/timeout handling)
      — either before any token was sent, or mid-stream after some already
      were. Callers must not treat a "token"-then-nothing-else sequence as a
      silent success: an "error" event always follows a failed generation,
      "done" always follows a successful one, and exactly one of the two
      terminates every call.

    Every query generates — there is no abstain path — so ``sources`` in the
    ``done`` event reflects what the finished answer actually cited (which
    may be the general-knowledge disclaimer line, carrying no paper title).
    """
    docs, top1_score = retrieve_context(query)

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
        # No corpus paper cited. This is expected and correct for a general-
        # knowledge answer (its "Sources:" block carries the disclaimer line,
        # not a paper title), so it's no longer treated as anomalous - but a
        # corpus-grounded answer that failed to cite would also land here, so
        # it's logged at INFO for visibility. `repr()` rather than the plain
        # string, so any invisible or unusual characters in a live streamed
        # response are visible instead of blending into normal whitespace.
        logger.info(
            "extract_cited_sources() found no corpus paper cited (expected for "
            "a general-knowledge answer); raw text: %r",
            full_text,
        )
    yield {
        "type": "done",
        "sources": sources,
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

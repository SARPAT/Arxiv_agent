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
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from app.config import settings
from ingestion.build_index import TARGET_PAPERS
from rag.generation import (
    GENERAL_KNOWLEDGE_MARKER,
    GenerationError,
    generate,
    generate_stream,
)
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


def retrieved_paper_titles(docs: list[Document]) -> list[str]:
    """Canonical corpus paper titles for the retrieved chunks, deduped in
    retrieval order.

    This is the structured source list for a grounded answer - drawn from
    the actual retrieved chunk metadata, not parsed from the model's
    output (the model no longer writes its own "Sources:" line; see the
    system prompt in ``rag/generation.py``). Each chunk's ``paper_key`` is
    mapped to its title via ``TARGET_PAPERS``, which naturally excludes
    synthetic non-paper chunks such as the doc-list chunk (``paper_key``
    "meta", not a real paper).
    """
    titles: list[str] = []
    for doc in docs:
        info = TARGET_PAPERS.get(doc.metadata.get("paper_key"))
        if info and info["title"] not in titles:
            titles.append(info["title"])
    return titles


def answered_from_general_knowledge(answer_text: str) -> bool:
    """True if ``answer_text`` carries the general-knowledge provenance
    marker the system prompt requires when (and only when) the answer is
    not grounded in the corpus.

    This is the signal used to empty the structured sources for a
    general-knowledge answer: retrieval always returns chunks (there is no
    gate), so the retrieved metadata alone can't distinguish "answered
    from these papers" from "answered from general knowledge despite these
    papers being retrieved" - the model's own marker is what does.
    """
    needle = GENERAL_KNOWLEDGE_MARKER.rstrip(".").lower()
    return needle in answer_text.lower()


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
    # A general-knowledge answer (model emitted the provenance marker) has
    # no corpus sources - the retrieved chunks were irrelevant and ignored,
    # so the structured list is empty and the frontend renders no "Sources:"
    # block. A grounded answer's sources come from the retrieved chunk
    # metadata, which is the single rendering (the model no longer writes
    # its own "Sources:" line).
    if answered_from_general_knowledge(full_text):
        sources: list[str] = []
    else:
        sources = retrieved_paper_titles(docs)
        if not sources:
            # Grounded (no general-knowledge marker) yet no corpus paper
            # among the retrieved chunks - unexpected; log for visibility.
            # `repr()` so any invisible/unusual characters in the streamed
            # response are visible instead of blending into whitespace.
            logger.info(
                "No general-knowledge marker and no corpus paper in the "
                "retrieved chunks; raw text: %r",
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
    from langchain_core.documents import Document as _Doc

    # retrieved_paper_titles(): maps retrieved chunks' paper_key to canonical
    # titles, deduped in order, excluding synthetic non-paper chunks.
    docs = [
        _Doc(page_content="a", metadata={"paper_key": "attention"}),
        _Doc(page_content="b", metadata={"paper_key": "attention"}),  # dup -> once
        _Doc(page_content="c", metadata={"paper_key": "bert"}),
        _Doc(page_content="d", metadata={"paper_key": "meta"}),  # doc-list, not a paper
    ]
    titles = retrieved_paper_titles(docs)
    assert titles == [
        TARGET_PAPERS["attention"]["title"],
        TARGET_PAPERS["bert"]["title"],
    ], titles
    assert TARGET_PAPERS["attention"]["title"] in REAL_PAPER_TITLES
    print("retrieved_paper_titles() dedupes in order and excludes non-paper chunks.")

    # answered_from_general_knowledge(): detects the exact provenance marker
    # the system prompt mandates, case-insensitively and tolerant of the
    # trailing period, but does not fire on an unrelated grounded answer.
    assert answered_from_general_knowledge(
        "Ronaldo is a footballer. " + GENERAL_KNOWLEDGE_MARKER
    )
    assert answered_from_general_knowledge(
        "prefix " + GENERAL_KNOWLEDGE_MARKER.upper().rstrip(".") + " suffix"
    )
    assert not answered_from_general_knowledge(
        "The Transformer is introduced in Attention Is All You Need."
    )
    print("answered_from_general_knowledge() matches the marker, ignores grounded text.")

    print("\nALL rag/pipeline.py SELF-TESTS PASSED")

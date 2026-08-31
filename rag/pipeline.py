"""Orchestrates the retrieve -> gate -> assemble context -> generate RAG pipeline.

This is the single entry point the rest of the app (and the eval harness)
should call — it wires ``retrieval.py``, ``gate.py``, and ``generation.py``
together and owns no state of its own beyond what's passed in as arguments.
"""

from dataclasses import dataclass

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from rag.gate import should_abstain
from rag.generation import generate
from rag.retrieval import retrieve

RETRIEVAL_K = 4
MAX_CONTEXT_CHARS = 2500

ABSTAIN_RESPONSE = "I don't have information on that in my corpus.\n\nSources: none"

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


def assemble_context(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
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
    """Retrieve the top ``RETRIEVAL_K`` chunks and apply the confidence gate.

    Returns ``(docs, top1_score, abstained)``. ``top1_score`` is the
    closest chunk's raw distance — it's the same value regardless of how
    many total documents are requested, since it's always whichever single
    result is nearest.
    """
    retrieved = retrieve(query, k=RETRIEVAL_K)
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


def answer_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    """Run one turn of the RAG pipeline and return just the answer text.

    ``history`` is optional prior conversation turns, forwarded straight to
    ``generation.generate()``. It is accepted as a parameter here — never
    stored as module or global state — so that whichever caller eventually
    wires up real user sessions (there is no web/session layer yet; that's
    Checkpoint 4) owns the actual session scoping, and this function stays
    safe to call concurrently for different users without their histories
    bleeding into each other.
    """
    return run_pipeline(query, history=history).answer

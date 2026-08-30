"""Orchestrates the retrieve -> assemble context -> generate RAG pipeline.

This is the single entry point the rest of the app (and the eval harness)
should call — it wires ``retrieval.py`` and ``generation.py`` together and
owns no state of its own beyond what's passed in as arguments.
"""

from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

from rag.generation import generate
from rag.retrieval import retrieve

RETRIEVAL_K = 4
MAX_CONTEXT_CHARS = 2500

# LongContextReorder is a stateless transformer (it holds no data between
# calls), so one shared instance is safe here for the same reason the
# vectorstore and ChatNVIDIA client are cached in retrieval.py/generation.py.
_long_reorder = LongContextReorder()


def docs2str(docs: list[Document], max_chars: int) -> str:
    """Join document chunks into a single context string, truncated to
    ``max_chars``.

    Each chunk is prefixed with its source paper's title so the model (and
    the eval harness's false-attribution check) can see which document a
    given piece of context came from. Chunks are appended **in the order
    given** until the next chunk would push the total past ``max_chars``,
    at which point the remaining chunks are dropped entirely — this
    function has no awareness of which chunks are most relevant; it just
    respects the order it's handed. See ``assemble_context()`` for why that
    ordering matters.
    """
    parts = []
    total_chars = 0
    for doc in docs:
        title = doc.metadata.get("Title", doc.metadata.get("paper_key", "Unknown"))
        chunk_text = f"[Source: {title}]\n{doc.page_content}"
        if total_chars + len(chunk_text) > max_chars:
            break
        parts.append(chunk_text)
        total_chars += len(chunk_text)
    return "\n\n".join(parts)


def assemble_context(docs: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Turn retrieved documents into the context string sent to the model.

    INTENTIONAL BUG (Checkpoint 2 baseline — fixed in Checkpoint 3):
    ``LongContextReorder`` is applied *before* truncation, not after.
    ``LongContextReorder`` assumes its input is sorted most-relevant-first
    and redistributes documents so the most relevant ones sit at the start
    *and end* of the list (mitigating LLMs' well-documented tendency to
    under-weight information buried in the middle of a long context). But
    that means the single best-ranked document can end up **last** in the
    reordered list — and ``docs2str`` truncates by walking the list in
    order and stopping once the character budget runs out. With
    ``RETRIEVAL_K = 4``, the top-ranked document is exactly the one
    ``LongContextReorder`` moves to the end, so a large enough set of
    lower-ranked chunks in front of it can push it past ``max_chars`` and
    cut it entirely — the reordering optimizes for the model's attention
    pattern while quietly sabotaging the truncation step that runs right
    after it. The fix is to select which documents fit in the budget
    *before* reordering for presentation, not after. This is measured in
    the Checkpoint 2 eval baseline and fixed in Checkpoint 3 — do not
    reorder the two lines below to "fix" this before that comparison is
    captured.
    """
    reordered = _long_reorder.transform_documents(docs)  # reorder first...
    return docs2str(reordered, max_chars=max_chars)  # ...then truncate (bug: can cut the best-ranked chunk)


def answer_query(query: str, history: list[dict[str, str]] | None = None) -> str:
    """Run one turn of the RAG pipeline: retrieve, assemble context, generate.

    ``history`` is optional prior conversation turns, forwarded straight to
    ``generation.generate()``. It is accepted as a parameter here — never
    stored as module or global state — so that whichever caller eventually
    wires up real user sessions (there is no web/session layer yet; that's
    Checkpoint 4) owns the actual session scoping, and this function stays
    safe to call concurrently for different users without their histories
    bleeding into each other.
    """
    retrieved = retrieve(query, k=RETRIEVAL_K)
    docs = [doc for doc, _score in retrieved]
    context = assemble_context(docs)
    return generate(query, context, history=history)

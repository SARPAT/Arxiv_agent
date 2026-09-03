"""Similarity search over the FAISS index built in Checkpoint 1.

Loads the persisted index from ``data/docstore_index/`` using the same
embedding model (``app.config.settings.embedding_model``) it was built
with — FAISS indexes are just vectors plus metadata, so querying with a
different embedder would silently produce meaningless nearest-neighbor
results.

Retrieval is cached in two layers (``app/cache.py``): a full-retrieval
cache keyed on the query and the current corpus version, and — on a
retrieval-cache miss — a query-embedding cache underneath it. Both are
optional from a correctness standpoint; see ``retrieve()``.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.cache import (
    get_cached_embedding,
    get_cached_retrieval,
    get_corpus_version,
    set_cached_embedding,
    set_cached_retrieval,
)
from app.config import settings

INDEX_PATH = "data/docstore_index"

# Module-level cache for the loaded index and embedder. This is *not*
# per-user session state (loading a ~300-chunk FAISS index and a
# sentence-transformers model from disk on every call would be needlessly
# slow) — it's a shared, stateless resource, no different from caching a
# database connection pool. Conversation history is a separate concern and
# is deliberately never stored here; see pipeline.py.
_vectorstore: FAISS | None = None


def _get_vectorstore() -> FAISS:
    """Load and cache the persisted FAISS index.

    ``allow_dangerous_deserialization=True`` is required because langchain's
    FAISS wrapper persists its docstore with pickle. This is safe here only
    because ``data/docstore_index/`` is a file we built and committed
    ourselves in Checkpoint 1, not an index accepted from an untrusted
    source.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        _vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


def _doc_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def _dict_to_doc(data: dict) -> Document:
    return Document(page_content=data["page_content"], metadata=data["metadata"])


def retrieve(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return the ``k`` chunks most similar to ``query``.

    Each result is a ``(document, score)`` pair, where ``score`` is the raw
    FAISS L2 distance between the query embedding and the chunk embedding —
    **lower means more similar**, not a normalized 0-1 similarity score.

    This function applies no relevance or confidence filtering: it always
    returns exactly ``k`` documents, however distant they are from the
    query. For an out-of-corpus question there may be no chunk that is
    genuinely relevant, but this function has no way to signal that and
    will still return its ``k`` nearest neighbors.

    Checks the retrieval cache first, keyed on the current corpus version
    so a future re-ingestion invalidates it automatically. A cached entry
    is only used if it holds at least ``k`` results — different callers in
    this codebase ask for different ``k`` (the live pipeline always uses
    ``settings.retrieval_k``, the eval scripts use larger values for
    Recall@8) and the cache key doesn't encode ``k``, so an entry with
    fewer results than requested is treated as a miss and overwritten with
    a freshly computed, larger one rather than silently under-returning.
    On a retrieval-cache miss, checks the embedding cache before falling
    back to embedding the query live; either way the result is written
    back to the retrieval cache. Neither cache is consulted for anything
    beyond the query and corpus version — a cache or Redis outage falls
    back to this function's exact previous (uncached) behavior.
    """
    corpus_version = get_corpus_version()

    cached = get_cached_retrieval(query, corpus_version)
    if cached is not None and len(cached["chunks"]) >= k:
        docs = [_dict_to_doc(chunk) for chunk in cached["chunks"][:k]]
        scores = cached["scores"][:k]
        return list(zip(docs, scores))

    vectorstore = _get_vectorstore()

    vector = get_cached_embedding(query)
    if vector is None:
        vector = vectorstore.embeddings.embed_query(query)
        set_cached_embedding(query, vector)

    results = vectorstore.similarity_search_with_score_by_vector(vector, k=k)

    set_cached_retrieval(
        query,
        corpus_version,
        chunks=[_doc_to_dict(doc) for doc, _score in results],
        scores=[score for _doc, score in results],
    )

    return results

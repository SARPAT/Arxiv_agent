"""Similarity search over the Qdrant Cloud collection.

Checkpoint 6 replaced the local FAISS index with Qdrant (see
``rag/vectorstore.py``). There is no index to load: the process holds no
corpus in memory, and there is no startup load step - the first query
just issues a network call. Queries are embedded with the same embedder
the corpus was ingested with (``rag.embedder.get_embedder()``); querying
with a different embedder, or the same model through a different runtime,
would silently produce meaningless nearest neighbours.

**Score direction changed with the backend.** FAISS returned a raw L2
distance where *lower* was a better match; Qdrant returns cosine
similarity where *higher* is better. Because the embedder L2-normalizes
its output the two rank identically, so nothing about ordering changed -
but any comparison against a score value had to be inverted, and Qdrant
already returns results best-first, so this module must not re-sort them.

Retrieval is cached in two layers (``app/cache.py``): a full-retrieval
cache keyed on the query, the corpus version and the retrieval backend,
and — on a retrieval-cache miss — a query-embedding cache underneath it.
Both are optional from a correctness standpoint; see ``retrieve()``.
"""

from langchain_core.documents import Document

from app.cache import (
    get_cached_embedding,
    get_cached_retrieval,
    get_corpus_version,
    set_cached_embedding,
    set_cached_retrieval,
)
from rag.embedder import get_embedder
from rag.vectorstore import search

# The tenant whose chunks make up the shared arXiv corpus. Checkpoint 7
# will extend this to ``["public", session_id]`` so a user's own uploads
# are searched alongside it; for now every caller searches only the
# shared corpus.
PUBLIC_TENANT_IDS = ["public"]


def _dict_to_doc(data: dict) -> Document:
    return Document(page_content=data["page_content"], metadata=data["metadata"])


def retrieve(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Return the ``k`` chunks most similar to ``query``, best first.

    Each result is a ``(document, score)`` pair, where ``score`` is the
    cosine similarity between the query embedding and the chunk embedding
    — **higher means more similar**, roughly on a 0-1 scale. (Before
    Checkpoint 6 this was a raw FAISS L2 distance, where lower was
    better.) Results arrive from Qdrant already ordered best-first and are
    returned in that order untouched.

    This function applies no relevance or confidence filtering: it always
    returns up to ``k`` documents, however weak the match. For an
    out-of-corpus question there may be no chunk that is genuinely
    relevant, but this function has no way to signal that and will still
    return its ``k`` nearest neighbours.

    Checks the retrieval cache first, keyed on the current corpus version
    so a re-ingestion invalidates it automatically. A cached entry is only
    used if it holds at least ``k`` results — different callers in this
    codebase ask for different ``k`` (the live pipeline always uses
    ``settings.retrieval_k``, the eval scripts use larger values for
    Recall@8) and the cache key doesn't encode ``k``, so an entry with
    fewer results than requested is treated as a miss and overwritten with
    a freshly computed, larger one rather than silently under-returning.
    On a retrieval-cache miss, checks the embedding cache before falling
    back to embedding the query live; either way the result is written
    back to the retrieval cache. A cache or Redis outage falls back to
    this function's exact uncached behaviour.
    """
    corpus_version = get_corpus_version()

    cached = get_cached_retrieval(query, corpus_version)
    if cached is not None and len(cached["chunks"]) >= k:
        docs = [_dict_to_doc(chunk) for chunk in cached["chunks"][:k]]
        scores = cached["scores"][:k]
        return list(zip(docs, scores))

    vector = get_cached_embedding(query)
    if vector is None:
        vector = get_embedder().embed_query(query)
        set_cached_embedding(query, vector)

    results = search(vector, k=k, tenant_ids=PUBLIC_TENANT_IDS)

    set_cached_retrieval(
        query,
        corpus_version,
        chunks=[chunk for chunk, _score in results],
        scores=[score for _chunk, score in results],
    )

    return [(_dict_to_doc(chunk), score) for chunk, score in results]

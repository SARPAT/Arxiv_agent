"""Similarity search over the FAISS index built in Checkpoint 1.

Loads the persisted index from ``data/docstore_index/`` using the same
``BAAI/bge-base-en-v1.5`` embedding model it was built with — FAISS indexes
are just vectors plus metadata, so querying with a different embedder
would silently produce meaningless nearest-neighbor results.
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
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
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


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
    """
    vectorstore = _get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)

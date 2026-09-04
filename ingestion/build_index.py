"""Build a persisted FAISS vector index from the 7 target arXiv papers.

Fetches each paper via ArxivLoader, strips its references section, chunks
the remaining text, adds synthetic doc-list and per-paper metadata chunks,
embeds everything with ``rag.embedder.get_embedder()``, and persists the
FAISS index to data/docstore_index/. The corpus is static, so this is
meant to be run once, not on every app boot.
"""

import re
from collections import Counter

from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embedder import get_embedder

# Shared with rag/retrieval.py (Checkpoint 4f) rather than each owning its
# own embedding logic - Checkpoint 4e found this script had drifted onto
# its own hardcoded model constant, which would have silently built an
# index at the wrong dimensionality for what the running app queries with.
INDEX_PATH = "data/docstore_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

TARGET_PAPERS = {
    "attention": {
        "arxiv_id": "1706.03762",
        "title": "Attention Is All You Need",
        "description": (
            "Introduces the Transformer architecture, based entirely on "
            "attention mechanisms and dispensing with recurrence and "
            "convolution for sequence transduction."
        ),
    },
    "bert": {
        "arxiv_id": "1810.04805",
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "description": (
            "Introduces BERT, a bidirectional Transformer pre-trained with "
            "masked language modeling and next sentence prediction for "
            "language understanding."
        ),
    },
    "rag_paper": {
        "arxiv_id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "description": (
            "Introduces Retrieval-Augmented Generation, combining a "
            "parametric seq2seq model with a non-parametric retriever over "
            "Wikipedia for knowledge-intensive NLP tasks."
        ),
    },
    "mrkl": {
        "arxiv_id": "2205.00445",
        "title": "MRKL Systems",
        "description": (
            "Proposes MRKL Systems, a neuro-symbolic architecture that "
            "routes queries between a large language model and external "
            "expert modules."
        ),
    },
    "mistral7b": {
        "arxiv_id": "2310.06825",
        "title": "Mistral 7B",
        "description": (
            "Introduces Mistral 7B, an efficient 7-billion-parameter "
            "language model using grouped-query and sliding window "
            "attention."
        ),
    },
    "judge": {
        "arxiv_id": "2306.05685",
        "title": "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena",
        "description": (
            "Studies using strong LLMs as judges of other LLMs' outputs, "
            "introducing MT-Bench and Chatbot Arena and analyzing judge "
            "biases and agreement with human preferences."
        ),
    },
    "survey": {
        "arxiv_id": "2312.10997",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "description": (
            "Surveys Retrieval-Augmented Generation techniques for large "
            "language models, covering Naive, Advanced, and Modular RAG "
            "paradigms."
        ),
    },
}


def strip_references(text: str) -> str:
    matches = list(re.finditer(r"\n\s*References\s*\n", text, re.IGNORECASE))
    if matches:
        text = text[: matches[-1].start()]
    return text


def fetch_paper(paper_key: str, arxiv_id: str) -> Document:
    loader = ArxivLoader(query=arxiv_id, load_max_docs=1)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No document returned for {paper_key} ({arxiv_id})")
    doc = docs[0]
    doc.page_content = strip_references(doc.page_content)
    doc.metadata["paper_key"] = paper_key
    doc.metadata["arxiv_id"] = arxiv_id
    doc.metadata["chunk_type"] = "body"
    return doc


def build_synthetic_chunks() -> list[Document]:
    doc_list_lines = ["Available Documents:"]
    for info in TARGET_PAPERS.values():
        doc_list_lines.append(f"- {info['title']} (arXiv:{info['arxiv_id']})")
    doc_list_chunk = Document(
        page_content="\n".join(doc_list_lines),
        metadata={"paper_key": "meta", "chunk_type": "doc_list"},
    )

    metadata_chunks = []
    for paper_key, info in TARGET_PAPERS.items():
        content = (
            f"Title: {info['title']}\n"
            f"arXiv ID: {info['arxiv_id']}\n"
            f"Description: {info['description']}"
        )
        metadata_chunks.append(
            Document(
                page_content=content,
                metadata={
                    "paper_key": paper_key,
                    "arxiv_id": info["arxiv_id"],
                    "chunk_type": "paper_metadata",
                },
            )
        )
    return [doc_list_chunk] + metadata_chunks


def main():
    print(f"Fetching {len(TARGET_PAPERS)} papers via ArxivLoader...")
    papers = [
        fetch_paper(paper_key, info["arxiv_id"])
        for paper_key, info in TARGET_PAPERS.items()
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    body_chunks = splitter.split_documents(papers)
    synthetic_chunks = build_synthetic_chunks()
    all_chunks = body_chunks + synthetic_chunks

    print("Loading ONNX embedder...")
    embeddings = get_embedder()
    embedding_dim = len(embeddings.embed_query("dimension check"))

    print(f"Embedding {len(all_chunks)} chunks and building FAISS index...")
    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    per_paper_counts = Counter(chunk.metadata["paper_key"] for chunk in all_chunks)

    print("\n--- Ingestion summary ---")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Embedding dimension: {embedding_dim}")
    print("Chunks per paper:")
    for paper_key in TARGET_PAPERS:
        print(f"  {paper_key}: {per_paper_counts[paper_key]}")
    print(f"  meta (doc-list): {per_paper_counts['meta']}")
    print(f"Index persisted to {INDEX_PATH}/")


if __name__ == "__main__":
    main()

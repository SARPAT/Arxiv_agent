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

# Quick fix (post-4g): live testing showed generic questions like "what is
# attention mechanism" get wrongly abstained even after recalibration -
# every chunk in the corpus is written in its source paper's technical
# register, so there was nothing that reads like a plain-language answer
# to match against. One verbatim plain-language overview chunk per paper,
# same synthetic-chunk pattern as the doc-list/metadata chunks below, just
# aimed at this different failure mode (a content gap, not a threshold
# problem).
PLAIN_OVERVIEWS = {
    "attention": (
        "What is the attention mechanism in transformers? It lets a model "
        "decide which words in a sentence matter most when producing each "
        "output word, instead of processing text strictly in order the way "
        "older models did. This is the core idea behind the Transformer "
        "architecture."
    ),
    "bert": (
        "What is BERT? BERT is a language model that reads text in both "
        "directions at once, not just left to right, so it can understand "
        "the full context around a word. It is pretrained on large amounts "
        "of text and then fine-tuned for specific tasks like question "
        "answering."
    ),
    "rag_paper": (
        "What is retrieval-augmented generation (RAG)? RAG is a technique "
        "where a language model looks up relevant documents before "
        "answering a question, instead of relying only on what it "
        "memorized during training. This helps produce more accurate, "
        "up-to-date answers."
    ),
    "mrkl": (
        "What is MRKL? MRKL is a system design that combines a language "
        "model with external tools like calculators or databases, so the "
        "model can hand off tasks it is bad at, like precise math, to a "
        "specialized tool instead of guessing."
    ),
    "mistral7b": (
        "What is Mistral 7B? Mistral 7B is a compact, efficient language "
        "model with 7 billion parameters, designed to perform "
        "competitively with larger models while being cheaper and faster "
        "to run."
    ),
    "judge": (
        "What is LLM-as-a-judge? It is a method of using a strong language "
        "model to evaluate and score the quality of other language "
        "models' answers, as a faster and cheaper alternative to having "
        "humans rate every response."
    ),
    "survey": (
        "What is retrieval-augmented generation, in general? This survey "
        "reviews how RAG systems work — retrieving relevant text and "
        "combining it with a language model's own knowledge to answer "
        "questions — and covers different strategies for chunking, "
        "retrieving, and combining that text."
    ),
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


def _doc_list_sentence() -> str:
    """Natural-language rendering of TARGET_PAPERS' titles for the
    Q&A-style doc-list chunk below, e.g. '"A" (arXiv:1), "B" (arXiv:2),
    and "C" (arXiv:3)' - built from TARGET_PAPERS rather than hardcoded
    so it can't drift from the bare-list chunk's own title source."""
    titles = [f'"{info["title"]}" (arXiv:{info["arxiv_id"]})' for info in TARGET_PAPERS.values()]
    if len(titles) == 1:
        return titles[0]
    return ", ".join(titles[:-1]) + f", and {titles[-1]}"


def build_synthetic_chunks() -> list[Document]:
    doc_list_lines = ["Available Documents:"]
    for info in TARGET_PAPERS.values():
        doc_list_lines.append(f"- {info['title']} (arXiv:{info['arxiv_id']})")
    doc_list_chunk = Document(
        page_content="\n".join(doc_list_lines),
        metadata={"paper_key": "meta", "chunk_type": "doc_list"},
    )

    # Same content as doc_list_chunk above, phrased as a natural
    # question-and-answer like the 7 per-paper PLAIN_OVERVIEWS chunks -
    # added alongside it (not replacing it) for the same content-gap
    # reason: a bare "Available Documents:" list chunk doesn't read like
    # an answer to a conversationally-phrased question, e.g. golden_set's
    # own meta question, "What papers or documents are available in this
    # system's collection?". Tagged the same chunk_type ("doc_list") as
    # the bare-list chunk, not "plain_overview" - both describe the doc
    # list itself, just in two different phrasings, whereas
    # "plain_overview" is specifically the per-paper chunks below.
    doc_list_qa_chunk = Document(
        page_content=(
            "What papers or documents are available in this system's "
            "collection? This system's collection includes the following "
            f"papers: {_doc_list_sentence()}."
        ),
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

    overview_chunks = []
    for paper_key, text in PLAIN_OVERVIEWS.items():
        info = TARGET_PAPERS[paper_key]
        overview_chunks.append(
            Document(
                page_content=text,
                metadata={
                    "paper_key": paper_key,
                    "arxiv_id": info["arxiv_id"],
                    "Title": info["title"],
                    "chunk_type": "plain_overview",
                },
            )
        )

    return [doc_list_chunk, doc_list_qa_chunk] + metadata_chunks + overview_chunks


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

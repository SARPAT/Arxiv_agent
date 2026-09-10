"""Verify the quick-fix plain_overview synthetic chunks in
ingestion/build_index.py: one per paper, verbatim text, correct
paper_key/Title/chunk_type metadata for Sources: attribution, and that
the existing doc_list/paper_metadata chunks are untouched."""
from ingestion.build_index import (
    PLAIN_OVERVIEWS,
    TARGET_PAPERS,
    build_synthetic_chunks,
)

EXPECTED_TEXT = {
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

# 1. PLAIN_OVERVIEWS has exactly one entry per target paper, verbatim text
assert set(PLAIN_OVERVIEWS.keys()) == set(TARGET_PAPERS.keys())
for key, text in EXPECTED_TEXT.items():
    assert PLAIN_OVERVIEWS[key] == text, f"mismatch for {key}"
print("PASSED: PLAIN_OVERVIEWS has exactly 7 entries, verbatim text matches the spec.")

# 2. build_synthetic_chunks() returns 1 doc_list (untouched bare-list) + 7
# paper_metadata + 8 plain_overview (7 per-paper + 1 new meta one) = 16
chunks = build_synthetic_chunks()
assert len(chunks) == 16, len(chunks)
overview_chunks = [c for c in chunks if c.metadata["chunk_type"] == "plain_overview"]
metadata_chunks = [c for c in chunks if c.metadata["chunk_type"] == "paper_metadata"]
doc_list_chunks = [c for c in chunks if c.metadata["chunk_type"] == "doc_list"]
assert len(overview_chunks) == 8, len(overview_chunks)
assert len(metadata_chunks) == 7, len(metadata_chunks)
assert len(doc_list_chunks) == 1, len(doc_list_chunks)
print("PASSED: build_synthetic_chunks() returns 1 doc_list + 7 paper_metadata + 8 plain_overview chunks.")

# 2b. The pre-existing bare-list doc_list chunk is untouched (regression
# check), and the new Q&A counterpart lives among overview_chunks instead,
# tagged chunk_type="plain_overview" (not "doc_list") per the corrected
# metadata spec, with arxiv_id=None since it isn't about one paper.
bare_list_chunk = doc_list_chunks[0]
assert bare_list_chunk.metadata == {"paper_key": "meta", "chunk_type": "doc_list"}
assert bare_list_chunk.page_content.startswith("Available Documents:")

meta_overview_chunks = [c for c in overview_chunks if c.metadata["paper_key"] == "meta"]
assert len(meta_overview_chunks) == 1, len(meta_overview_chunks)
qa_chunk = meta_overview_chunks[0]
assert qa_chunk.metadata == {"paper_key": "meta", "arxiv_id": None, "chunk_type": "plain_overview"}
assert qa_chunk.page_content.startswith(
    "What papers or documents are available in this system's collection?"
), qa_chunk.page_content
for info in TARGET_PAPERS.values():
    assert info["title"] in qa_chunk.page_content, info["title"]
    assert info["arxiv_id"] in qa_chunk.page_content, info["arxiv_id"]
print("PASSED: the new meta-tagged plain_overview chunk is additive, correctly tagged (arxiv_id=None), and mentions every paper's title/arxiv_id.")

# 3. Every per-paper plain_overview chunk has correct paper_key, Title,
# arxiv_id, chunk_type, and its page_content is exactly the verbatim text
# (no wrapping/mutation). Excludes the new meta-tagged chunk above, which
# isn't per-paper and is checked separately.
per_paper_overview_chunks = [c for c in overview_chunks if c.metadata["paper_key"] != "meta"]
by_key = {c.metadata["paper_key"]: c for c in per_paper_overview_chunks}
assert set(by_key.keys()) == set(TARGET_PAPERS.keys())
for paper_key, chunk in by_key.items():
    info = TARGET_PAPERS[paper_key]
    assert chunk.metadata["Title"] == info["title"], (paper_key, chunk.metadata)
    assert chunk.metadata["arxiv_id"] == info["arxiv_id"], (paper_key, chunk.metadata)
    assert chunk.metadata["chunk_type"] == "plain_overview"
    assert chunk.page_content == EXPECTED_TEXT[paper_key], paper_key
print("PASSED: every plain_overview chunk has correct paper_key/Title/arxiv_id metadata and verbatim content.")

# 4. Sources: attribution (rag/pipeline.py's extract logic) resolves to the
# real paper title for these chunks, same as it would for a real ArxivLoader
# body chunk (which also carries a "Title" key) - NOT the bare paper_key,
# which is what would happen without the "Title" metadata key.
for paper_key, chunk in by_key.items():
    title = chunk.metadata.get("Title", chunk.metadata.get("paper_key", "Unknown"))
    assert title == TARGET_PAPERS[paper_key]["title"], paper_key
print("PASSED: attribution lookup (doc.metadata.get('Title', ...)) resolves to the real title, not the bare paper_key.")

# 5. The pre-existing bare-list doc_list chunk and paper_metadata chunks
# are unchanged in shape (regression check - this fix must not touch them).
assert bare_list_chunk.page_content == (
    "Available Documents:\n"
    + "\n".join(f"- {info['title']} (arXiv:{info['arxiv_id']})" for info in TARGET_PAPERS.values())
)
for c in metadata_chunks:
    assert set(c.metadata.keys()) == {"paper_key", "arxiv_id", "chunk_type"}
    assert c.metadata["chunk_type"] == "paper_metadata"
print("PASSED: existing doc_list/paper_metadata chunks are untouched by this change.")

print("\nALL PLAIN_OVERVIEW QUICK-FIX TESTS PASSED")

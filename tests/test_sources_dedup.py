"""Verify the doubled-Sources fix end to end (mocked generation, no network):
- corpus-grounded answer -> done.sources = the retrieved papers (structured,
  single rendering); model text carries NO "Sources:" line.
- general-knowledge answer -> done.sources empty; model text carries the
  inline general-knowledge marker; frontend renders no Sources block.
Also checks the SSE done frame, the prompt no longer instructs a Sources:
line, and ui/app.py only renders sources when the array is non-empty."""
from unittest.mock import patch

from langchain_core.documents import Document

import rag.pipeline as pipeline
from ingestion.build_index import TARGET_PAPERS
from rag.generation import GENERAL_KNOWLEDGE_MARKER, SYSTEM_PROMPT

ATTN = TARGET_PAPERS["attention"]["title"]
BERT = TARGET_PAPERS["bert"]["title"]

# Retrieval always returns k chunks (no gate). Two attention chunks + one
# bert chunk + the synthetic doc-list meta chunk (not a real paper).
def fake_retrieve(query, k=4, tenant_ids=None):
    # tenant_ids added in Checkpoint 7; the real retrieve() now takes it.
    return [
        (Document(page_content="a1", metadata={"paper_key": "attention", "Title": ATTN}), 0.4),
        (Document(page_content="a2", metadata={"paper_key": "attention", "Title": ATTN}), 0.5),
        (Document(page_content="b1", metadata={"paper_key": "bert", "Title": BERT}), 0.6),
        (Document(page_content="m", metadata={"paper_key": "meta", "chunk_type": "doc_list"}), 0.7),
    ][:k]

# --- 1. Prompt no longer tells the model to write its own Sources: line ---
low = SYSTEM_PROMPT.lower()
assert 'do not write your own "sources:"' in low, "prompt must forbid a model Sources: block"
assert "the application appends the source papers automatically" in low
assert GENERAL_KNOWLEDGE_MARKER in SYSTEM_PROMPT, "marker must be embedded verbatim in the prompt"
print("PASSED: system prompt removes the model Sources: instruction and embeds the general-knowledge marker.")

# --- 2. Grounded answer: model writes NO Sources: line; done.sources are the
# retrieved papers (deduped, meta excluded) as the single structured rendering ---
def grounded_stream(query, context, history=None):
    yield "The Transformer weighs tokens by attention. "
    yield "BERT reads text bidirectionally."

with patch.object(pipeline, "retrieve", fake_retrieve), patch.object(
    pipeline, "generate_stream", grounded_stream
):
    events = list(pipeline.run_pipeline_stream("what is attention?"))

answer = "".join(e["delta"] for e in events if e["type"] == "token")
done = events[-1]
assert done["type"] == "done"
assert "Sources:" not in answer, f"grounded model text must not contain a Sources: block: {answer!r}"
assert done["sources"] == [ATTN, BERT], done["sources"]  # deduped, in order, meta excluded
print("PASSED: grounded answer -> no model Sources: line; done.sources = retrieved papers (deduped, meta excluded).")

# --- 3. General-knowledge answer: inline marker present; done.sources empty ---
def gk_stream(query, context, history=None):
    yield "Cristiano Ronaldo is a professional footballer. "
    yield GENERAL_KNOWLEDGE_MARKER

with patch.object(pipeline, "retrieve", fake_retrieve), patch.object(
    pipeline, "generate_stream", gk_stream
):
    events = list(pipeline.run_pipeline_stream("who is Ronaldo?"))

answer = "".join(e["delta"] for e in events if e["type"] == "token")
done = events[-1]
assert GENERAL_KNOWLEDGE_MARKER in answer, "general-knowledge answer must carry the inline marker"
assert done["sources"] == [], f"general-knowledge answer must have empty done.sources: {done['sources']}"
print("PASSED: general-knowledge answer -> inline marker present; done.sources empty (despite chunks retrieved).")

# --- 4. done frame shape unchanged otherwise (sources + top1_score, no abstained) ---
assert set(done.keys()) == {"type", "sources", "top1_score"}, done.keys()
print("PASSED: done event carries only type/sources/top1_score.")

# --- 5. Frontend renders nothing for an empty sources array (no empty header) ---
import inspect

import ui.app as ui

src = inspect.getsource(ui.chat_fn) if hasattr(ui, "chat_fn") else inspect.getsource(ui)
# The done handler guards the Sources render behind a truthiness check on the array.
assert 'if data["sources"]:' in inspect.getsource(ui), (
    "ui/app.py must only render Sources when the array is non-empty"
)
print("PASSED: ui/app.py renders a Sources block only when done.sources is non-empty.")

# --- 6. Eval false-attribution now scans the whole answer (no Sources: block) ---
from eval.run_eval import check_false_attribution

assert check_false_attribution(f"Something about {ATTN} in prose.", [ATTN, BERT]) is True
assert check_false_attribution("Ronaldo plays football. " + GENERAL_KNOWLEDGE_MARKER, [ATTN, BERT]) is False
print("PASSED: eval check_false_attribution scans the full answer text.")

print("\nALL DOUBLED-SOURCES FIX TESTS PASSED")

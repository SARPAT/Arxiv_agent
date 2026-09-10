"""Verify the gate-removal checkpoint: no abstention path anywhere in the
live retrieve->generate flow, similarity_threshold gone from config, the
done event/SSE frame carry no 'abstained' key, and the system prompt
enforces the ground/ignore/label/Sources contract. Retrieval and
generation are mocked so this runs with no network."""
from unittest.mock import patch

from langchain_core.documents import Document

import rag.pipeline as pipeline
from app.config import settings

# --- 1. similarity_threshold is gone from config; retrieval_k etc. remain ---
assert not hasattr(settings, "similarity_threshold"), "similarity_threshold must be removed"
assert hasattr(settings, "retrieval_k")
print("PASSED: settings.similarity_threshold removed; other settings intact.")

# --- 2. rag/gate.py is gone and nothing imports it ---
import importlib.util

assert importlib.util.find_spec("rag.gate") is None, "rag/gate.py should be deleted"
print("PASSED: rag/gate.py no longer exists.")

# --- 3. PipelineResult has no `abstained` field ---
from dataclasses import fields

field_names = {f.name for f in fields(pipeline.PipelineResult)}
assert "abstained" not in field_names, field_names
assert field_names == {"answer", "top1_score", "context", "docs"}, field_names
print("PASSED: PipelineResult dropped `abstained`.")

# --- Shared fakes: a FAR top-1 chunk (would have been abstained under the
# old gate) to prove generation now runs regardless of retrieval distance. ---
far_doc = Document(
    page_content="Some retrieved paper text about transformers.",
    metadata={"paper_key": "attention", "Title": "Attention Is All You Need"},
)
FAR_SCORE = 0.95  # well above any threshold the old gate ever used


def fake_retrieve(query, k=4, tenant_ids=None):
    # tenant_ids added in Checkpoint 7; the real retrieve() now takes it.
    return [(far_doc, FAR_SCORE)] + [(far_doc, FAR_SCORE + 0.01 * i) for i in range(1, k)]


# --- 4. run_pipeline always generates (no abstain), even on a far top-1 ---
def fake_generate(query, context, history=None):
    assert context, "context should be assembled and passed to generation"
    return "An answer.\n\nSources: Attention Is All You Need"


with patch.object(pipeline, "retrieve", fake_retrieve), patch.object(
    pipeline, "generate", fake_generate
):
    result = pipeline.run_pipeline("what is attention?")
assert result.answer == "An answer.\n\nSources: Attention Is All You Need"
assert result.top1_score == FAR_SCORE
assert result.context, "context must be non-empty (generation ran)"
print("PASSED: run_pipeline() always generates - no abstain path, even on a far top-1 score.")


# --- 5. run_pipeline_stream: no abstain branch, done event has no 'abstained' ---
def fake_generate_stream(query, context, history=None):
    yield "An "
    yield "answer. Based on general knowledge, not the provided papers."


with patch.object(pipeline, "retrieve", fake_retrieve), patch.object(
    pipeline, "generate_stream", fake_generate_stream
):
    events = list(pipeline.run_pipeline_stream("who is Ronaldo?"))

types = [e["type"] for e in events]
assert types.count("token") >= 1, types
assert types[-1] == "done", types
done = events[-1]
assert "abstained" not in done, f"done event must not carry 'abstained': {done}"
assert set(done.keys()) == {"type", "sources", "top1_score"}, done.keys()
# General-knowledge answer cites no corpus paper -> sources empty, and that's fine.
assert done["sources"] == [], done["sources"]
# The old ABSTAIN_RESPONSE must never appear.
full = "".join(e["delta"] for e in events if e["type"] == "token")
assert "I don't have information on that in my corpus" not in full
print("PASSED: run_pipeline_stream() always generates; done event has no 'abstained'; no abstain sentinel emitted.")

# --- 6. A corpus-grounded stream still surfaces the cited paper in done.sources ---
def fake_generate_stream_grounded(query, context, history=None):
    yield "Attention lets a model weigh tokens.\n\nSources: Attention Is All You Need"


with patch.object(pipeline, "retrieve", fake_retrieve), patch.object(
    pipeline, "generate_stream", fake_generate_stream_grounded
):
    events = list(pipeline.run_pipeline_stream("what is attention?"))
assert events[-1]["sources"] == ["Attention Is All You Need"], events[-1]
print("PASSED: a corpus-grounded answer still reports its cited paper in done.sources.")

# --- 7. System prompt enforces the four-part contract ---
from rag.generation import SYSTEM_PROMPT

low = SYSTEM_PROMPT.lower()
assert "primary source of truth" in low
assert "does not address" in low or "not address" in low  # ignore-irrelevant license
assert "ronaldo" in low  # the concrete ignore-irrelevant example from the spec
assert "general knowledge" in low
assert "never" in low and ("fabricate" in low or "invent" in low)  # no fabricated cites
assert "Based on general knowledge, not the provided papers." in SYSTEM_PROMPT
assert "sources:" in low
print("PASSED: SYSTEM_PROMPT enforces ground-when-relevant / ignore-irrelevant / label-general-knowledge / Sources contract.")

# --- 8. app/api.py's done SSE frame carries no 'abstained' ---
import inspect

import app.api as api

api_src = inspect.getsource(api._stream_chat)
assert "abstained" not in api_src, "app/api.py _stream_chat must not reference abstained"
print("PASSED: app/api.py done frame no longer includes 'abstained'.")

# --- 9. eval tooling still imports and summarize() works without gate metrics ---
import eval.run_eval as run_eval
import eval.calibrate_threshold  # noqa: F401  (must stay importable - calibrate imports load_golden_set)

summary = run_eval.summarize(
    in_corpus_results=[
        {"hit_at_5": True, "hit_at_8": True, "reciprocal_rank": 1.0, "context_survived_top_chunk": True},
    ],
    out_of_corpus_results=[
        {"subtype": "unrelated", "false_accept": False, "context_survived_top_chunk": True},
    ],
)
assert "false_reject_rate" not in summary, "gate metric should be gone"
assert "n_non_abstained" not in summary, "gate metric should be gone"
assert summary["recall_at_5"] == 1.0
assert summary["false_accept_rate"] == 0.0
assert summary["context_survival_rate"] == 1.0
print("PASSED: run_eval.summarize() dropped gate metrics, keeps retrieval/false-accept/context-survival.")

print("\nALL GATE-REMOVAL CHECKPOINT TESTS PASSED")

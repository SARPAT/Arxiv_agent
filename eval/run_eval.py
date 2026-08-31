"""Evaluate the gated RAG pipeline against eval/golden_set.jsonl.

Calls ``rag/pipeline.py`` and ``rag/retrieval.py`` directly, in-process.
``golden_set.jsonl`` is read as-is and never modified by this script.

Every question passes through the confidence gate (``rag/gate.py``): a
question can be abstained on regardless of category, so this script
tracks abstain/proceed decisions across both categories rather than
splitting gate evaluation off from category-specific correctness checks.

In-corpus questions are evaluated with retrieval and gate/context-assembly
checks only — no generation call, since neither ``context_survival_rate``
nor ``false_reject_rate`` depends on what the model would say. Generation
only runs for out-of-corpus questions the gate doesn't abstain on, since
that's the only case where a false attribution is even possible.

Metrics:

- ``recall_at_5`` / ``recall_at_8`` / ``mrr``: retrieval quality on
  in-corpus questions, matched by the golden set's ``paper_key`` field
  against each question's retrieved chunks. Independent of the gate —
  these measure what retrieval found, not what the gate decided to do
  with it.
- ``false_reject_rate``: fraction of in-corpus questions the gate
  abstained on — questions the corpus could answer that the pipeline
  declined to attempt.
- ``false_accept_rate`` (overall, and split by ``unrelated``/``adjacent``
  subtype): fraction of out-of-corpus questions where a non-abstained
  response's "Sources:" block cited a real paper title. There is no
  correct paper to cite for an out-of-corpus question, so any citation is
  a false one.
- ``context_survival_rate``: fraction of non-abstained questions (both
  categories) where the single closest retrieved chunk's content actually
  appears in the context string ``assemble_context()`` produced.

Requires ``SIMILARITY_THRESHOLD`` to be set in ``rag/gate.py`` (see
``eval/calibrate_threshold.py``) — every question goes through the gate,
so this script cannot run against the placeholder value.

Writes ``eval/results_checkpoint3.json`` (full per-question detail) and
``eval/summary_checkpoint3.json`` (headline numbers).
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document

from eval.utils import save_json
from ingestion.build_index import TARGET_PAPERS
from rag.pipeline import assemble_context, retrieve_and_gate, run_pipeline
from rag.retrieval import retrieve

GOLDEN_SET_PATH = Path("eval/golden_set.jsonl")
RESULTS_PATH = Path("eval/results_checkpoint3.json")
SUMMARY_PATH = Path("eval/summary_checkpoint3.json")

RETRIEVAL_EVAL_K = 8  # covers both Recall@5 and Recall@8 from one retrieval call

REAL_PAPER_TITLES = [info["title"] for info in TARGET_PAPERS.values()]


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Read golden_set.jsonl into a list of question dicts, unmodified."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def expected_paper_keys(entry: dict) -> list[str]:
    """The paper key(s) a correct retrieval should surface for this question.

    Every question in this golden set has exactly one expected key today
    (including the meta-question, whose expected key is the "meta"
    sentinel used by the synthetic doc-list chunk) — this returns a list
    in case a future golden-set question ever legitimately expects more
    than one source paper.
    """
    return [entry["paper_key"]]


def extract_sources_block(response_text: str) -> str:
    """Return the text of the response's "Sources:" block, or "" if absent."""
    match = re.search(r"Sources:\s*(.*)", response_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def check_false_attribution(sources_text: str, real_titles: list[str]) -> bool:
    """True if `sources_text` cites any of the corpus's real paper titles.

    Used only on out-of-corpus questions, where *any* real title cited as
    a source is by definition a false attribution — there is no genuinely
    relevant paper for the model to have correctly cited.
    """
    lowered = sources_text.lower()
    return any(title.lower() in lowered for title in real_titles)


def top_chunk_survived(top_doc: Document, context: str) -> bool:
    """Whether the single closest retrieved chunk's content appears in the
    final context string handed to generation."""
    return top_doc.page_content in context


def evaluate_in_corpus(entry: dict) -> dict:
    """Retrieval metrics plus gate/context-assembly outcome for one
    in-corpus question. No generation call is made."""
    expected = set(expected_paper_keys(entry))
    retrieved = retrieve(entry["question"], k=RETRIEVAL_EVAL_K)
    retrieved_keys = [doc.metadata.get("paper_key") for doc, _score in retrieved]

    hit_at_5 = any(key in expected for key in retrieved_keys[:5])
    hit_at_8 = any(key in expected for key in retrieved_keys[:8])

    reciprocal_rank = 0.0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in expected:
            reciprocal_rank = 1.0 / rank
            break

    docs, top1_score, abstained = retrieve_and_gate(entry["question"])
    survived = None
    if not abstained:
        context = assemble_context(docs)
        survived = top_chunk_survived(docs[0], context)

    return {
        "id": entry["id"],
        "question": entry["question"],
        "category": "in_corpus",
        "expected_paper_keys": sorted(expected),
        "retrieved_paper_keys": retrieved_keys,
        "hit_at_5": hit_at_5,
        "hit_at_8": hit_at_8,
        "reciprocal_rank": reciprocal_rank,
        "top1_score": top1_score,
        "abstained": abstained,
        "context_survived_top_chunk": survived,
    }


def evaluate_out_of_corpus(entry: dict) -> dict:
    """Full gated pipeline (including generation, when not abstained) for
    one out-of-corpus question, checked for false attribution."""
    result = run_pipeline(entry["question"])

    survived = None
    false_accept = False
    sources_text = ""
    if not result.abstained:
        survived = top_chunk_survived(result.docs[0], result.context)
        sources_text = extract_sources_block(result.answer)
        false_accept = check_false_attribution(sources_text, REAL_PAPER_TITLES)

    return {
        "id": entry["id"],
        "question": entry["question"],
        "category": "out_of_corpus",
        "subtype": entry["subtype"],
        "top1_score": result.top1_score,
        "abstained": result.abstained,
        "response": result.answer,
        "sources_block": sources_text,
        "context_survived_top_chunk": survived,
        "false_accept": false_accept,
    }


def summarize(
    in_corpus_results: list[dict], out_of_corpus_results: list[dict]
) -> dict:
    """Compute headline metrics from the per-question detail results."""
    n_ic = len(in_corpus_results)
    recall_at_5 = sum(r["hit_at_5"] for r in in_corpus_results) / n_ic
    recall_at_8 = sum(r["hit_at_8"] for r in in_corpus_results) / n_ic
    mrr = sum(r["reciprocal_rank"] for r in in_corpus_results) / n_ic
    false_reject_rate = sum(r["abstained"] for r in in_corpus_results) / n_ic

    n_ooc = len(out_of_corpus_results)
    false_accept_rate = sum(r["false_accept"] for r in out_of_corpus_results) / n_ooc

    def false_accept_rate_for(subtype: str) -> float:
        subset = [r for r in out_of_corpus_results if r["subtype"] == subtype]
        if not subset:
            return 0.0
        return sum(r["false_accept"] for r in subset) / len(subset)

    all_results = in_corpus_results + out_of_corpus_results
    non_abstained = [r for r in all_results if not r["abstained"]]
    context_survival_rate = (
        sum(r["context_survived_top_chunk"] for r in non_abstained)
        / len(non_abstained)
        if non_abstained
        else 0.0
    )

    return {
        "n_in_corpus_questions": n_ic,
        "n_out_of_corpus_questions": n_ooc,
        "n_non_abstained": len(non_abstained),
        "recall_at_5": recall_at_5,
        "recall_at_8": recall_at_8,
        "mrr": mrr,
        "false_accept_rate": false_accept_rate,
        "false_accept_rate_unrelated": false_accept_rate_for("unrelated"),
        "false_accept_rate_adjacent": false_accept_rate_for("adjacent"),
        "false_reject_rate": false_reject_rate,
        "context_survival_rate": context_survival_rate,
    }


def main():
    golden_set = load_golden_set()
    in_corpus_entries = [e for e in golden_set if e["category"] == "in_corpus"]
    out_of_corpus_entries = [e for e in golden_set if e["category"] == "out_of_corpus"]

    print(
        f"Evaluating {len(in_corpus_entries)} in-corpus questions "
        "(retrieval + gate, no generation)..."
    )
    in_corpus_results = [evaluate_in_corpus(e) for e in in_corpus_entries]

    print(
        f"Evaluating {len(out_of_corpus_entries)} out-of-corpus questions "
        "(full gated pipeline)..."
    )
    out_of_corpus_results = [evaluate_out_of_corpus(e) for e in out_of_corpus_entries]

    summary = summarize(in_corpus_results, out_of_corpus_results)

    results = {
        "in_corpus": in_corpus_results,
        "out_of_corpus": out_of_corpus_results,
    }

    save_json(RESULTS_PATH, results)
    save_json(SUMMARY_PATH, summary)

    print("\n--- Evaluation summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nFull detail written to {RESULTS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

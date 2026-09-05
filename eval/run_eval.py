"""Evaluate the RAG pipeline against eval/golden_set.jsonl.

Calls ``rag/pipeline.py`` and ``rag/retrieval.py`` directly, in-process.
``golden_set.jsonl`` is read as-is and never modified by this script.

There is no longer a confidence gate: the pipeline always retrieves top-k
and always generates (the gate and its abstain/proceed decision were
removed - see ``rag/pipeline.py``). So this script no longer tracks
abstention, and the gate-derived metrics it used to report
(``false_reject_rate``, ``n_non_abstained``) are gone with it. What
remains measures the two things the no-gate pipeline can still get wrong:
whether retrieval surfaces the right paper, and whether the model
falsely attributes an out-of-corpus answer to a real paper.

In-corpus questions are evaluated on retrieval and context-assembly only
- no generation call, since retrieval quality and context survival don't
depend on what the model would say. Generation runs for every
out-of-corpus question, since that's the only case a false attribution
can occur.

Metrics:

- ``recall_at_5`` / ``recall_at_8`` / ``mrr``: retrieval quality on
  in-corpus questions, matched by the golden set's ``expected_papers``
  field against each question's retrieved chunks.
- ``false_accept_rate`` (overall, and split by ``unrelated``/
  ``adjacent_uncovered`` subtype): fraction of out-of-corpus questions
  whose answer text names a real paper title anywhere (the model no longer
  writes a separate "Sources:" block, so the whole answer is scanned).
  There is no correct paper to cite for an out-of-corpus question, so
  naming one is a false one - and with the gate gone, the system prompt's
  general-knowledge labeling is the only thing standing between an
  out-of-corpus question and a false attribution, which is exactly what
  this metric now measures.
- ``context_survival_rate``: fraction of all questions where the single
  closest retrieved chunk's content actually appears in the context
  string ``assemble_context()`` produced.

Writes ``eval/results_<output-suffix>.json`` (full per-question detail)
and ``eval/summary_<output-suffix>.json`` (headline numbers), where
``--output-suffix`` (required) names the checkpoint this run is for, e.g.
``checkpoint4f``. Required, not defaulted: an earlier run of this script
against Checkpoint 3's fixed ``results_checkpoint3.json`` path silently
overwrote Checkpoint 3's own historical numbers the next time someone ran
it for a later checkpoint — a required flag makes that impossible to
repeat by accident.
"""

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from eval.utils import save_json
from rag.pipeline import (
    REAL_PAPER_TITLES,
    assemble_context,
    retrieve_context,
    run_pipeline,
)
from rag.retrieval import retrieve

GOLDEN_SET_PATH = Path("eval/golden_set.jsonl")

RETRIEVAL_EVAL_K = 8  # covers both Recall@5 and Recall@8 from one retrieval call


def result_paths(output_suffix: str) -> tuple[Path, Path]:
    """Return the ``(results_path, summary_path)`` this run should write
    to, given ``--output-suffix``."""
    return (
        Path(f"eval/results_{output_suffix}.json"),
        Path(f"eval/summary_{output_suffix}.json"),
    )


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[dict]:
    """Read golden_set.jsonl into a list of question dicts, unmodified."""
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# Checkpoint 4g's golden set names the meta-question's expected source
# "synthetic_doclist", but the actual synthetic doc-list chunk built in
# ingestion/build_index.py (unchanged, out of this checkpoint's scope)
# tags itself with paper_key "meta" - this alias bridges that naming
# mismatch rather than silently never matching the meta question.
_PAPER_KEY_ALIASES = {"synthetic_doclist": "meta"}


def expected_paper_keys(entry: dict) -> list[str]:
    """The paper key(s) a correct retrieval should surface for this
    question, aliased to match the retrieved chunks' actual metadata."""
    return [_PAPER_KEY_ALIASES.get(key, key) for key in entry["expected_papers"]]


def check_false_attribution(answer_text: str, real_titles: list[str]) -> bool:
    """True if `answer_text` names any of the corpus's real paper titles.

    Used only on out-of-corpus questions, where *any* real title named
    anywhere in the answer is by definition a false attribution — there is
    no genuinely relevant paper for the model to have correctly cited.
    Scans the whole answer (the model no longer emits a separate "Sources:"
    block to isolate).
    """
    lowered = answer_text.lower()
    return any(title.lower() in lowered for title in real_titles)


def top_chunk_survived(top_doc: Document, context: str) -> bool:
    """Whether the single closest retrieved chunk's content appears in the
    final context string handed to generation."""
    return top_doc.page_content in context


def evaluate_in_corpus(entry: dict) -> dict:
    """Retrieval metrics plus context-assembly outcome for one in-corpus
    question. No generation call is made."""
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

    docs, top1_score = retrieve_context(entry["question"])
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
        "context_survived_top_chunk": survived,
    }


def evaluate_out_of_corpus(entry: dict) -> dict:
    """Full pipeline (including generation) for one out-of-corpus question,
    checked for false attribution. Every question generates now - there is
    no gate - so this is the metric that catches the system prompt failing
    to label an out-of-corpus answer as general knowledge.

    The model no longer emits its own "Sources:" block (see
    rag/generation.py), so false attribution is detected by scanning the
    whole answer text for any real corpus paper title: for an out-of-corpus
    question there is no paper that should be named, so naming one anywhere
    is a false attribution."""
    result = run_pipeline(entry["question"])

    survived = top_chunk_survived(result.docs[0], result.context)
    false_accept = check_false_attribution(result.answer, REAL_PAPER_TITLES)

    return {
        "id": entry["id"],
        "question": entry["question"],
        "category": "out_of_corpus",
        "subtype": entry["subtype"],
        "top1_score": result.top1_score,
        "response": result.answer,
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

    n_ooc = len(out_of_corpus_results)
    false_accept_rate = sum(r["false_accept"] for r in out_of_corpus_results) / n_ooc

    def false_accept_rate_for(subtype: str) -> float:
        subset = [r for r in out_of_corpus_results if r["subtype"] == subtype]
        if not subset:
            return 0.0
        return sum(r["false_accept"] for r in subset) / len(subset)

    # Every question now proceeds to context assembly (no gate), so this is
    # measured over all of them rather than only the non-abstained subset.
    all_results = in_corpus_results + out_of_corpus_results
    context_survival_rate = (
        sum(r["context_survived_top_chunk"] for r in all_results) / len(all_results)
        if all_results
        else 0.0
    )

    return {
        "n_in_corpus_questions": n_ic,
        "n_out_of_corpus_questions": n_ooc,
        "recall_at_5": recall_at_5,
        "recall_at_8": recall_at_8,
        "mrr": mrr,
        "false_accept_rate": false_accept_rate,
        "false_accept_rate_unrelated": false_accept_rate_for("unrelated"),
        "false_accept_rate_adjacent_uncovered": false_accept_rate_for(
            "adjacent_uncovered"
        ),
        "context_survival_rate": context_survival_rate,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-suffix",
        required=True,
        help=(
            "Names which checkpoint this run is for, e.g. 'checkpoint4f' - "
            "writes eval/results_<suffix>.json and eval/summary_<suffix>.json."
        ),
    )
    args = parser.parse_args()
    results_path, summary_path = result_paths(args.output_suffix)

    golden_set = load_golden_set()
    in_corpus_entries = [e for e in golden_set if e["type"] == "in_corpus"]
    out_of_corpus_entries = [e for e in golden_set if e["type"] == "out_of_corpus"]

    print(
        f"Evaluating {len(in_corpus_entries)} in-corpus questions "
        "(retrieval + context assembly, no generation)..."
    )
    in_corpus_results = [evaluate_in_corpus(e) for e in in_corpus_entries]

    print(
        f"Evaluating {len(out_of_corpus_entries)} out-of-corpus questions "
        "(full pipeline including generation)..."
    )
    out_of_corpus_results = [evaluate_out_of_corpus(e) for e in out_of_corpus_entries]

    summary = summarize(in_corpus_results, out_of_corpus_results)

    results = {
        "in_corpus": in_corpus_results,
        "out_of_corpus": out_of_corpus_results,
    }

    save_json(results_path, results)
    save_json(summary_path, summary)

    print("\n--- Evaluation summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nFull detail written to {results_path}")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()

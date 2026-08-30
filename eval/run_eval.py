"""Evaluate the Checkpoint 2 RAG pipeline against eval/golden_set.jsonl.

Calls ``rag/pipeline.py`` and ``rag/retrieval.py`` directly, in-process —
this is a from-scratch rewrite of an earlier ``run_eval_v0_baseline.py``
that targeted a since-retired ``nv-embed-v1``/``llama-3.1-8b-instruct``
deployment and no longer applies to this codebase.

``golden_set.jsonl`` is read as-is and never modified by this script. It
has two question categories:

- ``in_corpus`` (50 questions, including 1 meta-question): evaluated with
  retrieval-only metrics — Recall@5, Recall@8, and MRR — against the
  question's expected paper, matched by the ``paper_key`` field. (This is
  the same concept the Checkpoint 2 spec calls ``expected_papers``; this
  repo's golden set — built in Checkpoint 0 — encodes one expected paper
  per question via ``paper_key`` rather than a list field of that exact
  name. The meta-question's expected key is the sentinel ``"meta"``,
  matching the synthetic doc-list chunk's own metadata.)
- ``out_of_corpus`` (14 questions: 6 unrelated, 8 adjacent-but-uncovered):
  run through the full pipeline (retrieve + generate), then checked for
  false attribution — did the model's "Sources:" block cite one of the 7
  real paper titles despite there being no genuinely relevant chunk to
  cite?

Writes two files:

- ``eval/results_checkpoint2.json`` — full per-question detail
- ``eval/summary_checkpoint2.json`` — headline numbers only
"""

import json
import re
from pathlib import Path

from rag.pipeline import answer_query
from rag.retrieval import retrieve
from ingestion.build_index import TARGET_PAPERS

GOLDEN_SET_PATH = Path("eval/golden_set.jsonl")
RESULTS_PATH = Path("eval/results_checkpoint2.json")
SUMMARY_PATH = Path("eval/summary_checkpoint2.json")

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
    for parity with the checkpoint spec's ``expected_papers`` concept, in
    case a future golden-set question ever legitimately expects more than
    one source paper.
    """
    return [entry["paper_key"]]


def evaluate_retrieval(entry: dict) -> dict:
    """Run retrieval-only evaluation for one in-corpus question.

    Retrieves the top ``RETRIEVAL_EVAL_K`` chunks once and derives both
    Recall@5 and Recall@8 (and this question's contribution to MRR) from
    that single call, rather than retrieving twice with different ``k``.
    """
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

    return {
        "id": entry["id"],
        "question": entry["question"],
        "expected_paper_keys": sorted(expected),
        "retrieved_paper_keys": retrieved_keys,
        "hit_at_5": hit_at_5,
        "hit_at_8": hit_at_8,
        "reciprocal_rank": reciprocal_rank,
    }


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


def evaluate_out_of_corpus(entry: dict) -> dict:
    """Run the full pipeline on one out-of-corpus question and check for
    false attribution in its "Sources:" block."""
    response = answer_query(entry["question"])
    sources_text = extract_sources_block(response)
    false_accept = check_false_attribution(sources_text, REAL_PAPER_TITLES)

    return {
        "id": entry["id"],
        "question": entry["question"],
        "subtype": entry["subtype"],
        "response": response,
        "sources_block": sources_text,
        "false_accept": false_accept,
    }


def summarize(retrieval_results: list[dict], out_of_corpus_results: list[dict]) -> dict:
    """Compute headline metrics from the per-question detail results."""
    n_retrieval = len(retrieval_results)
    recall_at_5 = sum(r["hit_at_5"] for r in retrieval_results) / n_retrieval
    recall_at_8 = sum(r["hit_at_8"] for r in retrieval_results) / n_retrieval
    mrr = sum(r["reciprocal_rank"] for r in retrieval_results) / n_retrieval

    n_ooc = len(out_of_corpus_results)
    false_accept_rate = sum(r["false_accept"] for r in out_of_corpus_results) / n_ooc

    def false_accept_rate_for(subtype: str) -> float:
        subset = [r for r in out_of_corpus_results if r["subtype"] == subtype]
        if not subset:
            return 0.0
        return sum(r["false_accept"] for r in subset) / len(subset)

    return {
        "n_in_corpus_questions": n_retrieval,
        "n_out_of_corpus_questions": n_ooc,
        "recall_at_5": recall_at_5,
        "recall_at_8": recall_at_8,
        "mrr": mrr,
        "false_accept_rate": false_accept_rate,
        "false_accept_rate_unrelated": false_accept_rate_for("unrelated"),
        "false_accept_rate_adjacent": false_accept_rate_for("adjacent"),
    }


def main():
    golden_set = load_golden_set()
    in_corpus = [e for e in golden_set if e["category"] == "in_corpus"]
    out_of_corpus = [e for e in golden_set if e["category"] == "out_of_corpus"]

    print(f"Evaluating retrieval on {len(in_corpus)} in-corpus questions...")
    retrieval_results = [evaluate_retrieval(e) for e in in_corpus]

    print(f"Running full pipeline on {len(out_of_corpus)} out-of-corpus questions...")
    out_of_corpus_results = [evaluate_out_of_corpus(e) for e in out_of_corpus]

    summary = summarize(retrieval_results, out_of_corpus_results)

    results = {
        "retrieval": retrieval_results,
        "out_of_corpus": out_of_corpus_results,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\n--- Checkpoint 2 baseline summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nFull detail written to {RESULTS_PATH}")
    print(f"Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

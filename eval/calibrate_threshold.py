"""Calibrate the confidence gate's similarity threshold against the golden set.

Retrieves the top-1 chunk for every golden-set question, pairs its raw
FAISS L2 distance with the question's true label (1 = in-corpus, should
answer; 0 = out-of-corpus, should abstain), and selects a threshold using
a fixed-specificity target rather than an equal-cost method like Youden's
J: a false accept (answering confidently from an irrelevant chunk, with a
fabricated citation) is a worse failure than a false reject (declining to
answer a question the corpus could actually have answered), so the
selection method treats the two error types asymmetrically instead of
assuming they cost the same.

The ROC-AUC this script reports is a diagnostic on how separable the two
classes are by top-1 distance alone — it is not itself the threshold
selection method.

This computes ROC/AUC manually rather than depending on scikit-learn,
which is reasonable at this sample size (64 points) and avoids adding a
new dependency for one calibration script.

Requires network access (the embedding model and the persisted FAISS
index) — this cannot run in an environment with no route to Hugging Face.
Run this externally, then copy the printed threshold into ``rag/gate.py``'s
``SIMILARITY_THRESHOLD``.
"""

import json
from pathlib import Path

from eval.run_eval import load_golden_set
from rag.retrieval import retrieve

CALIBRATION_RESULT_PATH = Path("eval/calibration_result.json")
MIN_SPECIFICITY = 0.90


def collect_scores_and_labels(
    golden_set: list[dict],
) -> tuple[list[float], list[int], list[str]]:
    """Top-1 raw distance, true label (1 = in-corpus), and id for every
    golden-set question."""
    scores, labels, ids = [], [], []
    for entry in golden_set:
        _top1_doc, top1_score = retrieve(entry["question"], k=1)[0]
        scores.append(top1_score)
        labels.append(1 if entry["category"] == "in_corpus" else 0)
        ids.append(entry["id"])
    return scores, labels, ids


def roc_points(
    labels: list[int], scores: list[float]
) -> list[tuple[float, float, float]]:
    """``(threshold, fpr, tpr)`` triples across every candidate threshold.

    Decision rule: predict in-corpus (proceed) when ``score <= threshold``,
    matching ``gate.should_abstain``'s use of the raw distance directly.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    uniq = sorted(set(scores))
    candidates = [uniq[0] - 1e-9] + uniq + [uniq[-1] + 1e-9]

    points = []
    for t in candidates:
        tp = sum(1 for label, s in zip(labels, scores) if label == 1 and s <= t)
        fp = sum(1 for label, s in zip(labels, scores) if label == 0 and s <= t)
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        points.append((t, fpr, tpr))
    return points


def roc_auc(points: list[tuple[float, float, float]]) -> float:
    """Trapezoidal-rule area under the ROC curve from ``(threshold, fpr,
    tpr)`` points."""
    pts = sorted((fpr, tpr) for _, fpr, tpr in points)
    area = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def select_threshold(
    points: list[tuple[float, float, float]], min_specificity: float = MIN_SPECIFICITY
) -> tuple[float, float, float]:
    """Return the ``(threshold, fpr, tpr)`` that maximizes sensitivity
    (``tpr``) among thresholds meeting the specificity floor
    (``1 - fpr >= min_specificity``).

    Ties in sensitivity are broken by the strictest (lowest) qualifying
    threshold, since a stricter threshold costs nothing in sensitivity
    while adding specificity margin.
    """
    eligible = [
        (t, fpr, tpr) for t, fpr, tpr in points if (1 - fpr) >= min_specificity
    ]
    if not eligible:
        raise RuntimeError(
            f"No threshold reaches {min_specificity:.0%} specificity on this "
            "sample — the two classes are not separable enough at that bar."
        )
    best_tpr = max(tpr for _, _, tpr in eligible)
    best = [pt for pt in eligible if pt[2] == best_tpr]
    return min(best, key=lambda pt: pt[0])


def main():
    golden_set = load_golden_set()
    print(f"Retrieving top-1 score for {len(golden_set)} golden-set questions...")
    scores, labels, ids = collect_scores_and_labels(golden_set)

    n_ooc = len(labels) - sum(labels)
    print(
        f"WARNING: this threshold is calibrated on only {n_ooc} out-of-corpus "
        "examples — with a sample this small, the chosen threshold carries "
        "real sampling noise and should be treated as a reasonable starting "
        "point, not a precisely tuned value."
    )

    points = roc_points(labels, scores)
    auc = roc_auc(points)
    print(f"ROC-AUC (diagnostic only, not the selection method): {auc:.4f}")

    threshold, fpr, tpr = select_threshold(points)
    specificity = 1 - fpr
    print(f"\nSelected SIMILARITY_THRESHOLD = {threshold:.6f}")
    print(f"  sensitivity (recall on in-corpus):     {tpr:.4f}")
    print(f"  specificity (recall on out-of-corpus): {specificity:.4f}")

    result = {
        "scores": scores,
        "labels": labels,
        "ids": ids,
        "roc_auc": auc,
        "min_specificity_target": MIN_SPECIFICITY,
        "selected_threshold": threshold,
        "sensitivity_at_threshold": tpr,
        "specificity_at_threshold": specificity,
        "n_in_corpus": sum(labels),
        "n_out_of_corpus": n_ooc,
    }
    CALIBRATION_RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nFull calibration detail written to {CALIBRATION_RESULT_PATH}")
    print("Next step: copy the threshold above into rag/gate.py's SIMILARITY_THRESHOLD.")


if __name__ == "__main__":
    main()

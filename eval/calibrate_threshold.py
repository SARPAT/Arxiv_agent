"""Calibrate the confidence gate's similarity threshold against the golden set.

Retrieves the top-1 chunk for every golden-set question, pairs its raw
FAISS L2 distance with the question's true label (1 = in-corpus, should
answer; 0 = out-of-corpus, should abstain), and reports sensitivity and
specificity at each of several candidate specificity targets rather than
picking one target and one threshold automatically.

Checkpoint 4g reworked this from a single fixed-specificity-target
selection (originally >=0.90) to a full tradeoff curve, for two reasons
found from live testing and from the original golden set's small size:

1. A single target value (0.90) was picked without strong justification
   for that exact number over any other nearby one.
2. With only 14 out-of-corpus examples, specificity was locked to coarse
   1/14 (~7.1%) increments anyway - too coarse to meaningfully compare
   targets even a few points apart. The expanded 50-example out-of-corpus
   set (see golden_set.jsonl) gives 2% resolution instead.

This script deliberately does NOT select a final threshold or write one
into app/config.py - that choice happens in review, after seeing the real
curve, not automatically here. It does still additionally compute and
save the same single-value output this script produced before
(``eval/calibration_result.json``, at the historical 90% target), since
that format is still useful for whatever target is eventually chosen; the
curve (``eval/calibration_curve.json``) is the new primary deliverable.

The ROC-AUC this script reports is a diagnostic on how separable the two
classes are by top-1 distance alone — it is not itself the threshold
selection method.

This computes ROC/AUC manually rather than depending on scikit-learn,
which is reasonable at this sample size (200 points) and avoids adding a
new dependency for one calibration script.

Requires network access (the embedding model and the persisted FAISS
index) — this cannot run in an environment with no route to Hugging Face.
Run this externally, review the printed curve and eval/calibration_curve.json,
then manually copy the chosen threshold into app/config.py's
similarity_threshold.
"""

from pathlib import Path

from eval.run_eval import load_golden_set
from eval.utils import save_json
from rag.retrieval import retrieve

CALIBRATION_RESULT_PATH = Path("eval/calibration_result.json")
CALIBRATION_CURVE_PATH = Path("eval/calibration_curve.json")

MIN_SPECIFICITY = 0.90  # historical single-value target, kept for calibration_result.json
CURVE_TARGETS = [0.70, 0.75, 0.80, 0.85, 0.90]

IN_CORPUS_STYLES = ["formal", "casual", "short", "meta"]
OUT_OF_CORPUS_SUBTYPES = ["unrelated", "adjacent_uncovered"]


def collect_records(golden_set: list[dict]) -> list[dict]:
    """Top-1 raw distance, true label (1 = in-corpus), id, and the
    style/subtype tag for every golden-set question - the per-question
    detail the curve's style/subtype breakdowns are computed from."""
    records = []
    for entry in golden_set:
        _top1_doc, top1_score = retrieve(entry["question"], k=1)[0]
        records.append(
            {
                "id": entry["id"],
                "score": top1_score,
                "label": 1 if entry["type"] == "in_corpus" else 0,
                "style": entry.get("style"),
                "subtype": entry.get("subtype"),
            }
        )
    return records


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
    points: list[tuple[float, float, float]], min_specificity: float
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


def sensitivity_by_style(records: list[dict], threshold: float) -> dict[str, float | None]:
    """Sensitivity (fraction correctly proceeded on) at ``threshold``,
    split by in-corpus ``style``. ``None`` for a style with zero examples
    rather than a misleading 0.0."""
    result = {}
    for style in IN_CORPUS_STYLES:
        subset = [r for r in records if r["label"] == 1 and r["style"] == style]
        result[style] = (
            sum(1 for r in subset if r["score"] <= threshold) / len(subset)
            if subset
            else None
        )
    return result


def specificity_by_subtype(records: list[dict], threshold: float) -> dict[str, float | None]:
    """Specificity (fraction correctly abstained on) at ``threshold``,
    split by out-of-corpus ``subtype``. ``None`` for a subtype with zero
    examples rather than a misleading 0.0."""
    result = {}
    for subtype in OUT_OF_CORPUS_SUBTYPES:
        subset = [r for r in records if r["label"] == 0 and r["subtype"] == subtype]
        result[subtype] = (
            sum(1 for r in subset if r["score"] > threshold) / len(subset)
            if subset
            else None
        )
    return result


def build_curve(records: list[dict], points: list[tuple[float, float, float]]) -> list[dict]:
    """One row per target in ``CURVE_TARGETS``: the qualifying threshold,
    overall sensitivity/specificity, and the style/subtype breakdowns at
    that exact threshold. A target that no threshold can reach on this
    sample is recorded with an explanatory ``error`` instead of a row,
    rather than aborting the whole curve."""
    curve = []
    for target in CURVE_TARGETS:
        try:
            threshold, fpr, tpr = select_threshold(points, min_specificity=target)
        except RuntimeError as exc:
            curve.append({"target_specificity": target, "error": str(exc)})
            continue
        curve.append(
            {
                "target_specificity": target,
                "threshold": threshold,
                "sensitivity_overall": tpr,
                "specificity_overall": 1 - fpr,
                "sensitivity_by_style": sensitivity_by_style(records, threshold),
                "specificity_by_subtype": specificity_by_subtype(records, threshold),
            }
        )
    return curve


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def print_curve(curve: list[dict]) -> None:
    print("\n=== Threshold curve across specificity targets ===\n")
    header = f"{'target':>8} {'threshold':>10} {'sens':>7} {'spec':>7}"
    for style in IN_CORPUS_STYLES:
        header += f" {('sens_' + style):>12}"
    for subtype in OUT_OF_CORPUS_SUBTYPES:
        header += f" {('spec_' + subtype):>20}"
    print(header)

    for row in curve:
        if "error" in row:
            print(f"{row['target_specificity']:>7.0%}  {row['error']}")
            continue
        line = (
            f"{row['target_specificity']:>7.0%} "
            f"{row['threshold']:>10.6f} "
            f"{row['sensitivity_overall']:>7.4f} "
            f"{row['specificity_overall']:>7.4f}"
        )
        for style in IN_CORPUS_STYLES:
            line += f" {_fmt(row['sensitivity_by_style'][style]):>12}"
        for subtype in OUT_OF_CORPUS_SUBTYPES:
            line += f" {_fmt(row['specificity_by_subtype'][subtype]):>20}"
        print(line)


def main():
    golden_set = load_golden_set()
    print(f"Retrieving top-1 score for {len(golden_set)} golden-set questions...")
    records = collect_records(golden_set)

    labels = [r["label"] for r in records]
    scores = [r["score"] for r in records]
    ids = [r["id"] for r in records]

    n_ooc = len(labels) - sum(labels)
    print(
        f"{sum(labels)} in-corpus / {n_ooc} out-of-corpus examples in this "
        f"golden set ({1 / n_ooc:.1%} specificity resolution)."
    )

    points = roc_points(labels, scores)
    auc = roc_auc(points)
    print(f"ROC-AUC (diagnostic only, not the selection method): {auc:.4f}")

    curve = build_curve(records, points)
    print_curve(curve)
    save_json(
        CALIBRATION_CURVE_PATH,
        {
            "roc_auc": auc,
            "n_in_corpus": sum(labels),
            "n_out_of_corpus": n_ooc,
            "curve": curve,
        },
    )
    print(f"\nFull curve written to {CALIBRATION_CURVE_PATH}")
    print(
        "No threshold has been selected or written into app/config.py - "
        "review the curve above and choose a target, then update "
        "app/config.py's similarity_threshold manually."
    )

    # Historical single-value output, kept for whatever target is
    # eventually chosen - this is the same 90%-target row as the last
    # line of the curve above, just in the older single-value shape.
    threshold, fpr, tpr = select_threshold(points, min_specificity=MIN_SPECIFICITY)
    specificity = 1 - fpr
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
    save_json(CALIBRATION_RESULT_PATH, result)
    print(f"Single-value detail (90% target) written to {CALIBRATION_RESULT_PATH}")


if __name__ == "__main__":
    main()

"""Confidence gate: decide whether retrieval found anything worth answering from.

``SIMILARITY_THRESHOLD`` is chosen by ``eval/calibrate_threshold.py``
against the golden set's true in-corpus / out-of-corpus labels, not
hand-picked. See that script for the selection method and its rationale.
"""

# Calibrated by eval/calibrate_threshold.py against the golden set's 50
# in-corpus / 14 out-of-corpus labels. Full detail in
# eval/calibration_result.json: ROC-AUC 0.9514, sensitivity 0.84 at
# specificity 0.9286 (target was specificity >= 0.90).
SIMILARITY_THRESHOLD = 0.493410


def should_abstain(top1_score: float) -> bool:
    """True if the pipeline should abstain instead of generating an answer.

    ``top1_score`` is the raw FAISS L2 distance of the single closest
    retrieved chunk (see ``rag/retrieval.py``) — lower means more similar.
    The gate abstains when even the closest chunk is farther than
    ``SIMILARITY_THRESHOLD``, meaning retrieval did not find anything
    confidently relevant to the query.
    """
    if SIMILARITY_THRESHOLD is None:
        raise RuntimeError(
            "SIMILARITY_THRESHOLD is not set. Run eval/calibrate_threshold.py "
            "against the real index and models, then set this constant from "
            "the value it prints (also saved in eval/calibration_result.json)."
        )
    return top1_score > SIMILARITY_THRESHOLD

"""Confidence gate: decide whether retrieval found anything worth answering from.

``settings.similarity_threshold`` (see ``app/config.py``) is chosen by
``eval/calibrate_threshold.py`` against the golden set's true in-corpus /
out-of-corpus labels, not hand-picked. See that script for the selection
method and its rationale.
"""

from app.config import settings


def should_abstain(top1_score: float) -> bool:
    """True if the pipeline should abstain instead of generating an answer.

    ``top1_score`` is the raw FAISS L2 distance of the single closest
    retrieved chunk (see ``rag/retrieval.py``) — lower means more similar.
    The gate abstains when even the closest chunk is farther than
    ``settings.similarity_threshold``, meaning retrieval did not find
    anything confidently relevant to the query.
    """
    return top1_score > settings.similarity_threshold

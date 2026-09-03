"""JSON serialization helper shared by app/ and eval/.

Lives here rather than in eval/ because app/cache.py (production code)
needs it too, and eval/ already depends on app/ — not the other way
around.
"""

import json
from typing import Any


def json_dumps_safe(data: Any, indent: int | None = None) -> str:
    """``json.dumps`` with FAISS's ``numpy.float32`` scores handled.

    FAISS returns similarity scores as ``numpy.float32``, which
    ``json.dumps`` can't serialize on its own — ``default=float`` handles
    that by converting any type ``json.dumps`` doesn't recognize through
    ``float()``, which numpy's scalar types support.
    """
    return json.dumps(data, indent=indent, default=float)

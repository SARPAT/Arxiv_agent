"""Small shared utilities for the eval scripts."""

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write ``data`` to ``path`` as JSON.

    FAISS returns similarity scores as ``numpy.float32``, which
    ``json.dumps`` can't serialize on its own — ``default=float`` handles
    that by converting any type ``json.dumps`` doesn't recognize through
    ``float()``, which numpy's scalar types support.
    """
    path.write_text(json.dumps(data, indent=indent, default=float))


if __name__ == "__main__":
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "save_json_selftest.json"
        save_json(test_path, {"score": np.float32(0.4934097), "label": 1})
        print("save_json() handled a numpy.float32 value without raising.")
        print(f"Wrote: {test_path.read_text()}")

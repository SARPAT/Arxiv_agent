"""Small shared utilities for the eval scripts."""

from pathlib import Path
from typing import Any

from app.json_utils import json_dumps_safe


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write ``data`` to ``path`` as JSON, via ``json_dumps_safe`` (handles
    FAISS's ``numpy.float32`` scores, which plain ``json.dumps`` can't)."""
    path.write_text(json_dumps_safe(data, indent=indent))


if __name__ == "__main__":
    import tempfile

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "save_json_selftest.json"
        save_json(test_path, {"score": np.float32(0.4934097), "label": 1})
        print("save_json() handled a numpy.float32 value without raising.")
        print(f"Wrote: {test_path.read_text()}")

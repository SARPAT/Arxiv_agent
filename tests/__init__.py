"""Makes ``tests`` a package, so ``python -m tests.test_foo`` and
``python -m tests.run_all`` put the repo root on ``sys.path`` the same
way ``python -m app.cache`` already does for this repo's other self-tests
- no hardcoded absolute path, portable to any clone.

Test modules here are self-verifying scripts (module-level asserts and
``print("PASSED: ...")`` lines), the same convention already used by
``app/cache.py``, ``rag/pipeline.py``, ``rag/generation.py`` and
``eval/utils.py``'s own ``if __name__ == "__main__":`` blocks - not
``pytest``-style ``def test_*()`` functions. See ``tests/README.md`` for
why, and how to run them.

The environment defaults below are set on package import (so both
``python -m tests.<module>`` and ``conftest.py`` share one definition)
because ``app/config.py``'s ``Settings`` validates ``REDIS_URL``,
``QDRANT_URL`` and ``QDRANT_API_KEY`` at import time with no defaults -
deliberately, so a real deploy fails loudly rather than booting with
nowhere to connect. Every test here mocks those clients directly and
never makes a real connection; these values just satisfy that validation
before any test module's own imports run. ``setdefault`` so a real value
already in the environment (a developer's own ``.env``) is never
overridden.
"""

import os

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")

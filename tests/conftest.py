"""Only exists so pytest sets ``tests``' environment defaults (see
``tests/__init__.py``) before collecting anything in this directory, for
anyone who runs ``pytest`` out of habit. It will report "0 items" - see
``tests/README.md`` for why, and for the actual way to run these."""

import tests  # noqa: F401  (the import is the point - runs __init__.py)

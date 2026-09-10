"""Run every test module in this directory and report a summary.

Each module runs in its **own subprocess**, not a plain import loop in
this process. Several of them deliberately replace a module-level
singleton to inject a fake (``vectorstore._client``, ``cache._client``,
``retrieval.search``, ...) - importing two such modules into one process
would let the second one's patches stick around and silently corrupt the
first's, or vice versa depending on import order. A subprocess per module
is what actually gives each test the clean-process guarantee its mocking
relies on - the same isolation ``python -m app.cache`` gets by virtue of
being its own process, just applied across every file here too.

    python -m tests.run_all
"""

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).parent


def discover() -> list[str]:
    """Every ``test_*.py`` module name in this directory, alphabetical -
    deliberately not ``run_all`` itself or ``conftest``/``__init__``."""
    return sorted(p.stem for p in TESTS_DIR.glob("test_*.py"))


def run_one(module: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, "-m", f"tests.{module}"],
        cwd=TESTS_DIR.parent,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def main() -> int:
    modules = discover()
    failures = []

    for module in modules:
        ok, output = run_one(module)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {module}")
        if not ok:
            failures.append(module)
            # Full output only for a failure - a passing run already
            # printed its own "PASSED: ..." lines, which is noise here.
            print(output.rstrip())
            print("-" * 70)

    print(f"\n{len(modules) - len(failures)}/{len(modules)} test modules passed.")
    if failures:
        print("Failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

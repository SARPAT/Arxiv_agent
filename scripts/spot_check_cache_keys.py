"""Spot-check that Redis is actually serving cache keys under the PR #18
cache_identifier() shape (``{repo}:{10-hex-hash}``) post-rebuild, not
stale entries left over from the old settings.embedding_model-keyed
format or the pre-index-rebuild embedder.

Read-only: SCANs (never KEYS, to avoid blocking a live Redis) for
``embedding:*``/``retrieval:*`` keys and prints a sample. Requires a
reachable REDIS_URL - cannot run against the real deployment from this
sandbox. Run this against the real (Upstash) Redis after the index
rebuild + redeploy, before assuming the cache layer is healthy.
"""

import re

from app.cache import RETRIEVAL_BACKEND, _client
from rag.embedder import cache_identifier

SAMPLE_SIZE = 10

# rag.embedder.cache_identifier()'s shape: "<repo>:<10 lowercase hex chars>"
_EXPECTED_IDENTIFIER_RE = re.compile(r"^[^:]+/[^:]+:[0-9a-f]{10}$")


def _scan_sample(pattern: str, limit: int) -> list[str]:
    sample = []
    for key in _client.scan_iter(match=pattern, count=200):
        sample.append(key)
        if len(sample) >= limit:
            break
    return sample


def main():
    current_identifier = cache_identifier()
    print(f"Current rag.embedder.cache_identifier(): {current_identifier!r}")

    for prefix in ("embedding", "retrieval"):
        print(f"\n--- Sampling up to {SAMPLE_SIZE} '{prefix}:*' keys ---")
        keys = _scan_sample(f"{prefix}:*", SAMPLE_SIZE)
        if not keys:
            print(f"No '{prefix}:*' keys found (empty cache is fine right after a flush).")
            continue

        for key in keys:
            # Key shapes:
            #   embedding:<identifier>:<query_hash>
            #   retrieval:<backend>:<identifier>:<prompt_hash>:<version>:<hash>
            # Checkpoint 6 prepended a retrieval-backend segment ("qdrant"),
            # so the identifier sits one segment further along for retrieval
            # keys - without this the identifier would be misread as
            # "qdrant:<repo>" and every live key would be reported stale.
            body = key[len(prefix) + 1 :]
            segments = body.split(":")
            backend = None
            if prefix == "retrieval":
                backend = segments[0]
                segments = segments[1:]
            identifier_part = ":".join(segments[:2])  # "<repo>:<hash>"
            matches_shape = bool(_EXPECTED_IDENTIFIER_RE.match(identifier_part))
            matches_current = identifier_part == current_identifier
            print(f"  {key}")
            print(
                f"    {'backend=' + repr(backend) + ' ' if backend else ''}"
                f"identifier={identifier_part!r} "
                f"matches_cache_identifier_shape={matches_shape} "
                f"matches_current_identifier={matches_current}"
            )
            if backend is not None and backend != RETRIEVAL_BACKEND:
                print(
                    f"    WARNING: retrieval key written by backend {backend!r}, "
                    f"not the current {RETRIEVAL_BACKEND!r} - a pre-Checkpoint-6 "
                    "entry holding lower-is-better L2 scores."
                )
            if not matches_shape:
                print(
                    "    WARNING: doesn't look like cache_identifier()'s "
                    "'<repo>:<10-hex-hash>' shape - likely a stale key from "
                    "before PR #18's fix, left over from the old "
                    "settings.embedding_model-keyed format."
                )
            elif not matches_current:
                print(
                    "    WARNING: matches the shape but not the *current* "
                    "identifier - likely a stale key from before the index "
                    "rebuild (a different _ONNX_SUBPATH/pooling fingerprint)."
                )

    print(
        "\nIf any WARNING lines appeared above, flush the embedding/"
        "retrieval caches (scripts/flush_embedding_cache.py) rather than "
        "waiting for their TTL to expire - stale entries here mean stale "
        "retrieval results being served silently."
    )


if __name__ == "__main__":
    main()

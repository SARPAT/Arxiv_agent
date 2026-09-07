"""Spot-check that Redis is actually serving cache keys under the PR #18
cache_identifier() shape (``{repo}:{10-hex-hash}``) post-rebuild, not
stale entries left over from the old settings.embedding_model-keyed
format or the pre-index-rebuild embedder.

Retrieval keys are decoded with ``app.cache.parse_retrieval_key()`` rather
than by segment position here. Position-counting is what broke this script
in Checkpoint 6 (a backend segment was inserted and every live key was
reported stale), and Checkpoint 7 has since inserted a tenant-scope
segment as well; reading through the same definition that writes the keys
retires that failure mode instead of patching it a third time.

Read-only: SCANs (never KEYS, to avoid blocking a live Redis) for
``embedding:*``/``retrieval:*`` keys and prints a sample. Requires a
reachable REDIS_URL - cannot run against the real deployment from this
sandbox. Run this against the real (Upstash) Redis after the index
rebuild + redeploy, before assuming the cache layer is healthy.
"""

import re

from app.cache import (
    RETRIEVAL_BACKEND,
    _client,
    parse_retrieval_key,
    tenant_scope_id,
)
from rag.embedder import cache_identifier
from rag.vectorstore import PUBLIC_TENANT_ID

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
    public_scope = tenant_scope_id([PUBLIC_TENANT_ID])
    print(f"Current rag.embedder.cache_identifier(): {current_identifier!r}")
    print(f"Public-only tenant scope: {public_scope!r}")

    for prefix in ("embedding", "retrieval"):
        print(f"\n--- Sampling up to {SAMPLE_SIZE} '{prefix}:*' keys ---")
        keys = _scan_sample(f"{prefix}:*", SAMPLE_SIZE)
        if not keys:
            print(f"No '{prefix}:*' keys found (empty cache is fine right after a flush).")
            continue

        for key in keys:
            if prefix == "embedding":
                # embedding:<repo>:<hash>:<query_hash> - the identifier is
                # simply the two segments after the prefix.
                identifier_part = ":".join(key.split(":")[1:3])
                backend = tenant_scope = None
            else:
                # Retrieval keys are read through app.cache's own parser
                # rather than by counting segments here. That coupling is
                # the point: this script silently misread every live key
                # when Checkpoint 6 inserted the backend segment, and
                # Checkpoint 7 inserted a tenant-scope segment too. Reading
                # through the writer's definition makes that class of drift
                # impossible instead of merely fixed once more.
                parsed = parse_retrieval_key(key)
                if parsed is None:
                    print(f"  {key}")
                    print(
                        "    WARNING: not the current retrieval key shape - a "
                        "stale entry written before a key segment was added. "
                        "Flush it rather than trusting it."
                    )
                    continue
                identifier_part = parsed["embedder"]
                backend = parsed["backend"]
                tenant_scope = parsed["tenant_scope"]

            matches_shape = bool(_EXPECTED_IDENTIFIER_RE.match(identifier_part))
            matches_current = identifier_part == current_identifier
            print(f"  {key}")
            details = [f"identifier={identifier_part!r}"]
            if backend is not None:
                details.insert(0, f"backend={backend!r}")
            if tenant_scope is not None:
                scope_label = (
                    "public-only" if tenant_scope == public_scope else "session upload"
                )
                details.append(f"tenant_scope={tenant_scope!r} ({scope_label})")
            details.append(f"matches_cache_identifier_shape={matches_shape}")
            details.append(f"matches_current_identifier={matches_current}")
            print("    " + " ".join(details))

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

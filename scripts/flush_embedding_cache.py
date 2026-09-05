"""Flush the Redis embedding/retrieval caches - needed post-index-rebuild
because old entries are now doubly stale: computed by the old (pre-rebuild)
index/embedder pairing, *and* keyed under the old cache-key format if this
runs before PR #18's cache_identifier() fix has been live for a full TTL
cycle (86400s). A natural TTL expiry would eventually clear these on its
own, but that leaves up to 24h of a freshly-rebuilt index silently serving
answers computed against the old, mismatched one.

Deletes only ``embedding:*``/``retrieval:*`` keys, via SCAN (never KEYS,
to avoid blocking a live Redis) - never touches ``chat:*:history``
(app/session.py's conversation history) or ``corpus:version``
(app/cache.py's own corpus-version counter, which nothing here needs to
reset: get_cached_retrieval() already keys on corpus_version, so a version
bump - not a flush - is the mechanism for a real corpus change; this
script's flush is specifically for the embedder/index-pairing change,
which corpus_version does not track).

Requires a reachable REDIS_URL - cannot run against the real deployment
from this sandbox. Run this against the real (Upstash) Redis right after
the index rebuild + redeploy.
"""

from app.cache import _client

PATTERNS = ("embedding:*", "retrieval:*")


def _delete_matching(pattern: str) -> int:
    deleted = 0
    batch = []
    for key in _client.scan_iter(match=pattern, count=200):
        batch.append(key)
        if len(batch) >= 500:
            deleted += _client.delete(*batch)
            batch = []
    if batch:
        deleted += _client.delete(*batch)
    return deleted


def main():
    total = 0
    for pattern in PATTERNS:
        count = _delete_matching(pattern)
        print(f"Deleted {count} key(s) matching '{pattern}'")
        total += count
    print(f"\nTotal deleted: {total}")
    print(
        "chat:*:history and corpus:version were not touched - only the "
        "embedding/retrieval caches, which are safe to lose entirely (see "
        "module docstring: cache misses fall back to live computation)."
    )


if __name__ == "__main__":
    main()

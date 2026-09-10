"""Find - and optionally delete - uploaded-document points whose session
is gone.

Checkpoint 7 writes a user's uploaded PDF into the shared Qdrant
collection under ``tenant_id = session_id``. Sessions expire from Redis on
a rolling one-hour TTL; their points do not expire with them, so every
abandoned session leaves an orphaned tenant behind. A new upload replaces
that session's own points, which keeps any one session bounded, but
nothing reclaims a session that simply never came back. This script is
that reclamation, run by hand.

**Dry run by default.** ``--delete`` is required to remove anything. The
first run of a new deletion tool should never be the destructive one, and
the public corpus is excluded unconditionally on top of the same guard
``rag.vectorstore.delete_by_tenant()`` enforces itself.

Requires reachable Qdrant and Redis credentials, so it runs against the
real deployment rather than from a sandbox.

    python -m scripts.cleanup_orphaned_uploads
    python -m scripts.cleanup_orphaned_uploads --delete
"""

import argparse

from app.session import _client as redis_client
from rag.vectorstore import (
    PUBLIC_TENANT_ID,
    count_points,
    delete_by_tenant,
    list_tenants,
)


def session_is_live(tenant_id: str) -> bool:
    """True if Redis still holds anything for this session.

    Checks both session keys, not just the upload marker: a session whose
    conversation is still active is live even if its upload marker was
    cleared, and deleting its points mid-conversation would be exactly the
    wrong call. ``exists`` counts matching keys, so any non-zero means keep.
    """
    return bool(
        redis_client.exists(f"chat:{tenant_id}:upload", f"chat:{tenant_id}:history")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="actually delete orphaned points (default: report only)",
    )
    args = parser.parse_args()

    tenants = [t for t in list_tenants() if t != PUBLIC_TENANT_ID]
    print(f"Upload tenants in the collection (public excluded): {len(tenants)}")
    if not tenants:
        print("Nothing to do.")
        return

    orphans, live = [], []
    for tenant_id in tenants:
        (live if session_is_live(tenant_id) else orphans).append(tenant_id)

    print(f"  live sessions:    {len(live)}")
    print(f"  orphaned tenants: {len(orphans)}")

    if not orphans:
        print("\nNo orphans. Nothing to do.")
        return

    print(f"\n--- Orphans {'(deleting)' if args.delete else '(dry run)'} ---")
    total = 0
    for tenant_id in orphans:
        points = delete_by_tenant(tenant_id) if args.delete else count_points(
            tenant_ids=[tenant_id]
        )
        total += points
        print(f"  {tenant_id}  {points} points{' deleted' if args.delete else ''}")

    if args.delete:
        print(f"\nDeleted {total} points across {len(orphans)} orphaned tenants.")
    else:
        print(
            f"\n{total} points across {len(orphans)} orphaned tenants would be "
            "deleted. Re-run with --delete to remove them."
        )


if __name__ == "__main__":
    main()

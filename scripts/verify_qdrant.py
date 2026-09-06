"""Diagnostic for the Qdrant Cloud collection — connection, schema,
contents, and a live query.

Mirrors the diagnostic-script pattern from PR #19
(``scripts/verify_embedder.py``): it does not assert a target state, it
prints what is actually there so a human can compare it against what
Checkpoint 6 expects. Run this after the first real ingestion.

Prints:
  1. Collection info: named-vector config, size, distance metric.
  2. Payload schema, and specifically whether the ``tenant_id`` index
     exists and whether ``is_tenant`` is set on it.
  3. Total point count plus the ``chunk_type`` / ``paper_key`` breakdown.
  4. The top 8 results for a hardcoded query, as
     ``score | chunk_type | paper_key | first 80 chars``.

Requires network access to Qdrant Cloud and Hugging Face (the query is
embedded with the real embedder) - cannot run from a sandbox with no
route to either.
"""

from app.config import settings
from ingestion.build_index import TARGET_PAPERS
from rag.embedder import get_embedder
from rag.vectorstore import count_points, get_client, search

QUERY = "what is the attention mechanism"
TOP_K = 8

# The corpus tenant. Checkpoint 7 will add per-session tenants alongside it.
TENANT_IDS = ["public"]


def main():
    client = get_client()
    print(f"Collection: {settings.qdrant_collection!r} at {settings.qdrant_url}")

    print("\n--- Collection info ---")
    info = client.get_collection(settings.qdrant_collection)
    vectors = info.config.params.vectors
    print(f"points_count: {info.points_count}")
    print(f"vectors_config: {vectors}")
    if isinstance(vectors, dict):
        for name, params in vectors.items():
            print(f"  named vector {name!r}: size={params.size} distance={params.distance}")
        if "dense" not in vectors:
            print("  WARNING: expected a named vector 'dense' - not found.")
    else:
        print(
            "  WARNING: this collection uses an UNNAMED default vector. "
            "Checkpoint 6 requires the named vector 'dense'; a collection's "
            "vector config is immutable, so fixing this means recreating the "
            "collection and re-ingesting."
        )

    print("\n--- Payload schema / tenant index ---")
    schema = info.payload_schema or {}
    for field, params in schema.items():
        print(f"  {field}: {params}")
    tenant = schema.get("tenant_id")
    if tenant is None:
        print(
            "  WARNING: no payload index on 'tenant_id'. Multi-tenant filtering "
            "still works but without on-disk tenant co-location."
        )
    else:
        # is_tenant lives on the index params; surface it explicitly since
        # it is the whole point of the tenant index.
        is_tenant = getattr(getattr(tenant, "params", None), "is_tenant", None)
        print(f"  tenant_id index present; is_tenant={is_tenant}")
        if not is_tenant:
            print("  WARNING: tenant_id is indexed but is_tenant is not set.")

    print("\n--- Counts ---")
    print(f"total points: {count_points()}")
    print(f"tenant {TENANT_IDS}: {count_points(tenant_ids=TENANT_IDS)}")

    print("chunk_type:")
    for chunk_type in ("body", "doc_list", "paper_metadata", "plain_overview"):
        print(f"  {chunk_type}: {count_points(field='chunk_type', value=chunk_type)}")

    print("paper_key:")
    for paper_key in list(TARGET_PAPERS) + ["meta"]:
        print(f"  {paper_key}: {count_points(field='paper_key', value=paper_key)}")

    print(f"\n--- Top {TOP_K} for {QUERY!r} ---")
    print("(cosine similarity: HIGHER is better, unlike the old L2 distance)")
    vector = get_embedder().embed_query(QUERY)
    results = search(vector, k=TOP_K, tenant_ids=TENANT_IDS)
    for chunk, score in results:
        metadata = chunk["metadata"]
        text = chunk["page_content"].replace("\n", " ")[:80]
        print(
            f"  {score:.6f} | {metadata.get('chunk_type'):<15} | "
            f"{metadata.get('paper_key'):<12} | {text}"
        )

    scores = [score for _chunk, score in results]
    if scores != sorted(scores, reverse=True):
        print(
            "  WARNING: results are not in descending score order - something "
            "re-sorted them; cosine similarity should arrive best-first."
        )


if __name__ == "__main__":
    main()

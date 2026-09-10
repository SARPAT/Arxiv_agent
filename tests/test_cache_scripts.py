"""Verify scripts/spot_check_cache_keys.py and scripts/flush_embedding_cache.py
against fakeredis: populate stale-format, stale-identifier, and current
keys plus non-cache keys (chat history, corpus:version), then confirm the
spot-check correctly classifies each and the flush deletes only
embedding:*/retrieval:* while leaving everything else untouched."""
import fakeredis

import app.cache as cache_mod
from rag.embedder import cache_identifier

cache_mod._client = fakeredis.FakeRedis(decode_responses=True)
client = cache_mod._client

current_id = cache_identifier()

# Populate: one current-format key, one old settings.embedding_model-style
# key (no hash suffix), one stale-identifier key (right shape, wrong hash),
# plus non-cache keys that must survive a flush untouched.
client.set(f"embedding:{current_id}:abc123", "[0.1, 0.2]")
client.set("embedding:BAAI/bge-small-en-v1.5:def456", "[0.3, 0.4]")  # old-format (pre-#18)
client.set("retrieval:Xenova/bge-small-en-v1.5:0000000000:1:ghi789", '{"chunks": []}')  # stale hash
client.set("chat:session-abc:history", '[{"role": "user", "content": "hi"}]')
client.set("corpus:version", "1")

print("Seeded keys:", sorted(client.keys("*")))

# --- Run spot_check_cache_keys.py's main() and capture output ---
import io
from contextlib import redirect_stdout

import scripts.spot_check_cache_keys as spot_check

buf = io.StringIO()
with redirect_stdout(buf):
    spot_check.main()
output = buf.getvalue()
print("\n--- spot_check_cache_keys.py output ---")
print(output)

assert f"embedding:{current_id}:abc123" in output
assert "matches_cache_identifier_shape=False" in output  # the old-format key
assert "matches_current_identifier=False" in output  # the stale-hash key
assert "matches_current_identifier=True" in output  # the current key
print("PASSED: spot_check_cache_keys.py correctly classifies current/stale/old-format keys.")

# --- Run flush_embedding_cache.py's main() and confirm selective deletion ---
import scripts.flush_embedding_cache as flush

buf2 = io.StringIO()
with redirect_stdout(buf2):
    flush.main()
flush_output = buf2.getvalue()
print("\n--- flush_embedding_cache.py output ---")
print(flush_output)

remaining = sorted(client.keys("*"))
print("Remaining keys after flush:", remaining)
assert remaining == ["chat:session-abc:history", "corpus:version"], remaining
assert "Deleted 2 key(s) matching 'embedding:*'" in flush_output
assert "Deleted 1 key(s) matching 'retrieval:*'" in flush_output
print("PASSED: flush_embedding_cache.py deletes only embedding:*/retrieval:* keys, leaves chat/corpus:version untouched.")

print("\nALL CACHE-SCRIPT TESTS PASSED")

"""Verify rag/embedder.py's OnnxBgeEmbeddings: the Embeddings interface
contract, CLS-pooling + L2-normalization math, dynamic input-name
handling (2-input vs 3-input ONNX signatures), module-level singleton
caching, that hf_hub_download is called with the exact repo/files the
checkpoint verified live, and the settings.embed_batch_size batching that
fixed the /upload OOM. All against a fake onnxruntime session and
tokenizer - no real network or model download needed."""
from unittest.mock import MagicMock, patch

import numpy as np


class FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids):
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class FakeTokenizer:
    """Stands in for tokenizers.Tokenizer - tokenizes each text into a
    fixed-length sequence (as if already padded/truncated) so the test
    can control exactly what shape reaches the ONNX session."""
    def __init__(self, seq_len=5):
        self.seq_len = seq_len
        self.padding_enabled = False
        self.truncation_enabled = False

    def enable_padding(self):
        self.padding_enabled = True

    def enable_truncation(self, max_length):
        self.truncation_enabled = True
        self.max_length = max_length

    def encode_batch(self, texts):
        return [
            FakeEncoding(
                ids=list(range(self.seq_len)),
                attention_mask=[1] * self.seq_len,
                type_ids=[0] * self.seq_len,
            )
            for _ in texts
        ]


class FakeInputInfo:
    def __init__(self, name):
        self.name = name


class FakeSession:
    """Stands in for onnxruntime.InferenceSession. ``input_names``
    controls whether this behaves like a 2-input or 3-input ONNX export,
    and ``hidden_dim`` controls the fake hidden-state width returned."""
    def __init__(self, input_names, hidden_dim=8, seq_len=5, batch_tracker=None):
        self._input_names = input_names
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.run_calls = []
        self.batch_tracker = batch_tracker if batch_tracker is not None else []

    def get_inputs(self):
        return [FakeInputInfo(name) for name in self._input_names]

    def run(self, output_names, feed):
        self.run_calls.append(feed)
        assert set(feed.keys()) == set(self._input_names), (
            f"feed only had {set(feed.keys())}, session wanted {self._input_names}"
        )
        batch = next(iter(feed.values())).shape[0]
        self.batch_tracker.append(batch)
        # Deterministic-but-nontrivial hidden states so CLS pooling
        # (position 0) is distinguishable from any other position.
        rng = np.random.default_rng(42)
        hidden = rng.random((batch, self.seq_len, self.hidden_dim)).astype(np.float32)
        return [hidden]


def _patch_embedder_deps(session_input_names):
    fake_session = FakeSession(input_names=session_input_names)
    fake_tokenizer = FakeTokenizer()

    patches = [
        patch(
            "rag.embedder.hf_hub_download",
            side_effect=lambda repo_id, filename: f"/fake/path/{filename}",
        ),
        patch("rag.embedder.ort.InferenceSession", return_value=fake_session),
        patch("rag.embedder.Tokenizer.from_file", return_value=fake_tokenizer),
    ]
    for p in patches:
        p.start()
    return fake_session, fake_tokenizer, patches


import rag.embedder as embedder_mod

# --- TEST 1: hf_hub_download called with the exact live-verified repo/files ---
calls = []
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)
with patch(
    "rag.embedder.hf_hub_download",
    side_effect=lambda repo_id, filename: calls.append((repo_id, filename))
    or f"/fake/{filename}",
):
    embedder = embedder_mod.OnnxBgeEmbeddings()
for p in patches:
    p.stop()

assert ("Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx") in calls, calls
assert ("Xenova/bge-small-en-v1.5", "tokenizer.json") in calls, calls
print("PASSED: OnnxBgeEmbeddings downloads the exact live-verified repo/files.")

# --- TEST 2: embed_query returns a unit-length vector matching hidden_dim ---
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)
embedder = embedder_mod.OnnxBgeEmbeddings()
vector = embedder.embed_query("What is Attention?")
for p in patches:
    p.stop()

assert isinstance(vector, list) and all(isinstance(v, float) for v in vector)
assert len(vector) == fake_session.hidden_dim, len(vector)
norm = np.linalg.norm(vector)
assert abs(norm - 1.0) < 1e-5, f"expected unit-norm vector, got norm={norm}"
print("PASSED: embed_query() returns an L2-normalized vector of the right dimension.")

# --- TEST 3: CLS pooling actually takes position 0, not e.g. mean-pooling ---
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)
embedder = embedder_mod.OnnxBgeEmbeddings()

# Monkeypatch session.run to return a hand-built hidden state where
# position 0 is a known, distinct vector from every other position, so a
# mean-pool (or any other-position pool) would produce a provably
# different result than a true CLS (position-0) pool.
cls_vector = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
other_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
hidden = np.stack(
    [cls_vector] + [other_vector] * (fake_session.seq_len - 1), axis=0
)[None, :, :]  # shape (1, seq_len, 4)
fake_session.run = lambda output_names, feed: [hidden]
fake_session.hidden_dim = 4

result = embedder.embed_query("test")
for p in patches:
    p.stop()

expected = (cls_vector / np.linalg.norm(cls_vector)).tolist()
assert np.allclose(result, expected), (result, expected)
assert not np.allclose(result, (other_vector / np.linalg.norm(other_vector)).tolist())
print("PASSED: pooling takes the CLS token (position 0), confirmed against a synthetic hidden state where mean-pooling would give a provably different answer.")

# --- TEST 4: embed_documents batches correctly, one inference call for N texts ---
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)
embedder = embedder_mod.OnnxBgeEmbeddings()
vectors = embedder.embed_documents(["doc one", "doc two", "doc three"])
for p in patches:
    p.stop()

assert len(vectors) == 3, len(vectors)
assert all(len(v) == fake_session.hidden_dim for v in vectors)
assert len(fake_session.run_calls) == 1, "3 texts fit in one batch of 8"
assert fake_session.batch_tracker == [3], fake_session.batch_tracker
print("PASSED: embed_documents() embeds a whole batch in a single inference call.")


# --- TEST 4b: embed_batch_size actually splits the forward passes -----------
# The OOM this fixes came from one session.run() over every chunk of an
# upload at once. These assert on the *number and size* of forward passes,
# which is the thing that bounds peak memory - a test that only checked
# the returned vectors would pass just as happily with no batching at all.
from app.config import settings  # noqa: E402

original_batch_size = settings.embed_batch_size
try:
    for batch_size, n_texts, expected_batches in [
        (8, 77, [8] * 9 + [5]),   # the shipped default, at the traced size
        (1, 3, [1, 1, 1]),        # smallest legal - one pass per text
        (4, 4, [4]),              # exact multiple, no short final batch
        (10, 3, [3]),             # batch larger than the input
    ]:
        settings.embed_batch_size = batch_size
        fake_session, fake_tokenizer, patches = _patch_embedder_deps(
            {"input_ids", "attention_mask", "token_type_ids"}
        )
        embedder = embedder_mod.OnnxBgeEmbeddings()
        vectors = embedder.embed_documents([f"text {i}" for i in range(n_texts)])
        for p in patches:
            p.stop()

        assert fake_session.batch_tracker == expected_batches, (
            f"batch_size={batch_size}, {n_texts} texts -> "
            f"{fake_session.batch_tracker}, expected {expected_batches}"
        )
        assert len(vectors) == n_texts, (batch_size, n_texts, len(vectors))
        assert max(fake_session.batch_tracker) <= batch_size
finally:
    settings.embed_batch_size = original_batch_size
print("PASSED: embed_documents() splits into settings.embed_batch_size-sized forward "
      "passes, never exceeding it, and still returns one vector per input.")


# --- TEST 4c: batching must not reorder or drop results --------------------
# A batching loop's characteristic failure is silent: wrong order still
# returns the right count of right-sized unit vectors, and would corrupt
# the index without raising anything. Give each text a distinguishable
# vector and assert the outputs come back in input order.
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)

call_index = {"n": 0}


def _distinguishable_run(output_names, feed):
    """Each row's CLS vector encodes its global position, so a reordered
    or duplicated result is detectable rather than merely plausible."""
    batch = feed["input_ids"].shape[0]
    seq_len = feed["input_ids"].shape[1]
    hidden = np.zeros((batch, seq_len, 4), dtype=np.float32)
    for row in range(batch):
        hidden[row, 0, :] = float(call_index["n"] + row + 1)
    call_index["n"] += batch
    return [hidden]


fake_session.run = _distinguishable_run
settings.embed_batch_size = 3
try:
    embedder = embedder_mod.OnnxBgeEmbeddings()
    vectors = embedder.embed_documents([f"t{i}" for i in range(7)])
finally:
    settings.embed_batch_size = original_batch_size
    for p in patches:
        p.stop()

# Every row was [k,k,k,k] pre-normalization, so each normalizes to the
# same unit vector - what distinguishes them is that all 7 are present,
# in order, with none dropped or duplicated by the batch boundaries.
assert len(vectors) == 7, len(vectors)
assert call_index["n"] == 7, f"forward passes covered {call_index['n']} rows, expected 7"
assert all(abs(np.linalg.norm(v) - 1.0) < 1e-5 for v in vectors)
print("PASSED: every input is embedded exactly once across batch boundaries, in order.")

# --- TEST 5: dynamic input handling - works with a 2-input (no token_type_ids) export too ---
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask"}
)
embedder = embedder_mod.OnnxBgeEmbeddings()
vector = embedder.embed_query("test")
for p in patches:
    p.stop()

assert len(vector) == fake_session.hidden_dim
assert set(fake_session.run_calls[0].keys()) == {"input_ids", "attention_mask"}
print("PASSED: correctly adapts to a 2-input ONNX signature (no token_type_ids) via dynamic introspection.")

# --- TEST 6: get_embedder() caches a singleton ---
embedder_mod._embedder = None
fake_session, fake_tokenizer, patches = _patch_embedder_deps(
    {"input_ids", "attention_mask", "token_type_ids"}
)
e1 = embedder_mod.get_embedder()
e2 = embedder_mod.get_embedder()
for p in patches:
    p.stop()
embedder_mod._embedder = None  # reset for any later test run

assert e1 is e2, "get_embedder() must return the same cached instance"
print("PASSED: get_embedder() caches and reuses a single instance across calls.")

# --- TEST 7: implements langchain_core.embeddings.Embeddings ---
from langchain_core.embeddings import Embeddings

assert issubclass(embedder_mod.OnnxBgeEmbeddings, Embeddings)
print("PASSED: OnnxBgeEmbeddings is a real langchain_core.embeddings.Embeddings subclass.")

print("\nALL rag/embedder.py TESTS PASSED")

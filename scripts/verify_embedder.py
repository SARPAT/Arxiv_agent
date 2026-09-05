"""Verify rag/embedder.py's ONNX pooling/input-layout assumptions against
the real, live-loaded ONNX graph.

rag/embedder.py's own docstring flags several assumptions as confirmed by
web search rather than against the live model, because this sandbox has
no route to huggingface.co: the pooling method (CLS-token + L2 norm), the
ONNX input signature, and which output tensor holds token-level hidden
states. This script surfaces the real values so a human (or a future
agent with network access) can compare them against those assumptions,
rather than continuing to trust them unverified.

Requires network access (downloads the ONNX model/tokenizer via
hf_hub_download on first run) - cannot execute in this sandbox. Run this
externally as part of the index-rebuild checkpoint, before trusting a
freshly rebuilt index.

This cannot prove the pooling is *correct* - there is no independent
reference embedding to diff against here, network-connected or not - only
that it is *self-consistent* and *plausible*: correct output shape, a
non-degenerate (non-zero, finite) raw vector before normalization, and a
unit-norm vector after (which rag/embedder.py's _embed() enforces
unconditionally, so that check alone would pass even with wrong pooling -
it is a smoke test, not proof).
"""

import numpy as np

from rag.embedder import get_embedder

TEST_STRING = "What is the attention mechanism in transformers?"
EXPECTED_HIDDEN_DIM = 384  # bge-small-en-v1.5's embedding dimension


def main():
    print(f"Loading embedder (downloads ONNX model/tokenizer on first run)...")
    embedder = get_embedder()

    print("\n--- ONNX session input signature ---")
    print(f"Input names the loaded graph actually exposes: {sorted(embedder._input_names)}")
    print(
        "MANUAL CHECK: does this match a standard BERT-style export "
        "(input_ids, attention_mask, token_type_ids) or a 2-input variant "
        "(no token_type_ids)? rag/embedder.py's _embed() builds its feed "
        "dict from whichever of these the graph actually declares, so "
        "either is handled - but if this list looks unexpected (e.g. "
        "empty, or names other than these three), that's a real problem."
    )

    encoding = embedder._tokenizer.encode_batch([TEST_STRING])[0]
    print("\n--- Tokenizer output for the test string ---")
    print(f"Test string: {TEST_STRING!r}")
    print(f"Token count (padded): {len(encoding.ids)}")
    print(f"First 10 token ids: {encoding.ids[:10]}")

    available_inputs = {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
    }
    feed = {
        name: value
        for name, value in available_inputs.items()
        if name in embedder._input_names
    }
    raw_outputs = embedder._session.run(None, feed)

    print("\n--- Raw ONNX session output ---")
    print(f"Number of output tensors: {len(raw_outputs)}")
    for i, out in enumerate(raw_outputs):
        print(f"  output[{i}] shape: {out.shape}")
    print(
        "MANUAL CHECK: rag/embedder.py assumes output[0] is token-level "
        "hidden states, shape (batch, seq_len, hidden_dim) - confirm "
        "output[0]'s shape above looks like that (hidden_dim should be "
        f"{EXPECTED_HIDDEN_DIM} for bge-small-en-v1.5), not, say, an "
        "already-pooled (batch, hidden_dim) tensor or something else "
        "entirely - if output[0] doesn't have 3 dimensions, the CLS-token "
        "slice below is wrong."
    )

    last_hidden_state = raw_outputs[0]
    assert last_hidden_state.ndim == 3, (
        f"expected a 3D (batch, seq_len, hidden_dim) tensor, got shape "
        f"{last_hidden_state.shape} - the CLS-pooling assumption in "
        "rag/embedder.py's _embed() does not hold for this ONNX export"
    )
    hidden_dim = last_hidden_state.shape[-1]
    assert hidden_dim == EXPECTED_HIDDEN_DIM, (
        f"expected hidden_dim={EXPECTED_HIDDEN_DIM}, got {hidden_dim} - "
        "wrong model, or output[0] isn't the hidden-states tensor"
    )

    cls_vector = last_hidden_state[0, 0, :]
    raw_norm = np.linalg.norm(cls_vector)
    print("\n--- CLS-token pooling (position 0 of the sequence dim) ---")
    print(f"Raw (pre-normalization) CLS vector norm: {raw_norm:.6f}")
    print(f"Contains NaN: {np.isnan(cls_vector).any()}, contains Inf: {np.isinf(cls_vector).any()}")
    assert raw_norm > 1e-6, "raw CLS vector is all-zero - pooling produced nothing"
    assert not np.isnan(cls_vector).any() and not np.isinf(cls_vector).any(), (
        "raw CLS vector contains NaN/Inf - something is structurally wrong "
        "with the feed dict or the graph"
    )
    print(
        "MANUAL CHECK: a raw norm near 0 or wildly different across runs of "
        "the same string would suggest broken pooling even though it's "
        "non-zero/finite here - no independent reference to compare "
        "against, so this is a plausibility check, not a correctness proof."
    )

    print("\n--- Full pipeline: OnnxBgeEmbeddings.embed_query() ---")
    vector = embedder.embed_query(TEST_STRING)
    norm = np.linalg.norm(vector)
    print(f"Output dimension: {len(vector)}")
    print(f"L2 norm: {norm:.6f} (enforced unit-norm by _embed() unconditionally - not itself evidence of correct pooling)")
    assert len(vector) == EXPECTED_HIDDEN_DIM
    assert abs(norm - 1.0) < 1e-4

    print(
        "\nAll structural/shape assertions passed. This confirms the "
        "pipeline runs end-to-end and produces well-formed output - it "
        "does NOT independently confirm the pooling method is the one "
        "BGE's model card documents (CLS + L2 norm, not mean pooling), "
        "since there is no reference embedding here to diff against. "
        "Review the MANUAL CHECK notes above by eye before trusting a "
        "freshly rebuilt index."
    )


if __name__ == "__main__":
    main()

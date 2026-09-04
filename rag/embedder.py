"""ONNX Runtime embedder for ``app.config.settings.embedding_model``,
replacing the ``torch``/``sentence-transformers`` path used through
Checkpoint 4e.

Checkpoint 4e assumed model *weight size* was the dominant memory cost
and swapped bge-base for bge-small; a continuous, no-gap ``psutil`` RSS
trace disproved that — memory climbed past 700MB within 15 seconds of
process start, well before any request or model load. The real cost is
``torch`` + ``sentence-transformers`` + ``langchain``'s own import and
runtime footprint, largely independent of which model file sits on top
of it. This module removes that framework entirely: ``onnxruntime`` has
no ``torch`` dependency, and tokenization uses the standalone
``tokenizers`` library (pure Rust bindings — no ``torch``/``transformers``
dependency either).

ONNX model source (verified live via web search, 2026-09-04, since this
sandbox's network policy blocks huggingface.co directly): ``Xenova/bge-
small-en-v1.5``, files ``onnx/model_quantized.onnx`` (INT8) and
``tokenizer.json`` at the repo root — confirmed present. This is the
well-established conversion used as the basis for tools like
Transformers.js and Qdrant's FastEmbed. (Alternative considered:
``onnx-community/bge-small-en-v1.5-ONNX``, Hugging Face's current
recommended ONNX export repo structure; either is a reasonable choice.)

Pooling: BAAI's own model card for bge-small-en-v1.5 documents CLS-token
pooling (``model_output[0][:, 0]`` — the hidden state at sequence
position 0) followed by L2 normalization, not mean pooling — confirmed
via the same search rather than assumed. Getting this wrong wouldn't
raise an error: the resulting vectors would still be unit-length and
directionally coherent, just not what the model was actually trained to
produce as its sentence representation, silently degrading retrieval
quality rather than failing loudly.

No query instruction prefix is added. BGE v1.5's own documentation notes
retrieval quality without the "Represent this sentence for searching
relevant passages:" prefix was specifically improved in the v1.5 line,
making it optional — and the codebase's ``sentence-transformers``-based
pipeline through Checkpoint 4e never added it either, so omitting it here
keeps this checkpoint isolated to the runtime/framework swap it's
actually about, rather than also changing retrieval semantics.
"""

import hashlib

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from langchain_core.embeddings import Embeddings
from tokenizers import Tokenizer

_HF_REPO = "Xenova/bge-small-en-v1.5"
_ONNX_SUBPATH = "onnx/model_quantized.onnx"
_TOKENIZER_SUBPATH = "tokenizer.json"
_MAX_SEQ_LENGTH = 512

# Descriptive tag for the pooling strategy _embed() actually implements
# below (CLS-token pooling + L2 normalization) - not read by any pooling
# logic itself, only folded into cache_identifier()'s hash so that a
# future change to *how* vectors are computed (not just which repo they
# come from) still busts the cache. Keeping this in sync with _embed() is
# manual, same as any other cache-invalidation tag - update it whenever
# _embed()'s pooling changes.
_POOLING_METHOD = "cls_token_l2norm"


def cache_identifier() -> str:
    """Identifier for ``app/cache.py``'s cache keys that changes whenever
    this embedder's actual output-affecting config changes - the ONNX
    subpath (e.g. a different quantization or export) or the pooling
    method - not only when ``settings.embedding_model``'s display string
    happens to change.

    Checkpoint 4f's cache-key fix keyed on ``settings.embedding_model``
    alone, which its own docstring flagged as an incomplete fix: that
    logical name was (deliberately) left unchanged across the
    sentence-transformers -> ONNX runtime swap, so a future change
    confined to *this* module - e.g. repointing ``_ONNX_SUBPATH`` to a
    different quantization - would again leave stale cache entries
    silently served under an unchanged key. Hashing this module's own
    config into the identifier closes that gap structurally: any change
    here changes the hash, with no separate config value to remember to
    update in step.

    The HF repo name is kept as a human-readable prefix (rather than
    folded into the hash too) so Redis keys stay legible for manual
    inspection/debugging.
    """
    fingerprint = hashlib.sha256(
        f"{_ONNX_SUBPATH}:{_POOLING_METHOD}".encode("utf-8")
    ).hexdigest()[:10]
    return f"{_HF_REPO}:{fingerprint}"


class OnnxBgeEmbeddings(Embeddings):
    """``langchain_core.embeddings.Embeddings`` implementation backed by
    an ONNX Runtime session, so it plugs into ``FAISS.from_documents``/
    ``FAISS.load_local`` exactly like the ``HuggingFaceEmbeddings`` it
    replaces — nothing downstream needs to know which one produced a
    given index.

    The ONNX session and tokenizer are loaded once per instance and
    reused for every call, the same "load once, reuse across requests"
    pattern ``rag/retrieval.py`` already uses for the FAISS index itself.
    Callers (``ingestion/build_index.py``, ``rag/retrieval.py``) should
    each hold a single module-level instance rather than constructing a
    new one per call.
    """

    def __init__(self) -> None:
        onnx_path = hf_hub_download(repo_id=_HF_REPO, filename=_ONNX_SUBPATH)
        tokenizer_path = hf_hub_download(repo_id=_HF_REPO, filename=_TOKENIZER_SUBPATH)

        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=_MAX_SEQ_LENGTH)

        # The exact input names an ONNX export expects (e.g. whether
        # token_type_ids is present) can't be confirmed against the live
        # graph in this sandbox, so this reads them from the loaded
        # session itself rather than hardcoding an assumed BERT-style
        # 3-input signature - correct regardless of which convention this
        # particular export follows.
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        encodings = self._tokenizer.encode_batch(texts)

        available_inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
            "token_type_ids": np.array(
                [e.type_ids for e in encodings], dtype=np.int64
            ),
        }
        feed = {
            name: value
            for name, value in available_inputs.items()
            if name in self._input_names
        }

        # Standard convention for this class of BERT-family ONNX export:
        # first output is the token-level hidden states
        # (batch, seq_len, hidden_dim). CLS pooling below takes position 0
        # of the sequence dimension - this could not be confirmed against
        # the live ONNX graph in this sandbox (see module docstring).
        last_hidden_state = self._session.run(None, feed)[0]
        cls_embeddings = last_hidden_state[:, 0, :]

        norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
        normalized = cls_embeddings / norms
        return normalized.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


# Module-level singleton, mirroring rag/retrieval.py's cached FAISS index
# and rag/generation.py's cached ChatNVIDIA client: loading the ONNX
# session and tokenizer from disk (downloading them on first use) is a
# one-time cost, not something to repeat per call or per request.
_embedder: OnnxBgeEmbeddings | None = None


def get_embedder() -> OnnxBgeEmbeddings:
    """Return the shared embedder instance, constructing it on first call.

    Both ``ingestion/build_index.py`` and ``rag/retrieval.py`` call this
    rather than constructing ``OnnxBgeEmbeddings`` themselves, so there is
    exactly one embedding implementation in this codebase - the drift
    Checkpoint 4e found (``ingestion/build_index.py`` had its own
    disconnected embedding logic) can't recur by construction.
    """
    global _embedder
    if _embedder is None:
        _embedder = OnnxBgeEmbeddings()
    return _embedder

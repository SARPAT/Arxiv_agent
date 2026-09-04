"""Centralized runtime configuration for the FastAPI service and the RAG
pipeline it wraps.

Every runtime knob that previously lived as a module-level constant
scattered across ``rag/`` (embedding/generation model names, the
confidence gate's threshold, retrieval and context-assembly sizes) is
defined here instead, loaded once from the environment (and a local
``.env`` file) via pydantic-settings. ``rag/`` modules import from this
module rather than defining their own constants.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    nvidia_api_key: str = ""

    # No default: app/session.py needs a real Redis endpoint to store
    # conversation history, so an unset REDIS_URL should fail at startup
    # rather than the service silently running with nowhere to persist
    # sessions.
    redis_url: str

    # Checkpoint 4e swapped bge-base (768-dim, ~440MB weights) for
    # bge-small (384-dim, ~130MB weights), assuming model weight size was
    # the dominant memory cost. A continuous psutil RSS trace disproved
    # that: baseline memory climbed past 700MB within 15 seconds of
    # process start, before any request or model load - the real cost is
    # torch + sentence-transformers + langchain's own import/runtime
    # footprint, largely independent of which model file sits on top of
    # it. Checkpoint 4f keeps this same logical model but removes that
    # framework: rag/embedder.py runs it via ONNX Runtime instead (no
    # torch dependency at all). This string is now a logical/display
    # name only - see rag/embedder.py for the actual ONNX model source.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generation_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    max_tokens: int = 512

    # STALE as of the ONNX runtime switch (Checkpoint 4f): calibrated by
    # eval/calibrate_threshold.py against bge-small-en-v1.5 run through
    # sentence-transformers in fp32 (golden set's 50 in-corpus / 14
    # out-of-corpus labels; full detail in eval/calibration_result.json -
    # ROC-AUC 0.9386, sensitivity 0.84 at specificity 0.9286, target was
    # specificity >= 0.90). Quantization and a different inference
    # runtime produce different float values than that path, even for
    # the same logical model, so this value is meaningless against an
    # ONNX-built index and MUST be replaced with a fresh calibration run
    # before the index is rebuilt and this change ships - do not deploy
    # embedding_model/rag/embedder.py and this threshold out of sync with
    # each other or with the committed index.
    similarity_threshold: float = 0.433317

    retrieval_k: int = 4
    max_context_chars: int = 2500

    # The deployed frontend's origin (a Hugging Face Space), so app/api.py
    # can allow cross-origin requests from it. Empty by default - unset
    # until that Space exists and its URL is known - which app/api.py
    # treats as "don't add CORS support" rather than "allow everything."
    cors_allowed_origin: str = ""


settings = Settings()

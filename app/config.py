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

    # Recalibrated post-index-rebuild (PR #19: the torch-index/ONNX-embedder
    # mismatch fix, plus the new meta plain_overview chunk), from
    # eval/calibrate_threshold.py's tradeoff curve at the same 70%
    # specificity target used since Checkpoint 4g - not re-derived from
    # scratch, since the 4g rationale for that target (the "unrelated"
    # out-of-corpus category already saturated at 100% specificity even at
    # the loosest tested target, so a stricter target only trades away
    # real-question sensitivity for marginal "adjacent_uncovered"
    # protection) is about the shape of the tradeoff, not the specific
    # embedder/index pairing, and hasn't been re-examined against the new
    # curve. The previous value (0.500706) was calibrated against the
    # mismatched pairing PR #19 fixes and should not be assumed comparable.
    #
    # NOTE: eval/calibration_curve.json and eval/calibration_result.json
    # in this repo are still the pre-rebuild artifacts as of this commit -
    # this value has not yet been reconciled against a committed curve
    # file reflecting the same run it came from.
    similarity_threshold: float = 0.5035042762756348

    retrieval_k: int = 4
    max_context_chars: int = 2500

    # The deployed frontend's origin (a Hugging Face Space), so app/api.py
    # can allow cross-origin requests from it. Empty by default - unset
    # until that Space exists and its URL is known - which app/api.py
    # treats as "don't add CORS support" rather than "allow everything."
    cors_allowed_origin: str = ""


settings = Settings()

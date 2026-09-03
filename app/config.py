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

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    generation_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    max_tokens: int = 512

    # Calibrated by eval/calibrate_threshold.py against the golden set's
    # 50 in-corpus / 14 out-of-corpus labels. Full detail in
    # eval/calibration_result.json: ROC-AUC 0.9514, sensitivity 0.84 at
    # specificity 0.9286 (target was specificity >= 0.90).
    similarity_threshold: float = 0.493409663438797

    retrieval_k: int = 4
    max_context_chars: int = 2500

    # The deployed frontend's origin (a Hugging Face Space), so app/api.py
    # can allow cross-origin requests from it. Empty by default - unset
    # until that Space exists and its URL is known - which app/api.py
    # treats as "don't add CORS support" rather than "allow everything."
    cors_allowed_origin: str = ""


settings = Settings()

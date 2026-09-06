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

    # No similarity_threshold: the confidence gate that used to consume it
    # was removed (see rag/pipeline.py). On dense-only retrieval the raw L2
    # score bands for correct in-corpus answers and out-of-corpus junk
    # overlapped too much for any single threshold to separate them without
    # rejecting legitimate answers, so the pipeline now always answers and
    # relies on the system prompt for provenance labeling instead. The
    # calibration tooling (eval/calibrate_threshold.py) is kept for the
    # eventual hybrid-search recalibration but no longer feeds any runtime
    # setting.

    # Qdrant Cloud replaced the local FAISS index in Checkpoint 6. No
    # defaults for the URL or API key: an unset QDRANT_URL/QDRANT_API_KEY
    # should fail loudly at startup rather than the service booting with
    # nowhere to retrieve from, the same reasoning as redis_url above.
    # Every consumer (rag/vectorstore.py is the only module that builds a
    # QdrantClient, plus ingestion through it) reads these from here -
    # nothing hardcodes a connection value, because config drift between
    # this file and a module's own copy already caused a real bug once
    # (Checkpoint 4e's hardcoded embedding-model constant in ingestion).
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str = "arxiv_agent"

    retrieval_k: int = 4
    max_context_chars: int = 2500

    # The deployed frontend's origin (a Hugging Face Space), so app/api.py
    # can allow cross-origin requests from it. Empty by default - unset
    # until that Space exists and its URL is known - which app/api.py
    # treats as "don't add CORS support" rather than "allow everything."
    cors_allowed_origin: str = ""


settings = Settings()

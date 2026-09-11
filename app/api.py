"""FastAPI backend exposing the RAG pipeline as a streaming HTTP API.

``POST /chat`` streams the answer back as Server-Sent Events so the
frontend can render tokens as they arrive instead of waiting for the full
response. ``POST /upload`` (Checkpoint 7) ingests one PDF into the
uploading session's own Qdrant tenant, and ``GET /upload/status`` reports
whether a session has one - the frontend needs that to show an accurate
indicator after a page refresh, when its own in-memory state is gone.
``GET /corpus/info`` gives the frontend the corpus's paper titles and the
upload size cap so neither is hardcoded on that side.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.session import append_turn, get_history, get_upload
from ingestion.upload import UploadRejected, process_upload
from rag.pipeline import REAL_PAPER_TITLES, run_pipeline_stream

# uvicorn configures its own "uvicorn"/"uvicorn.access"/"uvicorn.error"
# loggers but never touches the root logger, so without this, every
# `logging.getLogger(__name__)` call elsewhere in app/ and rag/ (cache
# hit/miss lines, Redis-degradation warnings, etc.) is silently dropped
# under a real server run - the root logger has no handler, and its
# default level (WARNING) filters out INFO-level records before a handler
# ever gets a chance to run. Configuring this here, at the module level of
# the app's entry point, ensures it's in place regardless of whether the
# process was started via `uvicorn app.api:app` or by importing this
# module directly.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

app = FastAPI()

# The frontend (a Hugging Face Space) runs on a different origin than this
# API, so browser requests to /chat need CORS allowed explicitly. The
# origin comes from an env var, not a hardcoded value, because it isn't
# known until that Space exists - and stays configurable afterward without
# a code change if the Space's URL ever changes. Left unconfigured
# (CORS_ALLOWED_ORIGIN unset), no cross-origin access is granted at all,
# rather than defaulting to "allow everything."
if settings.cors_allowed_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.cors_allowed_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


class ChatRequest(BaseModel):
    session_id: str = ""
    message: str


def _format_sse(event: str, data: dict) -> str:
    """Render one Server-Sent Events frame: an ``event:`` line, a
    ``data:`` line holding JSON, and the blank line SSE requires between
    frames."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_chat(session_id: str, message: str) -> Iterator[str]:
    """Drive one ``/chat`` request: yield SSE frames as the pipeline
    streams tokens, then record the finished turn in session history and
    yield the closing ``done`` frame.

    ``session_id`` is passed down to the pipeline, not just used for
    history: it is what decides whether retrieval searches the shared
    corpus alone or the corpus plus this session's uploaded document.

    A generation failure (see ``rag/pipeline.py``'s "error" event) yields
    an ``error`` frame instead of ``done`` and returns without ever calling
    ``append_turn`` — an incomplete or failed exchange has no complete
    answer to persist, and shouldn't show up as one in future history."""
    history = get_history(session_id)
    accumulated = []

    for event in run_pipeline_stream(message, history=history, session_id=session_id):
        if event["type"] == "token":
            accumulated.append(event["delta"])
            yield _format_sse("token", {"delta": event["delta"]})
        elif event["type"] == "done":
            append_turn(session_id, message, "".join(accumulated))
            yield _format_sse(
                "done",
                {
                    "sources": event["sources"],
                    "session_id": session_id,
                },
            )
        elif event["type"] == "error":
            yield _format_sse(
                "error", {"message": event["message"], "session_id": session_id}
            )


@app.post("/chat")
def chat(request: ChatRequest) -> StreamingResponse:
    """Stream an answer to ``request.message`` as Server-Sent Events.

    A missing or empty ``session_id`` gets a fresh one generated
    server-side, so the frontend never has to guarantee it already has
    one before the first message.
    """
    session_id = request.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _stream_chat(session_id, request.message),
        media_type="text/event-stream",
    )


@app.post("/upload")
async def upload(
    file: UploadFile = File(...), session_id: str = Form(...)
) -> dict:
    """Ingest one PDF into ``session_id``'s own tenant, replacing whatever
    that session uploaded before.

    Runs the synchronous pipeline on a worker thread under an explicit
    timeout, so a pathological PDF returns a 504 rather than occupying a
    request slot indefinitely - which matters on a free-tier instance with
    very few of them. (The thread itself cannot be killed; the timeout
    bounds how long the *client* and the request slot wait on it.)

    Every failure the user can act on arrives here as an ``UploadRejected``
    carrying its own status and a message written to be displayed as-is -
    above all the scanned-PDF case, which is the one failure a normal user
    will actually hit and the one that is useless as "upload failed".
    """
    if file.size is not None and file.size > settings.max_upload_bytes:
        # Starlette spools a large body to disk rather than memory, but
        # there is still no reason to read a file we already know we will
        # reject.
        raise HTTPException(
            status_code=413,
            detail=(
                f"That file is too large. The limit is "
                f"{settings.max_upload_bytes / 1_048_576:.0f} MB."
            ),
        )

    file_bytes = await file.read()
    try:
        summary = await asyncio.wait_for(
            run_in_threadpool(
                process_upload, file_bytes, file.filename or "document.pdf", session_id
            ),
            timeout=settings.upload_timeout_seconds,
        )
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        logger.warning("upload timed out for session %s", session_id)
        raise HTTPException(
            status_code=504,
            detail="That document took too long to process. Please try a smaller file.",
        ) from exc
    except Exception as exc:
        # Qdrant, the embedder, or Redis. Nothing here is the user's fault
        # and nothing in the detail should leak internals, but the traceback
        # belongs in the logs.
        logger.exception("upload failed for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Could not process the document. Please try again.",
        ) from exc

    return {**summary, "status": "ok"}


@app.get("/corpus/info")
def corpus_info() -> dict:
    """The corpus's paper titles and the upload size cap, for the
    frontend's welcome message and upload widget - neither hardcoded on
    that side, both read from here so they can never drift from what the
    backend actually has and actually enforces.

    ``papers`` is ``rag.pipeline.REAL_PAPER_TITLES``, itself derived from
    ``ingestion.build_index.TARGET_PAPERS`` - the same mapping
    ``retrieved_paper_titles()`` resolves a retrieved chunk's title
    through, so this list and what a real answer can cite are the same
    list, never two. ``max_upload_mb`` is computed from
    ``settings.max_upload_bytes`` on every call rather than cached: both
    reads are an in-memory dict and a config field, cheap enough that
    caching would only add a staleness risk for no measurable benefit.
    """
    return {
        "papers": REAL_PAPER_TITLES,
        "max_upload_mb": settings.max_upload_bytes / (1024 * 1024),
    }


@app.get("/upload/status")
def upload_status(session_id: str) -> dict:
    """Whether ``session_id`` currently has an uploaded document, and which.

    Lets the frontend restore its indicator after a page refresh instead of
    claiming a session has no document when its points are still there.
    """
    upload = get_upload(session_id)
    return {
        "has_upload": upload is not None,
        "filename": upload["filename"] if upload else None,
        "chunk_count": upload["chunk_count"] if upload else None,
    }


@app.get("/health")
def health() -> dict:
    """Liveness check only — does not verify the NVIDIA API or the
    embedding model are reachable."""
    return {"status": "ok"}

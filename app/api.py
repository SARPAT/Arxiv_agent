"""FastAPI backend exposing the RAG pipeline as a streaming HTTP API.

The only route that matters is ``POST /chat``, which streams the answer
back as Server-Sent Events so the frontend can render tokens as they
arrive instead of waiting for the full response.
"""

import json
import uuid
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.session import append_turn, get_history
from rag.pipeline import run_pipeline_stream

app = FastAPI()


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
    yield the closing ``done`` frame."""
    history = get_history(session_id)
    accumulated = []

    for event in run_pipeline_stream(message, history=history):
        if event["type"] == "token":
            accumulated.append(event["delta"])
            yield _format_sse("token", {"delta": event["delta"]})
        elif event["type"] == "done":
            append_turn(session_id, message, "".join(accumulated))
            yield _format_sse(
                "done",
                {
                    "sources": event["sources"],
                    "abstained": event["abstained"],
                    "session_id": session_id,
                },
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


@app.get("/health")
def health() -> dict:
    """Liveness check only — does not verify the NVIDIA API or the
    embedding model are reachable."""
    return {"status": "ok"}

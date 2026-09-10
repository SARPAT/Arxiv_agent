"""Gradio chat frontend for the Arxiv Agent RAG pipeline.

This process talks to the FastAPI backend over plain HTTP (see
``app/api.py``'s ``/chat`` contract) — it has no direct dependency on
``rag/`` at all, so it can run as a genuinely separate process, possibly
on a different machine, from the backend it calls.
"""

import json
import os
import uuid

import gradio as gr
import httpx
import spaces

# Not hardcoded to localhost, so this can point at a deployed backend
# later without a code change.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Generous, because it has to cover a cold start. Render's free tier spins
# the backend down after inactivity, so the first upload after an idle
# period pays a full wake-up before any parsing begins.
UPLOAD_TIMEOUT_SECONDS = 180

WAKE_UP_NOTICE = (
    "the first request after a period of inactivity can take 30-60 seconds "
    "while the backend wakes up"
)


@spaces.GPU
def _zerogpu_stub():
    """Satisfies Hugging Face Spaces' free-tier ZeroGPU requirement.

    This app does no local GPU or model work at all - embedding and
    generation both happen on the Render backend over HTTP - but Spaces'
    CPU Basic tier is paid-only, and the free tier requires at least one
    `@spaces.GPU`-decorated call to pass its startup check. Called once
    below, at import time, not per-request: it's a no-op that exists
    purely to satisfy that platform requirement.
    """


_zerogpu_stub()


def _parse_sse_stream(response: httpx.Response):
    """Yield ``(event, data)`` pairs from a raw SSE response body.

    A minimal parser matching only the shape ``app/api.py`` actually
    sends: one ``event: <name>`` line followed by one ``data: <json>``
    line per frame, frames separated by a blank line.
    """
    event_name = None
    for line in response.iter_lines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data = json.loads(line[len("data:") :].strip())
            yield event_name, data


def chat_fn(message: str, chat_history: list[dict], session_id: str):
    """Gradio submit handler: streams tokens into the chat bubble as they
    arrive, then attaches sources and adopts the session id the backend
    returns once its ``done`` event lands.

    An ``error`` event (generation failed, either before any tokens were
    sent or partway through) appends the backend's message to whatever
    content the bubble already has - empty, if the failure was before the
    first token - and still adopts ``session_id`` from it, so a failure on
    the very first message of a session doesn't leave the frontend without
    one. Either ``done`` or ``error`` ends the generator normally, so the
    UI never hangs waiting for a frame that isn't coming.
    """
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    answer = ""

    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/chat",
        json={"session_id": session_id, "message": message},
        timeout=None,
    ) as response:
        for event_name, data in _parse_sse_stream(response):
            if event_name == "token":
                answer += data["delta"]
                chat_history[-1]["content"] = answer
                yield chat_history, session_id
            elif event_name == "done":
                if data["sources"]:
                    answer += "\n\nSources:\n" + "\n".join(
                        f"- {source}" for source in data["sources"]
                    )
                    chat_history[-1]["content"] = answer
                session_id = data["session_id"]
                yield chat_history, session_id
            elif event_name == "error":
                answer += f"\n\n{data['message']}" if answer else data["message"]
                chat_history[-1]["content"] = answer
                session_id = data["session_id"]
                yield chat_history, session_id


def _error_detail(response: httpx.Response) -> str:
    """The backend's own error message, or a fallback if the body isn't the
    JSON shape FastAPI produces.

    Surfacing the backend's text verbatim is deliberate: the scanned-PDF
    rejection explains something the user must actually do something about,
    and "upload failed" would throw that away."""
    try:
        return response.json()["detail"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return f"Upload failed ({response.status_code})."


def upload_fn(file_path: str | None, session_id: str):
    """Gradio upload handler: send one PDF to the backend and report what
    happened.

    A generator so the processing state is visible while the request is in
    flight - a silent spinner across a cold start reads as broken. Wrapped
    end to end in try/except: a network failure here must render as a
    message, not a stack trace in the Space's logs and nothing in the UI.
    """
    if not file_path:
        yield ""
        return

    filename = os.path.basename(file_path)
    yield f"Processing **{filename}**... ({WAKE_UP_NOTICE})."

    try:
        with open(file_path, "rb") as handle:
            response = httpx.post(
                f"{BACKEND_URL}/upload",
                files={"file": (filename, handle, "application/pdf")},
                data={"session_id": session_id},
                timeout=UPLOAD_TIMEOUT_SECONDS,
            )
    except httpx.HTTPError as exc:
        yield f"Could not reach the backend: {exc}"
        return

    if response.status_code != 200:
        yield _error_detail(response)
        return

    body = response.json()
    yield (
        f"Document in this session: **{body['filename']}** "
        f"({body['chunk_count']} chunks). Ask a question about it."
    )


def upload_status_fn(session_id: str) -> str:
    """Restore the document indicator on page load.

    Without this, a refresh would show no document while the session's
    chunks are still in Qdrant and still being searched - the frontend's
    own state is gone, the backend's is not."""
    try:
        response = httpx.get(
            f"{BACKEND_URL}/upload/status",
            params={"session_id": session_id},
            timeout=30,
        )
        response.raise_for_status()
        status = response.json()
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return ""
    if not status.get("has_upload"):
        return ""
    return (
        f"Document in this session: **{status['filename']}** "
        f"({status['chunk_count']} chunks)."
    )


with gr.Blocks(title="Arxiv Agent") as demo:
    # A CALLABLE, not a value. gr.State deep-copies whatever it is given
    # at build time, so gr.State(str(uuid.uuid4())) would hand every
    # visitor to the Space the same id - one shared conversation history
    # for everyone, and, now that a session id is also a Qdrant tenant,
    # one user's uploaded PDF searched on another user's questions. A
    # callable is invoked per page load, which is what actually makes a
    # session a session.
    session_state = gr.State(lambda: str(uuid.uuid4()))
    chatbot = gr.Chatbot(label="Arxiv Agent")
    msg = gr.Textbox(
        label="Ask a question",
        placeholder="What is the Transformer architecture?",
    )

    with gr.Accordion("Ask about your own PDF", open=False):
        upload = gr.File(
            label="Upload a PDF",
            file_types=[".pdf"],
            file_count="single",
            type="filepath",
        )
        upload_status = gr.Markdown()

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, session_state],
    ).then(lambda: "", outputs=msg)

    # change fires on both a new file and a cleared one; upload_fn returns
    # an empty status for the cleared case rather than erroring on None.
    upload.change(upload_fn, inputs=[upload, session_state], outputs=upload_status)

    # A session's uploaded document outlives this page, so the indicator is
    # restored from the backend on load rather than assumed absent.
    demo.load(upload_status_fn, inputs=session_state, outputs=upload_status)


if __name__ == "__main__":
    demo.launch()

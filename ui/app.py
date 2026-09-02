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

# Not hardcoded to localhost, so this can point at a deployed backend
# later without a code change.
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


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
    returns once its ``done`` event lands."""
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


with gr.Blocks(title="Arxiv Agent") as demo:
    session_state = gr.State(str(uuid.uuid4()))
    chatbot = gr.Chatbot(label="Arxiv Agent")
    msg = gr.Textbox(
        label="Ask a question",
        placeholder="What is the Transformer architecture?",
    )

    msg.submit(
        chat_fn,
        inputs=[msg, chatbot, session_state],
        outputs=[chatbot, session_state],
    ).then(lambda: "", outputs=msg)


if __name__ == "__main__":
    demo.launch()

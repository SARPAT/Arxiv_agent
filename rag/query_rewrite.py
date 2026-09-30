"""Turn the user's latest message into a standalone search query.

Retrieval embeds exactly the text it's given. A follow-up like "who are
the authors of this paper?" carries no paper at all, so embedded as-is it
lands on whichever corpus chunks talk most about authors and papers -
in production, the RAG paper's, while the conversation was about an
uploaded CacheBlend PDF. The generator sees the conversation history;
the retriever never did. This module is the step that gives the
retriever the same context, by rewriting the message into one that
stands on its own.

It also decides one thing about scope. When a session has uploaded a
document and the question is only about that document ("summarize
this", "who wrote it"), retrieval searches that document alone instead of
letting the shared corpus compete with it on cosine similarity. That's
what makes "this paper" right after an upload find the upload, even
before there is any history to resolve it from.

Costs one short model call per turn, and only on turns that need it:
the first message of a session with no upload goes straight through
untouched. The call is best-effort. Any failure (timeout, API error,
malformed output) falls back to the raw message and the default scope,
which is exactly the behaviour before this module existed - rewriting
can make retrieval better but never makes a turn fail.
"""

import json
import logging
import re
from dataclasses import dataclass

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from app.config import settings

logger = logging.getLogger(__name__)

# Well under the answer's own 15s timeout: this runs before retrieval, so
# every second here is a second before the first answer token. No retry,
# for the same reason - the fallback is always available and always safe.
REWRITE_TIMEOUT_SECONDS = 8

# Only recent turns can hold what "this" or "it" refers to, and answers
# can be long - bounding both keeps the call small whatever the session's
# length.
_MAX_HISTORY_MESSAGES = 6
_MAX_MESSAGE_CHARS = 600
_MAX_QUERY_CHARS = 500

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT = """You turn a user's latest message in a conversation about \
research papers into a standalone search query for retrieving passages \
from those papers.

Rules:
- Resolve every reference ("this paper", "it", "they", "the method", \
"that") to the specific paper or topic it points to in the conversation, \
and name it explicitly in the query.
- Do not answer the question. Do not add facts that aren't in the \
conversation.
- If the latest message already stands on its own, return it unchanged.
{upload_rule}
Respond with only a JSON object, no other text:
{{"query": "<standalone query>", "upload_only": <true or false>}}"""

_UPLOAD_RULE = """- The user has uploaded a document named "{filename}". Set \
"upload_only" to true only when the latest message is solely about that \
uploaded document - including "this paper"/"this document" when nothing \
else in the conversation is what it refers to. If it's about another \
paper, or compares the upload with another paper, set it to false."""

_NO_UPLOAD_RULE = '- Always set "upload_only" to false.'


@dataclass(frozen=True)
class SearchPlan:
    """What retrieval should search for, and whether only the session's
    uploaded document should be searched."""

    query: str
    upload_only: bool = False


_client: ChatNVIDIA | None = None


def _get_client() -> ChatNVIDIA:
    """A separate client from generation's: a short output cap, a shorter
    timeout, and temperature 0, since this is extraction, not writing."""
    global _client
    if _client is None:
        _client = ChatNVIDIA(
            model=settings.generation_model,
            api_key=settings.nvidia_api_key,
            max_tokens=128,
            temperature=0,
            chat_template_kwargs={"enable_thinking": False},
            timeout=REWRITE_TIMEOUT_SECONDS,
        )
    return _client


def _build_messages(
    query: str, history: list[dict[str, str]], upload_filename: str | None
) -> list[dict[str, str]]:
    rule = (
        _UPLOAD_RULE.format(filename=upload_filename)
        if upload_filename
        else _NO_UPLOAD_RULE
    )
    transcript = "\n".join(
        f"{turn['role']}: {turn['content'][:_MAX_MESSAGE_CHARS]}"
        for turn in history[-_MAX_HISTORY_MESSAGES:]
    )
    return [
        {"role": "system", "content": _PROMPT.format(upload_rule=rule)},
        {
            "role": "user",
            "content": f"Conversation so far:\n{transcript or '(none)'}\n\n"
            f"Latest message: {query}",
        },
    ]


def _parse(raw: str, has_upload: bool) -> SearchPlan | None:
    """The model's reply as a ``SearchPlan``, or ``None`` if it isn't a
    usable one. Tolerates prose or code fences around the JSON object,
    but not a missing, empty or runaway query."""
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    query = data.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    if len(query) > _MAX_QUERY_CHARS:
        return None
    # upload_only only means something when there's an upload to restrict
    # to; anything but a literal true is false.
    upload_only = has_upload and data.get("upload_only") is True
    return SearchPlan(query=query.strip(), upload_only=upload_only)


def plan_search(
    query: str,
    history: list[dict[str, str]] | None = None,
    upload_filename: str | None = None,
) -> SearchPlan:
    """The query retrieval should embed, resolved against ``history``,
    and whether to search only the uploaded document.

    With no history and no upload there is nothing to resolve, so this
    returns ``query`` unchanged without a model call.
    """
    if not history and not upload_filename:
        return SearchPlan(query)

    try:
        response = _get_client().invoke(
            _build_messages(query, history or [], upload_filename)
        )
        plan = _parse(response.content, has_upload=bool(upload_filename))
    except Exception as exc:
        logger.warning("Query rewrite failed (%s); searching the raw message", exc)
        return SearchPlan(query)

    if plan is None:
        logger.warning(
            "Query rewrite returned unusable output; searching the raw message"
        )
        return SearchPlan(query)

    logger.info(
        "search query: %r -> %r (upload_only=%s)", query, plan.query, plan.upload_only
    )
    return plan

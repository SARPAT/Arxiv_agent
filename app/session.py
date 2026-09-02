"""In-memory conversation history store."""

# Interim in-memory session store. Works correctly for a single
# process, but does not survive a restart and would not be shared
# across multiple worker processes. Replaced with Redis in a
# follow-up checkpoint once multi-worker deployment is in scope.
_sessions: dict[str, list[dict[str, str]]] = {}


def get_history(session_id: str) -> list[dict[str, str]]:
    """Return the stored conversation history for ``session_id``, or an
    empty list if this session has no history yet."""
    return _sessions.get(session_id, [])


def append_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Record one user/assistant turn onto ``session_id``'s history."""
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": assistant_msg})

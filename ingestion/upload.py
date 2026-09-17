"""User document upload: PDF bytes in, Qdrant points out.

Checkpoint 7's payoff for the multi-tenancy landed in Checkpoint 6. An
uploaded document goes into the *same* collection as the shared corpus,
separated only by ``tenant_id`` - no second collection, no second store.
The tenant is the session id, which is what makes uploads session-scoped:
they need no account system, no per-user quota and no deletion story,
because a session's points become orphans the moment its Redis keys
expire (see ``scripts/cleanup_orphaned_uploads.py``).

``process_upload()`` is deliberately a standalone function rather than
logic inside the endpoint handler. Processing is synchronous today; moving
it to a queue later should mean changing what calls this, not rewriting
it.

**Ordering matters and is not arbitrary.** Validation runs before any
network call, so a rejected upload costs nothing; the session's previous
points are deleted *before* the new ones are written, so a session holds
one document rather than an accumulating pile; and the Redis marker that
tells retrieval this session has an upload is cleared before the delete
and only set after a successful upsert. That ordering is what makes a
crash anywhere in the middle safe: the session falls back to searching the
shared corpus alone, rather than pointing at points that are half-written
or gone.

Rejections are raised as ``UploadRejected`` subclasses carrying the HTTP
status they deserve, so ``app/api.py`` maps all of them in one place
instead of re-deriving which failure means what.
"""

import io
import logging
import re
import time
from contextlib import contextmanager
from pathlib import PurePosixPath

from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import settings
from app.session import clear_upload, set_upload
from ingestion.build_index import build_splitter
from rag.vectorstore import PUBLIC_TENANT_ID, delete_by_tenant, upsert_chunks

logger = logging.getLogger(__name__)

# Longest filename kept as a chunk's Title/paper_key. It is displayed back
# to the user in the answer's source list, so it is capped for the sake of
# that rendering, not for storage.
MAX_TITLE_CHARS = 120

# Everything outside this set is collapsed to "_". Deliberately narrow: the
# sanitized name becomes a payload value that is filtered on, matched
# against and rendered back to the user, so exotic characters buy nothing
# and risk something.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9 ._-]+")

# A PDF's magic bytes. Some real files carry a little junk before the
# header, which pypdf tolerates, so this looks in the first kilobyte
# rather than only at offset zero.
_PDF_MAGIC = b"%PDF"
_PDF_MAGIC_SEARCH_BYTES = 1024

# Metadata keys a corpus chunk carries that an uploaded one cannot know.
# Set to empty strings rather than omitted, so every point in the
# collection has the same payload shape and no reader has to special-case
# where a chunk came from.
_UNKNOWN_CORPUS_FIELDS = {
    "arxiv_id": "",
    "Published": "",
    "Authors": "",
    "Summary": "",
}


@contextmanager
def _timed(stage: str, timings: dict[str, float]):
    """Record wall-clock seconds for one stage into ``timings``.

    Wall clock, not CPU time, deliberately: what matters here is what the
    request's own timeout measures, which includes time blocked on
    Qdrant's network round trips as much as time spent computing.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = timings.get(stage, 0.0) + (time.perf_counter() - start)


class UploadRejected(Exception):
    """An upload the user can do something about, with the HTTP status it
    should surface as. Subclasses set ``status_code``; the message is
    written to be shown to a user verbatim, not logged and swallowed."""

    status_code = 400


class FileTooLarge(UploadRejected):
    status_code = 413


class NotAPdf(UploadRejected):
    status_code = 400


class NoExtractableText(UploadRejected):
    status_code = 400


def sanitize_filename(filename: str) -> str:
    """Reduce a client-supplied filename to something safe to store and
    display.

    Strips directory components (both separator conventions, so a Windows
    path or a ``../../etc/passwd`` traversal attempt yields just the final
    name), drops a trailing ``.pdf``, collapses anything outside a narrow
    safe set, trims punctuation from the ends and caps the length. Falls
    back to a placeholder when that leaves nothing - a name of ``.pdf`` or
    ``///`` must not produce an empty ``paper_key``.
    """
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    safe = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(" ._-")[:MAX_TITLE_CHARS]
    return safe or "uploaded document"


def extract_text(file_bytes: bytes) -> str:
    """All text from all pages of a PDF, joined.

    Every pypdf failure becomes one ``NotAPdf``. The exception surface here
    is genuinely wide - a malformed file surfaces as ``PdfReadError``,
    ``KeyError``, ``struct.error``, ``ValueError`` and more depending on
    which structure is broken, and an encrypted file raises something else
    again - and every one of them means the same thing to the user, so
    catching broadly and translating once is more honest than an
    ever-growing tuple that silently 500s on the type nobody listed.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - translated to one user-facing error
        logger.info("PDF parse failed: %s", exc)
        raise NotAPdf(
            "That file could not be read as a PDF. Please upload a valid, "
            "unencrypted PDF file."
        ) from exc
    return "\n".join(pages).strip()


def _validate(file_bytes: bytes, session_id: str) -> None:
    """Everything that can be decided without touching Qdrant or the
    embedder, cheapest check first, so a rejected upload costs nothing."""
    if not session_id or session_id == PUBLIC_TENANT_ID:
        # Not user-facing in normal operation - the frontend always sends a
        # session id - but an empty one would otherwise become a tenant of
        # "", and the public one would be an attempt to write into the
        # shared corpus.
        raise UploadRejected("A valid session is required to upload a document.")
    if len(file_bytes) > settings.max_upload_bytes:
        raise FileTooLarge(
            f"That file is {len(file_bytes) / 1_048_576:.1f} MB. The limit is "
            f"{settings.max_upload_bytes / 1_048_576:.0f} MB."
        )
    if _PDF_MAGIC not in file_bytes[:_PDF_MAGIC_SEARCH_BYTES]:
        raise NotAPdf("Only PDF files can be uploaded.")


def _log_timings(session_id: str, timings: dict[str, float], outcome: str) -> None:
    """Emit the one timing line. ``outcome`` distinguishes a completed run
    from one that died partway, because the partial breakdown of a failed
    run is the more useful of the two - it names the stage that was still
    running when everything stopped."""
    total = sum(timings.values())
    breakdown = " ".join(
        f"{stage}={timings[stage]:.2f}s"
        for stage in ("extract", "delete", "chunk", "embed", "upsert")
        if stage in timings
    )
    logger.info(
        "upload timings [%s] session=%s total=%.2fs %s",
        outcome,
        session_id,
        total,
        breakdown,
    )


def process_upload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """Ingest one PDF for one session. Returns a summary for the response.

    Raises an ``UploadRejected`` subclass for anything the user can fix;
    anything else propagating out of here is a genuine server fault and is
    the caller's to turn into a 500.
    """
    _validate(file_bytes, session_id)

    # Every stage below is timed, and the totals go out as one INFO line at
    # the end. An upload that hits the request timeout otherwise tells you
    # only that 60 seconds went somewhere; these say where. One line rather
    # than one per stage so a single grep in Render's logs gets the whole
    # breakdown for a request, including for the run that timed out.
    timings: dict[str, float] = {}

    try:
        return _ingest(file_bytes, filename, session_id, timings)
    except Exception:
        # A failed or abandoned run is the one whose breakdown matters
        # most: whichever stage is missing from the line is the one it was
        # still inside when it stopped.
        _log_timings(session_id, timings, "failed")
        raise


def _ingest(
    file_bytes: bytes, filename: str, session_id: str, timings: dict[str, float]
) -> dict:
    """The upload body proper. Split out from ``process_upload`` only so
    that its timing line is emitted on both the success and failure paths
    without duplicating the whole thing in a ``finally``."""
    with _timed("extract", timings):
        text = extract_text(file_bytes)
    if len(text) < settings.min_extracted_chars:
        # The scanned-PDF case, and the one failure a normal user will
        # actually hit. pypdf returns empty text for an image-only page
        # without raising, so without this check the upload would "succeed"
        # and then quietly retrieve nothing.
        raise NoExtractableText(
            "No readable text could be extracted - this document appears to "
            "be scanned or image-only. Please upload a text-based PDF."
        )

    title = sanitize_filename(filename)

    # Marker off first, then the old points, then the new ones, then the
    # marker back on. A failure at any point leaves the session searching
    # the shared corpus alone rather than a half-written tenant.
    clear_upload(session_id)
    with _timed("delete", timings):
        replaced = delete_by_tenant(session_id)

    document = Document(
        page_content=text,
        metadata={
            "chunk_type": "body",
            "paper_key": title,
            "Title": title,
            **_UNKNOWN_CORPUS_FIELDS,
        },
    )
    with _timed("chunk", timings):
        chunks = build_splitter().split_documents([document])

    # The whole-document text is dead weight from here on - the chunks
    # carry their own copies - and what follows (embedding) is the most
    # memory-hungry step in this function, so it is the worst possible
    # moment to still be holding it. Dropping both references frees the
    # string, since the chunk Documents hold substrings, not views of it.
    #
    # ``file_bytes`` deliberately is NOT dropped here: this frame's is not
    # the only reference - app/api.py's handler and the threadpool call
    # holding it both outlive this function, so a ``del`` here would free
    # nothing while reading as though it did. Freeing it for real means
    # handing this function a file handle instead of bytes; at 1.7MB
    # measured against a 483MB embedding peak, that signature change has
    # not earned itself yet.
    char_count = len(text)
    del text, document

    # embed and upsert are interleaved inside this one call, so it reports
    # its own split rather than being timed from out here as one blob.
    chunk_count = upsert_chunks(chunks, tenant_id=session_id, timings=timings)
    set_upload(session_id, title, chunk_count)

    _log_timings(session_id, timings, "ok")
    logger.info(
        "upload ingested for session %s: %r -> %d chunks (%d chars), "
        "replaced %d previous points",
        session_id,
        title,
        chunk_count,
        char_count,
        replaced,
    )
    return {"filename": title, "chunk_count": chunk_count, "char_count": char_count}

"""
archive_safety.py — bounded handling of uploaded ZIP archives.

``/index/upload`` accepts an archive from the network and expands it onto the
server's disk. Three of the four things that can go wrong there are size
problems, and none of them were bounded (issue #310):

  * the request body was streamed to disk with ``shutil.copyfileobj``, which
    copies until the client stops sending;
  * ``extractall()`` decompressed every member in full, so a well-formed
    archive that passes every structural check can still write orders of
    magnitude more than it occupies — the classic 42.zip is 42 KB and expands
    to 4.5 PB;
  * nothing swept the temp archives left behind when extraction failed.

The fourth, path traversal, was already handled correctly by the endpoint and
is re-checked here so that extraction is safe on its own terms rather than
only in combination with its caller.

Declared sizes are read from the central directory, which costs nothing and
catches the ordinary case. They are also *claims* — a ZIP may declare one size
and deliver another — so :func:`safe_extract_all` additionally counts the bytes
it actually writes and stops when the budget is spent. Both checks are needed:
the cheap one to reject before touching the disk, the expensive one because the
cheap one can be lied to.

All limits are configured in ``config.py``; nothing here reads the environment.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import BinaryIO

from logger import logger


class ArchiveTooLarge(Exception):
    """An upload exceeded one of the configured size limits.

    Carries a message written for the person who uploaded the file: it names
    the limit that was hit and the value that hit it, since "too large" alone
    gives them nothing to act on.
    """


class UnsafeArchiveMember(Exception):
    """An archive member would be written outside the extraction directory."""


def stream_to_file(source: BinaryIO, destination: str, max_bytes: int) -> int:
    """
    Copy *source* to *destination*, refusing to write more than *max_bytes*.

    The point of streaming rather than reading the body into memory and
    measuring it is that an oversized upload must be rejected while it is
    arriving, not after it has been buffered. The partial file is removed
    before raising, so a rejected upload leaves nothing behind.

    ``Content-Length`` is deliberately not trusted: it is supplied by the
    client, and a chunked request need not send one at all.

    Args:
        source: Readable binary stream, e.g. ``UploadFile.file``.
        destination: Path to write to.
        max_bytes: Byte budget. Non-positive disables the limit.

    Returns:
        Number of bytes written.

    Raises:
        ArchiveTooLarge: If the stream exceeds *max_bytes*.
    """
    chunk_size = 1024 * 1024
    written = 0

    try:
        with open(destination, "wb") as handle:
            while True:
                chunk = source.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes > 0 and written > max_bytes:
                    raise ArchiveTooLarge(
                        f"Upload exceeds the {_megabytes(max_bytes)} limit."
                    )
                handle.write(chunk)
    except ArchiveTooLarge:
        _remove_quietly(destination)
        raise

    return written


def inspect_archive(zip_path: str) -> dict:
    """
    Summarise an archive from its central directory, without decompressing.

    Args:
        zip_path: Path to a readable ZIP file.

    Returns:
        A dict with ``entry_count``, ``declared_total`` (sum of uncompressed
        member sizes), ``largest_entry``, ``compressed_size`` (the archive on
        disk) and ``ratio`` (declared_total / compressed_size, 0.0 for an
        empty archive).
    """
    with zipfile.ZipFile(zip_path, "r") as archive:
        infos = archive.infolist()

    declared_total = sum(info.file_size for info in infos)
    compressed_size = os.path.getsize(zip_path)

    return {
        "entry_count": len(infos),
        "declared_total": declared_total,
        "largest_entry": max((info.file_size for info in infos), default=0),
        "compressed_size": compressed_size,
        "ratio": (declared_total / compressed_size) if compressed_size else 0.0,
    }


def validate_archive_limits(
    zip_path: str,
    *,
    max_extracted_bytes: int,
    max_entries: int,
    max_ratio: float,
) -> dict:
    """
    Reject an archive whose declared contents exceed the configured limits.

    Every value used here comes from the central directory, so this runs
    before a single member is decompressed.

    The compression ratio is checked in addition to the absolute total because
    the two catch different things: the total catches "this is simply too big",
    the ratio catches a small upload engineered to expand enormously, which is
    the shape of a deliberate bomb rather than a large project.

    An archive at or below one member is exempt from the ratio check — a
    single highly compressible file is unremarkable, and the absolute limits
    already bound it.

    Args:
        zip_path: Path to the archive.
        max_extracted_bytes: Budget for the total uncompressed size.
        max_entries: Maximum number of members.
        max_ratio: Maximum declared_total / compressed_size.

    Returns:
        The :func:`inspect_archive` summary, for logging.

    Raises:
        ArchiveTooLarge: If any limit is exceeded.
    """
    stats = inspect_archive(zip_path)

    if max_entries > 0 and stats["entry_count"] > max_entries:
        raise ArchiveTooLarge(
            f"Archive contains {stats['entry_count']} entries, which exceeds "
            f"the limit of {max_entries}."
        )

    if max_extracted_bytes > 0 and stats["declared_total"] > max_extracted_bytes:
        raise ArchiveTooLarge(
            f"Archive expands to {_megabytes(stats['declared_total'])}, which "
            f"exceeds the {_megabytes(max_extracted_bytes)} limit."
        )

    if (
        max_ratio > 0
        and stats["entry_count"] > 1
        and stats["ratio"] > max_ratio
    ):
        raise ArchiveTooLarge(
            f"Archive expands {stats['ratio']:.0f}x, which exceeds the "
            f"maximum compression ratio of {max_ratio:.0f}x."
        )

    return stats


def is_safe_member(name: str) -> bool:
    """
    Return whether *name* stays inside the extraction directory.

    Rejects absolute paths and any name that normalises to a location above
    the root. Backslashes are folded to forward slashes first, so a Windows
    style ``..\\..\\evil`` is caught when extracting on POSIX, where
    ``os.path.normpath`` would otherwise treat it as one long filename.
    """
    if not name:
        return False

    candidate = name.replace("\\", "/")
    if candidate.startswith("/") or os.path.isabs(candidate):
        return False
    if os.path.splitdrive(candidate)[0]:
        return False

    normalised = os.path.normpath(candidate)
    return not (normalised == ".." or normalised.startswith("../"))


def safe_extract_all(
    zip_path: str,
    destination: str,
    *,
    max_extracted_bytes: int,
) -> int:
    """
    Extract *zip_path* into *destination*, one member at a time, under budget.

    :func:`validate_archive_limits` has already checked the sizes the archive
    *declares*. Those are claims: the central directory may say one thing and
    the compressed stream deliver another, and a bomb is exactly the kind of
    file that would lie. So this counts bytes as they are written and stops
    when the budget is spent, rather than trusting the header a second time.

    On any failure the partially extracted tree is removed, so a rejected
    upload does not leave half a repository behind for the indexer to find.

    Args:
        zip_path: Path to the archive.
        destination: Directory to extract into. Created if absent.
        max_extracted_bytes: Byte budget across all members. Non-positive
            disables the limit.

    Returns:
        Total bytes written.

    Raises:
        ArchiveTooLarge: If the extracted total exceeds the budget.
        UnsafeArchiveMember: If a member would be written outside
            *destination*.
    """
    os.makedirs(destination, exist_ok=True)
    root = os.path.realpath(destination)
    written = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if not is_safe_member(info.filename):
                    raise UnsafeArchiveMember(
                        f"Refusing to extract {info.filename!r}: it would be "
                        "written outside the extraction directory."
                    )

                target = os.path.realpath(
                    os.path.join(root, info.filename.replace("\\", "/"))
                )
                if target != root and not target.startswith(root + os.sep):
                    raise UnsafeArchiveMember(
                        f"Refusing to extract {info.filename!r}: it resolves "
                        "outside the extraction directory."
                    )

                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue

                os.makedirs(os.path.dirname(target), exist_ok=True)
                written += _extract_member(
                    archive, info, target, max_extracted_bytes, written
                )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return written


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    target: str,
    max_extracted_bytes: int,
    already_written: int,
) -> int:
    """Write one member to *target*, counting bytes against the budget."""
    chunk_size = 1024 * 1024
    written = 0

    with archive.open(info, "r") as source, open(target, "wb") as handle:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if (
                max_extracted_bytes > 0
                and already_written + written > max_extracted_bytes
            ):
                raise ArchiveTooLarge(
                    f"Archive expands beyond the "
                    f"{_megabytes(max_extracted_bytes)} limit while "
                    f"extracting {info.filename!r}."
                )
            handle.write(chunk)

    return written


def sweep_orphan_uploads(directory: str = ".", prefix: str = "temp_upload_") -> int:
    """
    Delete temp upload archives left behind by earlier runs.

    An archive is orphaned whenever the process stops between accepting an
    upload and extracting it — a restart with jobs still queued, or an
    extraction that raised. Nothing else in the system will ever look at these
    files again, so a sweep at startup is safe and is the only thing that
    bounds their accumulation.

    Args:
        directory: Directory to sweep.
        prefix: Filename prefix identifying a temp upload.

    Returns:
        Number of files removed.
    """
    removed = 0
    try:
        names = os.listdir(directory)
    except OSError:
        return 0

    for name in names:
        if not (name.startswith(prefix) and name.endswith(".zip")):
            continue
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if _remove_quietly(path):
            removed += 1

    if removed:
        logger.info("Removed %d orphaned upload archive(s).", removed)
    return removed


def _remove_quietly(path: str) -> bool:
    """Delete *path*, returning whether it went away. Never raises."""
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("Could not remove %s: %s", path, exc)
        return False


def _megabytes(num_bytes: int) -> str:
    """Format a byte count for a user-facing message."""
    return f"{num_bytes / (1024 * 1024):.0f} MB"

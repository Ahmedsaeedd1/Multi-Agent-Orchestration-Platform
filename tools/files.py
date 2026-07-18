"""
File-system tools for agents: ``read_file`` and ``write_file``.

Both tools enforce that the path is within the current working directory
to prevent directory-traversal attacks.
"""

import os
import logging

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 20_000


def _resolve_path(path: str) -> str:
    """
    Resolve *path* relative to the CWD and ensure it does not escape.
    Raises ``PermissionError`` if the resolved path is outside the CWD.
    """
    cwd = os.path.abspath(os.getcwd())
    requested = os.path.abspath(os.path.join(cwd, path))

    # Normalise both paths so comparison is reliable
    cwd = os.path.realpath(cwd)
    requested = os.path.realpath(requested)

    if not requested.startswith(cwd + os.sep) and requested != cwd:
        raise PermissionError(
            f"Path '{path}' resolves to '{requested}', which is outside "
            f"the working directory '{cwd}' — blocked."
        )
    return requested


def read_file(path: str) -> str:
    """
    Read a text file from the project directory.  Returns up to
    20 000 characters of content.  Raises if the path attempts to
    escape the working directory.
    """
    resolved = _resolve_path(path)

    if not os.path.isfile(resolved):
        return f"Error: file not found at '{resolved}'"

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(_MAX_READ_CHARS + 1)
        if len(content) > _MAX_READ_CHARS:
            content = content[:_MAX_READ_CHARS] + "\n\n... (truncated)"
        return content
    except Exception as e:
        logger.error("read_file failed for %s: %s", resolved, e)
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """
    Write text content to a file in the project directory.  Returns
    ``"ok: <path>"`` on success.  Raises if the path attempts to escape
    the working directory, or if the parent directory does not exist.
    """
    resolved = _resolve_path(path)

    parent = os.path.dirname(resolved)
    if not os.path.isdir(parent):
        return f"Error: parent directory '{parent}' does not exist"

    try:
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content)
        return f"ok: {path}"
    except Exception as e:
        logger.error("write_file failed for %s: %s", resolved, e)
        return f"Error writing file: {e}"
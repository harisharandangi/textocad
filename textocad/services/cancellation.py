"""Cooperative cancellation for a running generation."""

from __future__ import annotations

import contextvars
import os
import re
import time
from pathlib import Path

_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_current: contextvars.ContextVar[str | None] = contextvars.ContextVar("textocad_generation_id", default=None)
_MAX_AGE_S = 6 * 3600


class GenerationCancelled(BaseException):
    """Raised at the next checkpoint after Stop."""


def _root() -> Path:
    base = Path(os.getenv("TEXTOCAD_BUILD_ROOT", os.getenv("TEXTOCAD_BUILD_DIR", "generated_builds")))
    root = base / "_cancel"
    root.mkdir(parents=True, exist_ok=True)
    return root


def valid_id(request_id: str | None) -> bool:
    return bool(request_id and _ID_RE.match(request_id))


def begin(request_id: str | None, owner: str) -> contextvars.Token | None:
    """Mark a generation as running for ``owner`` and make it the current one."""
    if not valid_id(request_id):
        return None
    root = _root()
    try:
        (root / f"{request_id}.owner").write_text(str(owner or ""), encoding="utf-8")
        _sweep(root)
    except OSError:
        pass
    return _current.set(request_id)


def end(token: contextvars.Token | None) -> None:
    if token is None:
        return
    request_id = _current.get()
    _current.reset(token)
    if request_id:
        for suffix in (".owner", ".stop"):
            try:
                (_root() / f"{request_id}{suffix}").unlink()
            except OSError:
                pass


def request_cancel(request_id: str | None, owner: str) -> bool:
    """Flag a running generation to stop. Only its owner may."""
    if not valid_id(request_id):
        return False
    root = _root()
    try:
        started_by = (root / f"{request_id}.owner").read_text(encoding="utf-8")
    except OSError:
        return False
    if started_by != str(owner or ""):
        return False
    (root / f"{request_id}.stop").write_text(str(time.time()), encoding="utf-8")
    return True


def check() -> None:
    """Checkpoint: raise if the current generation was stopped."""
    request_id = _current.get()
    if request_id and (_root() / f"{request_id}.stop").exists():
        raise GenerationCancelled("Generation stopped.")


def _sweep(root: Path) -> None:
    """Remove flags left behind by crashed workers."""
    cutoff = time.time() - _MAX_AGE_S
    for f in root.iterdir():
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass

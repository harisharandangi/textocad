"""Run heavy geometry work (CadQuery / OpenCASCADE / trimesh) in a short-lived child process instead of."""

from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
import time
from typing import Any, Callable


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


ISOLATION_ENABLED = os.getenv("TEXTOCAD_ISOLATE_GEOMETRY", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
VERIFY_TIMEOUT_SECONDS = _env_int("TEXTOCAD_VERIFY_TIMEOUT_SECONDS", 300)
_SLOTS = threading.BoundedSemaphore(_env_int("TEXTOCAD_VERIFY_CONCURRENCY", 2))


_CTX = multiprocessing.get_context("spawn")


class IsolationError(RuntimeError):
    """The child could not deliver a result: timeout, crash (e.g."""


def _log(message: str) -> None:
    print(f"[TextoCAD][geometry-isolation] {message}", flush=True)


_OOM_ADJ_PATH = "/proc/self/oom_score_adj"
_EXPENDABLE_SCORE = "1000"


def mark_expendable() -> None:
    """Mark the CURRENT process as the first one to kill when RAM runs out."""
    try:
        with open(_OOM_ADJ_PATH, "w", encoding="ascii") as handle:
            handle.write(_EXPENDABLE_SCORE)
    except OSError:
        pass


def expendable_command(cmd: list[str]) -> list[str]:
    """Wrap a subprocess command so the started program is marked expendable."""
    if not os.path.exists(_OOM_ADJ_PATH):
        return list(cmd)
    return [
        "sh", "-c",
        f'{{ echo {_EXPENDABLE_SCORE} > {_OOM_ADJ_PATH}; }} 2>/dev/null; exec "$@"',
        "textocad-cad-job",
        *[str(part) for part in cmd],
    ]


def _child_entry(conn: Any, fn: Callable[..., Any], args: tuple[Any, ...]) -> None:
    """Runs inside the child: call fn, send ('ok', result) or ('err', exception)."""
    mark_expendable()
    try:
        result = fn(*args)
        message: tuple[str, Any] = ("ok", result)
    except BaseException as exc:
        message = ("err", exc)
    try:
        pickle.dumps(message[1])
    except Exception as exc:
        message = ("err", RuntimeError(f"unpicklable child {message[0]}: {exc!r}"))
    try:
        conn.send(message)
    finally:
        conn.close()


def _run_in_child(fn: Callable[..., Any], args: tuple[Any, ...], timeout_s: float) -> Any:
    parent_conn, child_conn = _CTX.Pipe(duplex=False)
    proc = _CTX.Process(target=_child_entry, args=(child_conn, fn, args), daemon=True)
    try:
        proc.start()
    except Exception as exc:
        parent_conn.close()
        child_conn.close()
        raise IsolationError(f"could not start child: {exc!r}") from exc
    child_conn.close()

    started = time.monotonic()
    try:


        if not parent_conn.poll(timeout_s):
            raise IsolationError(f"timed out after {timeout_s:.0f}s")
        try:
            status, payload = parent_conn.recv()
        except EOFError as exc:
            proc.join(5)
            raise IsolationError(
                f"child exited without a result (exit code {proc.exitcode}; "
                "-9 usually means it was killed for using too much memory)"
            ) from exc
    finally:
        proc.join(3)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(5)
        parent_conn.close()
        _log(f"{getattr(fn, '__name__', fn)} finished in {time.monotonic() - started:.1f}s "
             f"(exit code {proc.exitcode})")

    if status == "ok":
        return payload
    raise payload


def run_isolated(fn: Callable[..., Any], *args: Any, timeout_s: float | None = None) -> Any:
    """Call ``fn(*args)`` in a fresh child process and return its result."""
    if not ISOLATION_ENABLED:
        return fn(*args)

    try:
        pickle.dumps((fn, args))
    except Exception as exc:
        _log(f"cannot isolate {getattr(fn, '__name__', fn)} ({exc!r}); running in-process")
        return fn(*args)

    with _SLOTS:
        try:
            return _run_in_child(fn, args, float(timeout_s or VERIFY_TIMEOUT_SECONDS))
        except IsolationError as exc:
            if str(exc).startswith("could not start child"):
                _log(f"{exc}; running in-process")
                return fn(*args)
            raise

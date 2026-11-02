from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from textocad.core.config import settings
from textocad.services.cad_schema import CadDocument, cad_document_to_dict
from textocad.services.cadquery_compiler import compile_cadquery


EXECUTABLE_SOURCE_NAME = "cadquery_source.cqbuild"
VIEW_SOURCE_NAME = "source.py.txt"


def _env_true(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def _tail(value: str | bytes | None, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-max_chars:]


def _timeout_seconds() -> float:
    try:
        return float(os.getenv("TEXTOCAD_CADQUERY_TIMEOUT_SECONDS", str(settings.max_build_seconds)))
    except ValueError:
        return float(settings.max_build_seconds)


def _format_failure_message(
    completed: subprocess.CompletedProcess[str],
    step_path: Path,
    stl_path: Path,
    require_step: bool = True,
) -> str:
    details = (
        completed.stderr.strip()
        or completed.stdout.strip()
        or f"CadQuery process exited with code {completed.returncode}"
    )

    missing: list[str] = []
    if require_step and not step_path.exists():
        missing.append("result.step")
    if not stl_path.exists():
        missing.append("result.stl")

    if missing:
        details += "\nMissing expected output file(s): " + ", ".join(missing)

    if "KeyboardInterrupt" in details and "OCP" in details:
        details += (
            "\n\nLikely cause: Uvicorn reload interrupted CadQuery/OCP while it was importing. "
            "Run backend with `python -m uvicorn main:app --reload --reload-dir app`, "
            "or run without `--reload`, and keep TEXTOCAD_BUILD_ROOT outside the backend reload-watched folder."
        )

    return details


def build_cadquery_artifacts(document: CadDocument, build_dir: Path, mode: str = "extended") -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)


    require_step = mode != "standard"
    compiled = compile_cadquery(document, export_step=require_step)


    executable_source_path = build_dir / EXECUTABLE_SOURCE_NAME
    view_source_path = build_dir / VIEW_SOURCE_NAME

    step_path = build_dir / "result.step"
    stl_path = build_dir / "result.stl"
    model_json_path = build_dir / "model.json"

    metadata: dict[str, Any] = {
        "sourcePath": str(view_source_path),
        "executableSourcePath": str(executable_source_path),
        "stepPath": str(step_path),
        "stlPath": str(stl_path),
        "unsupportedOperations": compiled.unsupported_operations,
        "operationSupportSummary": compiled.support_summary,
        "nativeOperations": compiled.support_summary.get("native", []),
        "approximatedOperations": compiled.support_summary.get("approximate", []),
        "skippedOperations": compiled.support_summary.get("skipped", []),
        "warnings": list(compiled.warnings),
        "realCadQueryUsed": False,
        "safeWorker": True,
        "safeFallbackUsed": True,
        "directLlmCodeExecution": False,
        "cadKernel": "none-yet",
        "stdout": "",
        "stderr": "",
    }

    executable_source_path.write_text(compiled.source, encoding="utf-8")


    view_source_path.write_text(compiled.source, encoding="utf-8")

    model_json_path.write_text(
        json.dumps(cad_document_to_dict(document), indent=2),
        encoding="utf-8",
    )

    if not _env_true("TEXTOCAD_ALLOW_CADQUERY_EXEC"):
        metadata["warnings"].append(
            "CadQuery execution disabled. Set TEXTOCAD_ALLOW_CADQUERY_EXEC=true to enable real CadQuery/OpenCascade rebuilds."
        )
        return metadata

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        from textocad.services.geometry_isolation import expendable_command

        completed = subprocess.run(
            expendable_command([sys.executable, str(executable_source_path)]),
            check=False,
            timeout=_timeout_seconds(),
            capture_output=True,
            text=True,
            cwd=str(build_dir),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        metadata["stdout"] = _tail(exc.stdout)
        metadata["stderr"] = _tail(exc.stderr)
        raise RuntimeError(
            "CadQuery timed out after "
            f"{_timeout_seconds()} seconds. Increase TEXTOCAD_CADQUERY_TIMEOUT_SECONDS "
            "or simplify the model."
        ) from exc

    metadata["stdout"] = _tail(completed.stdout)
    metadata["stderr"] = _tail(completed.stderr)

    step_ok = step_path.exists() or not require_step
    if completed.returncode != 0 or not step_ok or not stl_path.exists():
        raise RuntimeError(
            "CadQuery failed: "
            + _format_failure_message(
                completed, step_path=step_path, stl_path=stl_path, require_step=require_step
            )
        )

    metadata.update(
        {
            "realCadQueryUsed": True,
            "safeWorker": False,
            "safeFallbackUsed": False,
            "cadKernel": "cadquery",
        }
    )

    return metadata

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from textocad.core.config import settings
from textocad.services.cad_schema import CadDocument, cad_document_to_dict
from textocad.services.openscad_compiler import compile_openscad


def _env_true(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def build_openscad_artifacts(document: CadDocument, build_dir: Path) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    compiled = compile_openscad(document)
    scad_path = build_dir / "model.scad"
    stl_path = build_dir / "result.stl"
    metadata: dict[str, Any] = {
        "sourcePath": str(scad_path),
        "stlPath": str(stl_path),
        "unsupportedOperations": compiled.unsupported_operations,
        "operationSupportSummary": compiled.support_summary,
        "nativeOperations": compiled.support_summary.get("native", []),
        "approximatedOperations": compiled.support_summary.get("approximate", []),
        "skippedOperations": compiled.support_summary.get("skipped", []),
        "warnings": list(compiled.warnings),
        "realOpenScadUsed": False,
        "safeWorker": True,
        "safeFallbackUsed": True,
        "directLlmCodeExecution": False,
        "cadKernel": "none-yet",
        "stdout": "",
        "stderr": "",
    }

    scad_path.write_text(compiled.source, encoding="utf-8")
    (build_dir / "model.json").write_text(
        __import__("json").dumps(cad_document_to_dict(document), indent=2),
        encoding="utf-8",
    )

    if not _env_true("TEXTOCAD_ALLOW_OPENSCAD_EXEC"):
        metadata["warnings"].append(
            "OpenSCAD execution disabled. Set TEXTOCAD_ALLOW_OPENSCAD_EXEC=true to enable real OpenSCAD rebuilds."
        )
        return metadata

    executable = os.getenv("TEXTOCAD_OPENSCAD_BIN", "openscad")
    resolved = shutil.which(executable) or (executable if Path(executable).exists() else None)
    if not resolved:
        metadata["warnings"].append("OpenSCAD executable not found. Safe fallback was used.")
        return metadata

    try:
        timeout = float(os.getenv("TEXTOCAD_OPENSCAD_TIMEOUT_SECONDS", str(settings.max_build_seconds)))
    except ValueError:
        timeout = float(settings.max_build_seconds)

    from textocad.services.geometry_isolation import expendable_command

    completed = subprocess.run(
        expendable_command([resolved, "-o", str(stl_path), str(scad_path)]),
        check=False,
        timeout=timeout,
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    metadata["stdout"] = completed.stdout[-4000:]
    metadata["stderr"] = completed.stderr[-4000:]

    if completed.returncode != 0 or not stl_path.exists():
        raise RuntimeError(
            "OpenSCAD failed: "
            + (completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}")
        )

    metadata.update(
        {
            "realOpenScadUsed": True,
            "safeWorker": False,
            "safeFallbackUsed": False,
            "cadKernel": "openscad",
        }
    )
    return metadata

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from textocad.core.config import settings
from textocad.services import llm_models, provider_errors
from textocad.services.codegen_pipeline import CodegenPackage, generate_and_build, route_mode, run_codegen
from textocad.services.codegen_sandbox import ExecResult

MODES = ("printing", "precision", "organic", "robotics")


@dataclass
class Result:
    ok: bool
    mode: str
    title: str = ""
    script: str = ""
    params: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    bbox: list[float] | None = None
    volume: float | None = None
    error: str = ""
    model: str | None = None
    seconds: float = 0.0
    parts: list[dict[str, Any]] = field(default_factory=list)
    robot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _selection():
    return llm_models.resolve_own_key() or llm_models.resolve_default()


def _files(res: ExecResult, extra: dict[str, str | None] | None = None) -> dict[str, str]:
    found = {
        "step": res.step_path,
        "stl": res.stl_path,
        "glb": res.glb_path,
        "3mf": res.threemf_path,
        "source": res.source_path,
        **(extra or {}),
    }
    files = {k: v for k, v in found.items() if v}
    from textocad.services.part_export import stamp_exports

    stamp_exports(files.values())
    return files


def _parts(res: ExecResult) -> list[dict[str, Any]]:
    if res.parts:
        return list(res.parts)
    if not res.glb_path:
        return []
    from textocad.services.part_export import read_parts_manifest

    return read_parts_manifest(Path(res.glb_path).parent)


def generate(
    prompt: str,
    mode: str | None = None,
    *,
    out_dir: str | Path | None = None,
    prior_script: str | None = None,
    prior_prompt: str | None = None,
    extra_details: bool = False,
    kernel: str = "openscad",
    image: dict[str, Any] | None = None,
) -> Result:
    """Prompt -> parametric script -> built STEP/STL/GLB in out_dir."""
    mode = mode or route_mode(prompt)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    build_dir = Path(out_dir or Path("builds") / uuid.uuid4().hex[:12]).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    selection = _selection()
    if selection is None:
        return Result(ok=False, mode=mode, error="No AI provider configured. Set TEXTOCAD_API_KEY (see .env.example).")

    started = time.perf_counter()
    active = llm_models.set_active(selection)
    tracking = provider_errors.start_tracking()
    extra: dict[str, str | None] = {}
    robot_info: dict[str, Any] = {}
    try:
        if mode == "robotics":
            from textocad.services.robot_pipeline import generate_robot

            robot = generate_robot(prompt, kernel, str(build_dir), prior_script=prior_script, image=image)
            pkg = CodegenPackage(
                mode=mode, title=robot.title, param_schema=robot.param_schema,
                script=robot.script, prompt=prompt, notes=robot.notes,
            )
            res = robot.exec
            extra = {"urdf": robot.urdf_path, "xacro": robot.xacro_path, "zip": robot.zip_path}
            robot_info = robot.robot
        elif mode == "organic":
            from textocad.services.sdf_organic import build_organic_sdf

            scene, mesh = build_organic_sdf(prompt, build_dir=str(build_dir), prior_scene_json=prior_script, image=image)
            pkg = CodegenPackage(
                mode=mode, title=str(scene.get("title") or "Organic model"),
                param_schema=[], script=json.dumps(scene), prompt=prompt,
            )
            res = ExecResult(
                ok=True, stl_path=mesh["stl_path"], glb_path=mesh["glb_path"],
                bbox=tuple(mesh["bbox"]), volume=mesh["volume"],
            )
        else:
            pkg, res = generate_and_build(
                prompt, mode, build_dir=str(build_dir), prior_script=prior_script,
                prior_prompt=prior_prompt, image=image, extra_details=extra_details,
            )
    except provider_errors.ProviderError as exc:
        return Result(ok=False, mode=mode, error=str(exc), model=selection.model)
    except Exception as exc:
        return Result(ok=False, mode=mode, error=str(exc), model=selection.model)
    finally:
        switched = provider_errors.switched_model()
        provider_errors.stop_tracking(tracking)
        llm_models.reset_active(active)

    seconds = round(time.perf_counter() - started, 2)
    model = (switched or {}).get("to") or getattr(pkg, "escalated_to", "") or selection.model
    if pkg is None or not res.ok:
        return Result(ok=False, mode=mode, error=res.error, model=model, seconds=seconds)
    return Result(
        ok=True, mode=mode, title=pkg.title, script=pkg.script, params=pkg.param_schema,
        files=_files(res, extra), bbox=list(res.bbox) if res.bbox else None,
        volume=res.volume, model=model, seconds=seconds, parts=_parts(res), robot=robot_info,
    )


def rebuild(
    script: str,
    params: dict[str, Any],
    mode: str = "precision",
    *,
    param_schema: list[dict[str, Any]] | None = None,
    out_dir: str | Path | None = None,
) -> Result:
    """Re-run a generated script with new parameter values. No AI call."""
    build_dir = Path(out_dir or Path("builds") / uuid.uuid4().hex[:12]).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    schema = [{**p, "value": params[p["key"]]} if p.get("key") in params else p for p in (param_schema or [])]
    pkg = CodegenPackage(mode=mode, title="", param_schema=schema, script=script)
    res = run_codegen(pkg, params, build_dir=str(build_dir), timeout_s=settings.max_build_seconds)
    if not res.ok:
        return Result(ok=False, mode=mode, error=res.error)
    return Result(
        ok=True, mode=mode, script=script, params=pkg.param_schema, files=_files(res),
        bbox=list(res.bbox) if res.bbox else None, volume=res.volume, parts=_parts(res),
    )

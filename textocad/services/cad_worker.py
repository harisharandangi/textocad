from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from textocad.core.config import settings
from textocad.services.cad_repair import repair_cad_document
from textocad.services.cad_schema import CadDocument, cad_document_to_dict, param_context, resolve_number
from textocad.services.cad_operations import get_operation_support, operation_support_summary
from textocad.services.cadquery_worker import build_cadquery_artifacts
from textocad.services.urdf_worker import build_urdf_artifacts

GenerationMode = Literal["fast", "standard", "extended", "robot"]
GenerationPipeline = Literal["fast_mesh", "openscad", "cadquery", "robot_urdf"]
PreviewSource = Literal["none", "empty", "fast", "compiled"]
CompileStatus = Literal["idle", "dirty", "queued", "building", "ready", "error"]
ExportFormat = Literal[
    "step",
    "stl",
    "3mf",
    "gltf",
    "glb",
    "iges",
    "obj",
    "json",
    "urdf",
    "xacro",
    "ros_zip",
]

PIPELINE_BY_MODE: dict[GenerationMode, GenerationPipeline] = {
    "fast": "fast_mesh",
    "standard": "openscad",
    "extended": "cadquery",
    "robot": "robot_urdf",
}

EXPORTS_BY_MODE: dict[GenerationMode, list[ExportFormat]] = {
    "fast": ["stl", "glb", "json"],
    "standard": ["stl", "3mf", "glb", "json"],
    "extended": ["step", "stl", "glb", "json"],
    "robot": ["urdf", "xacro", "ros_zip", "glb", "json"],
}

BACKEND_ONLY_FORMATS: set[str] = {"step", "3mf", "urdf", "xacro", "ros_zip"}
MESH_EXPORTS: set[str] = {"stl", "obj", "glb", "gltf", "3mf"}

Vec3 = tuple[float, float, float]
Face = tuple[int, int, int]


@dataclass
class MeshData:
    vertices: list[Vec3] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)

    def extend(self, other: "MeshData") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend((a + offset, b + offset, c + offset) for a, b, c in other.faces)

    @property
    def empty(self) -> bool:
        return len(self.vertices) == 0 or len(self.faces) == 0


@dataclass
class WorkerBuildResult:
    build_id: str
    status: CompileStatus
    mode: GenerationMode
    pipeline: GenerationPipeline
    preview_source: PreviewSource
    build_dir: Path
    files: dict[ExportFormat, Path] = field(default_factory=dict)
    exports: dict[ExportFormat, str] = field(default_factory=dict)
    preview_url: str | None = None
    runtime_preview: dict[str, Any] | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _iso_now())
    updated_at: str = field(default_factory=lambda: _iso_now())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self, *, include_file_paths: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "buildId": self.build_id,
            "build_id": self.build_id,
            "status": self.status,
            "mode": self.mode,
            "generationMode": self.mode,
            "generation_mode": self.mode,
            "pipeline": self.pipeline,
            "generationPipeline": self.pipeline,
            "generation_pipeline": self.pipeline,
            "previewSource": self.preview_source,
            "preview_source": self.preview_source,
            "previewUrl": self.preview_url,
            "preview_url": self.preview_url,
            "exports": self.exports,
            "exportUrls": self.exports,
            "export_urls": self.exports,
            "runtimePreview": self.runtime_preview,
            "runtime_preview": self.runtime_preview,
            "preview": self.runtime_preview,
            "error": self.error,
            "warnings": self.warnings,
            "unsupportedOperations": self.metadata.get("unsupportedOperations", []),
            "unsupported_operations": self.metadata.get("unsupportedOperations", []),
            "operationSupportSummary": self.metadata.get("operationSupportSummary", {}),
            "operation_support_summary": self.metadata.get("operationSupportSummary", {}),
            "nativeOperations": self.metadata.get("nativeOperations", []),
            "native_operations": self.metadata.get("nativeOperations", []),
            "approximatedOperations": self.metadata.get("approximatedOperations", []),
            "approximated_operations": self.metadata.get("approximatedOperations", []),
            "skippedOperations": self.metadata.get("skippedOperations", []),
            "skipped_operations": self.metadata.get("skippedOperations", []),
            "realOpenScadUsed": bool(self.metadata.get("realOpenScadUsed", False)),
            "real_open_scad_used": bool(self.metadata.get("realOpenScadUsed", False)),
            "realCadQueryUsed": bool(self.metadata.get("realCadQueryUsed", False)),
            "real_cad_query_used": bool(self.metadata.get("realCadQueryUsed", False)),
            "safeWorker": bool(self.metadata.get("safeWorker", True)),
            "safe_worker": bool(self.metadata.get("safeWorker", True)),
            "safeFallbackUsed": bool(self.metadata.get("safeFallbackUsed", False)),
            "safe_fallback_used": bool(self.metadata.get("safeFallbackUsed", False)),
            "directLlmCodeExecution": bool(self.metadata.get("directLlmCodeExecution", False)),
            "direct_llm_code_execution": bool(self.metadata.get("directLlmCodeExecution", False)),
            "cadKernel": self.metadata.get("cadKernel", "none-yet"),
            "cad_kernel": self.metadata.get("cadKernel", "none-yet"),
            "createdAt": self.created_at,
            "created_at": self.created_at,
            "updatedAt": self.updated_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

        if include_file_paths:
            result["files"] = {
                key: _relpath_in_build(path, self.build_dir)
                for key, path in self.files.items()
            }
            result["buildDir"] = str(self.build_dir)
            result["build_dir"] = str(self.build_dir)

        return result


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value.strip())
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            return fallback
    return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    return int(round(_safe_float(value, float(fallback))))


def _as_record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    return {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _safe_text(value).lower())


def _slug(value: str, fallback: str = "textocad") -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    clean = re.sub(r"_+", "_", clean).strip("._")
    return (clean or fallback)[:120]


def _normalize_mode(value: Any) -> GenerationMode:
    clean = _safe_text(value).lower()
    if clean in {"standard", "openscad"}:
        return "standard"
    if clean in {"extended", "cadquery", "step", "cad", "brep", "b-rep"}:
        return "extended"
    if clean in {"robot", "robotics", "urdf", "xacro", "ros"}:
        return "robot"
    return "fast"


def _normalize_pipeline(value: Any, mode: GenerationMode) -> GenerationPipeline:
    clean = _safe_text(value).lower()
    if clean in {"fast_mesh", "openscad", "cadquery", "robot_urdf"}:
        return clean
    return PIPELINE_BY_MODE[mode]


def _normalize_format(value: Any) -> ExportFormat | None:
    clean = _safe_text(value).lower()
    if clean in {"zip", "roszip"}:
        clean = "ros_zip"
    allowed = {"step", "stl", "3mf", "gltf", "glb", "iges", "obj", "json", "urdf", "xacro", "ros_zip", "scad"}
    return clean if clean in allowed else None


def _get_output_root(output_root: str | Path | None = None) -> Path:
    if output_root is not None:
        root = Path(output_root)
    else:
        root = Path(
            os.getenv(
                "TEXTOCAD_BUILD_ROOT",
                os.getenv("TEXTOCAD_BUILD_DIR", "generated_builds"),
            )
        )
        if not root.is_absolute():
            root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_build_dir(build_id: str, output_root: str | Path | None = None) -> Path:
    return _get_output_root(output_root) / _slug(build_id, "build")


def _read_param(params: Sequence[Mapping[str, Any]], names: Sequence[str], fallback: float) -> float:
    candidates = [_normalize_key(name) for name in names]
    for param in params:
        key = _normalize_key(param.get("key"))
        label = _normalize_key(param.get("label"))
        if key in candidates or label in candidates:
            return _safe_float(param.get("value", param.get("defaultValue")), fallback)
    for param in params:
        key = _normalize_key(param.get("key"))
        label = _normalize_key(param.get("label"))
        for candidate in candidates:
            if candidate and (candidate in key or candidate in label or key in candidate or label in candidate):
                return _safe_float(param.get("value", param.get("defaultValue")), fallback)
    return fallback


def _read_int_param(params: Sequence[Mapping[str, Any]], names: Sequence[str], fallback: int) -> int:
    return max(0, int(round(_read_param(params, names, float(fallback)))))


def _detect_kind(prompt: str, features: Sequence[Mapping[str, Any]], code: str = "") -> str:
    text = " ".join(
        [prompt, code]
        + [_safe_text(feature.get("type")) for feature in features]
        + [_safe_text(feature.get("name")) for feature in features]
    ).lower()

    if any(word in text for word in ["robot", "urdf", "xacro", "joint", "link", "servo"]):
        return "robot"
    if "heat sink" in text or "heatsink" in text:
        return "heat_sink"
    if "helical" in text and "gear" in text:
        return "helical_gear"
    if "gear" in text or "teeth" in text:
        return "gear"
    if "flange" in text:
        return "flange"
    if "bracket" in text:
        return "bracket"
    if "drone" in text or "quadcopter" in text:
        return "drone_frame"
    if "phone stand" in text or "tablet stand" in text:
        return "phone_stand"
    if "cup" in text or "mug" in text:
        return "cup"
    if "mouse" in text:
        return "mouse"
    if "pulley" in text:
        return "pulley"
    if "bearing" in text:
        return "bearing"
    if "spring" in text:
        return "spring"
    if "fan" in text or "propeller" in text:
        return "fan"
    if "hinge" in text:
        return "hinge"
    if "clamp" in text:
        return "clamp"
    if "bolt" in text:
        return "bolt"
    if "nut" in text:
        return "nut"
    if "washer" in text:
        return "washer"
    if "sphere" in text or "ball" in text:
        return "sphere"
    if "cone" in text or "nozzle" in text:
        return "cone"
    if "cylinder" in text or "tube" in text or "pipe" in text or "bushing" in text:
        return "cylinder"
    if "box" in text or "cube" in text or "enclosure" in text:
        return "box"
    return "generic"


def _primitive(
    primitive_id: str,
    kind: str,
    *,
    position: Vec3 = (0.0, 0.0, 0.0),
    rotation: Vec3 = (0.0, 0.0, 0.0),
    material: str = "steel",
    **values: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": primitive_id,
        "kind": kind,
        "position": [position[0], position[1], position[2]],
        "rotation": [rotation[0], rotation[1], rotation[2]],
        "material": material,
        "visible": True,
    }
    item.update(values)
    return item


def _payload_cad_document(payload: Mapping[str, Any], mode: GenerationMode) -> tuple[CadDocument, list[str]]:
    raw_document = (
        payload.get("cadDocument")
        or payload.get("cad_document")
        or payload.get("canonicalModel")
        or payload.get("canonical_model")
        or payload
    )
    return repair_cad_document(
        raw_document,
        prompt=_safe_text(payload.get("prompt")),
        mode="robot" if mode == "robot" else "part",
    )


def _resolve_vec(value: Any, params: Mapping[str, float], fallback: Vec3 = (0.0, 0.0, 0.0)) -> Vec3:
    if not isinstance(value, list) or len(value) < 3:
        return fallback
    return (
        resolve_number(value[0], params, fallback[0]),
        resolve_number(value[1], params, fallback[1]),
        resolve_number(value[2], params, fallback[2]),
    )


def _canonical_position_to_runtime(position: Vec3) -> Vec3:

    return (position[0], position[2], position[1])


def _runtime_preview_from_document(document: CadDocument, mode: GenerationMode, warnings: list[str] | None = None) -> dict[str, Any]:
    params = param_context(document.params)
    primitives: list[dict[str, Any]] = []
    unsupported: list[str] = []
    skipped: list[str] = []
    runtime_warnings = list(warnings or [])

    for op in document.operations:
        if not op.enabled or not op.visible:
            continue
        position = _canonical_position_to_runtime(_resolve_vec(op.transform.translate, params))
        rotation = _resolve_vec(op.transform.rotate, params)

        if op.type == "box":
            size = op.params.get("size")
            sx, sy, sz = _resolve_vec(size, params, (20, 20, 20))
            primitives.append(
                _primitive(op.id, "box", position=position, rotation=rotation, width=sx, depth=sy, height=sz, material="cad-blue")
            )
        elif op.type == "cylinder":
            height = resolve_number(op.params.get("height", op.params.get("length")), params, 20)
            diameter = resolve_number(op.params.get("diameter"), params, 0)
            radius = resolve_number(op.params.get("radius"), params, diameter / 2 if diameter else 10)
            primitives.append(
                _primitive(op.id, "cylinder", position=position, rotation=rotation, outerRadius=radius, height=height, segments=96, material="cad-blue")
            )
        elif op.type == "sphere":
            radius = resolve_number(op.params.get("radius"), params, resolve_number(op.params.get("diameter"), params, 20) / 2)
            primitives.append(_primitive(op.id, "sphere", position=position, rotation=rotation, radius=radius, material="cad-blue"))
        elif op.type == "cone":
            height = resolve_number(op.params.get("height"), params, 30)
            radius = resolve_number(op.params.get("bottom_radius", op.params.get("radius")), params, 15)
            top_radius = resolve_number(op.params.get("top_radius"), params, 5)
            primitives.append(_primitive(op.id, "cone", position=position, rotation=rotation, radius=radius, topRadius=top_radius, height=height, material="cad-blue"))
        elif op.type in {"gear_spur_approx", "gear_helical_approx"}:
            outer = resolve_number(op.params.get("outer_diameter"), params, 60)
            bore = resolve_number(op.params.get("bore_diameter"), params, 0)
            face_width = resolve_number(op.params.get("face_width", op.params.get("width")), params, 12)
            teeth = max(6, int(round(resolve_number(op.params.get("tooth_count"), params, 24))))
            primitives.append(_primitive(op.id, "gear", position=position, rotation=rotation, outerRadius=outer / 2, innerRadius=bore / 2, height=face_width, teeth=teeth, material="cad-blue"))
        elif op.type in {"wedge", "triangular_prism"}:
            length = resolve_number(op.params.get("length"), params, 30)
            width = resolve_number(op.params.get("width"), params, 8)
            height = resolve_number(op.params.get("height"), params, 30)
            primitives.append(_primitive(op.id, "triangular-prism", position=position, rotation=rotation, width=length, depth=width, height=height, material="steel"))
        elif op.type == "hole":
            hole_position = _canonical_position_to_runtime(_resolve_vec(op.params.get("position"), params, position))
            diameter = resolve_number(op.params.get("diameter"), params, 5)
            depth = resolve_number(op.params.get("depth"), params, 20)
            primitives.append(_primitive(op.id, "hole-marker", position=hole_position, radius=diameter / 2, height=depth, material="warning"))
        elif op.type == "hole_pattern":
            positions = op.params.get("positions") if isinstance(op.params.get("positions"), list) else []
            diameter = resolve_number(op.params.get("diameter"), params, 5)
            depth = resolve_number(op.params.get("depth"), params, 20)
            for index, item in enumerate(positions):
                primitives.append(
                    _primitive(
                        f"{op.id}-{index + 1}",
                        "hole-marker",
                        position=_canonical_position_to_runtime(_resolve_vec(item, params)),
                        radius=diameter / 2,
                        height=depth,
                        material="warning",
                    )
                )
        elif op.type in {"fillet", "chamfer"}:
            skipped.append(op.type)
            runtime_warnings.append(f"{op.id}: {op.type} is skipped in fast preview; backend CadQuery can attempt it.")
        elif op.type == "unsupported":
            unsupported.append(str(op.metadata.get("unsupportedType") or op.id))
        else:
            support = get_operation_support(op.type, "fast")
            if support == "unsupported":
                unsupported.append(op.type)
            elif support == "fallback_skip":
                skipped.append(op.type)
                runtime_warnings.append(f"{op.id}: {op.type} was skipped in fast preview.")

    if not primitives:
        primitives.append(_primitive("safe-fallback-body", "box", width=20, depth=20, height=20, material="warning"))

    support_summary = operation_support_summary(document.operations, "fast")
    support_summary["unsupported"] = sorted(set(support_summary.get("unsupported", []) + unsupported))
    support_summary["skipped"] = sorted(set(support_summary.get("skipped", []) + skipped))

    metadata = {
        **document.metadata,
        "source": "canonical_cad_json",
        "previewAccuracy": "approximate",
        "operationSupportSummary": support_summary,
        "nativeOperations": support_summary.get("native", []),
        "approximatedOperations": support_summary.get("approximate", []),
        "skippedOperations": support_summary.get("skipped", []),
        "unsupportedOperations": sorted(set(unsupported + list(document.metadata.get("unsupportedOperations", [])))),
        "warnings": runtime_warnings,
    }
    return {
        "label": document.title,
        "kind": document.metadata.get("intent") or document.modelType,
        "units": document.units,
        "mode": mode,
        "pipeline": PIPELINE_BY_MODE[mode],
        "primitives": primitives,
        "metadata": metadata,
    }


def create_runtime_preview(payload: Mapping[str, Any]) -> dict[str, Any]:
    project_name = _safe_text(payload.get("projectName") or payload.get("project_name"), "TextoCAD Build")
    prompt = _safe_text(payload.get("prompt"))
    code = _safe_text(payload.get("code"))
    params = [_as_record(item) for item in _as_list(payload.get("params"))]
    features = [_as_record(item) for item in _as_list(payload.get("features"))]
    mode = _normalize_mode(payload.get("mode") or payload.get("generationMode") or payload.get("generation_mode"))

    if isinstance(payload.get("cadDocument") or payload.get("cad_document") or payload.get("operations"), (Mapping, list)):
        document, repair_warnings = _payload_cad_document(payload, mode)
        return _runtime_preview_from_document(document, mode, repair_warnings)

    kind = _detect_kind(prompt, features, code)

    primitives: list[dict[str, Any]] = []

    width = _read_param(params, ["width", "baseWidth", "flangeWidth"], 80)
    depth = _read_param(params, ["depth", "baseDepth"], 50)
    height = _read_param(params, ["height", "baseHeight", "thickness"], 20)
    length = _read_param(params, ["length", "armLength"], max(width, 100))
    outer_diameter = _read_param(params, ["outerDiameter", "outer_diameter", "diameter", "outsideDiameter"], 60)
    inner_diameter = _read_param(params, ["innerDiameter", "inner_diameter", "boreDiameter", "bore_diameter", "holeDiameter"], 0)
    radius = _read_param(params, ["radius"], outer_diameter / 2)
    face_width = _read_param(params, ["faceWidth", "face_width", "thickness", "width"], 15)
    tooth_count = max(6, _read_int_param(params, ["toothCount", "tooth_count", "teeth", "numTeeth"], 24))
    hole_count = max(0, _read_int_param(params, ["holeCount", "boltHoleCount", "holes"], 4))
    fin_count = max(2, _read_int_param(params, ["finCount", "fins"], 8))

    if kind in {"gear", "helical_gear"}:
        primitives.append(
            _primitive(
                "gear-body",
                "gear",
                outerRadius=max(outer_diameter / 2, 2),
                innerRadius=max(inner_diameter / 2, 0),
                height=max(face_width, 1),
                teeth=tooth_count,
                helixAngle=_read_param(params, ["helixAngle", "helix_angle"], 0),
                pressureAngle=_read_param(params, ["pressureAngle", "pressure_angle"], 20),
                segments=max(tooth_count * 2, 32),
                material="cad-blue",
            )
        )
    elif kind == "heat_sink":
        base_w = _read_param(params, ["baseWidth", "width"], 90)
        base_d = _read_param(params, ["baseDepth", "depth"], 60)
        base_h = _read_param(params, ["baseThickness", "baseHeight", "thickness"], 8)
        fin_h = _read_param(params, ["finHeight", "height"], 32)
        fin_t = _read_param(params, ["finThickness"], 2)
        primitives.append(_primitive("base", "box", width=base_w, depth=base_d, height=base_h, material="aluminum"))
        spacing = base_w / max(fin_count, 1)
        for index in range(fin_count):
            x = -base_w / 2 + spacing * (index + 0.5)
            primitives.append(
                _primitive(
                    f"fin-{index+1}",
                    "box",
                    position=(x, base_h / 2 + fin_h / 2, 0),
                    width=fin_t,
                    depth=base_d * 0.92,
                    height=fin_h,
                    material="aluminum",
                )
            )
    elif kind == "flange":
        primitives.append(
            _primitive(
                "flange-body",
                "cylinder",
                outerRadius=max(outer_diameter / 2, 10),
                innerRadius=max(inner_diameter / 2, 0),
                height=max(height, 8),
                segments=96,
                material="steel",
            )
        )
        bolt_circle = _read_param(params, ["boltCircleDiameter", "bolt_circle_diameter"], outer_diameter * 0.75)
        bolt_hole = _read_param(params, ["boltHoleDiameter", "bolt_hole_diameter"], max(3, outer_diameter * 0.06))
        for index in range(max(hole_count, 4)):
            angle = (index / max(hole_count, 4)) * math.tau
            primitives.append(
                _primitive(
                    f"bolt-hole-{index+1}",
                    "hole-marker",
                    position=(math.cos(angle) * bolt_circle / 2, 0, math.sin(angle) * bolt_circle / 2),
                    radius=bolt_hole / 2,
                    height=height * 1.2,
                    material="warning",
                )
            )
    elif kind == "robot":
        base_w = _read_param(params, ["baseWidth"], 120)
        arm_l = _read_param(params, ["armLength", "upperArmLength"], 220)
        forearm_l = _read_param(params, ["forearmLength"], arm_l * 0.75)
        link_t = _read_param(params, ["linkThickness", "armThickness"], 30)
        shoulder = math.radians(_read_param(params, ["shoulderAngle"], 0))
        elbow = math.radians(_read_param(params, ["elbowAngle"], 0))
        primitives.append(_primitive("base-link", "box", width=base_w, depth=base_w * 0.7, height=35, material="dark"))
        upper_center = (math.cos(shoulder) * arm_l / 2, 60 + math.sin(shoulder) * arm_l / 2, 0)
        primitives.append(
            _primitive(
                "upper-arm",
                "box",
                position=upper_center,
                rotation=(0, 0, shoulder),
                width=arm_l,
                depth=link_t,
                height=link_t,
                material="cad-blue",
            )
        )
        joint_x = math.cos(shoulder) * arm_l
        joint_y = 60 + math.sin(shoulder) * arm_l
        total_angle = shoulder + elbow
        fore_center = (joint_x + math.cos(total_angle) * forearm_l / 2, joint_y + math.sin(total_angle) * forearm_l / 2, 0)
        primitives.append(
            _primitive(
                "forearm",
                "box",
                position=fore_center,
                rotation=(0, 0, total_angle),
                width=forearm_l,
                depth=link_t * 0.85,
                height=link_t * 0.85,
                material="orange",
            )
        )
        primitives.append(_primitive("shoulder-joint", "sphere", position=(0, 60, 0), radius=link_t * 0.75, material="joint"))
        primitives.append(_primitive("elbow-joint", "sphere", position=(joint_x, joint_y, 0), radius=link_t * 0.6, material="joint"))
    elif kind == "drone_frame":
        arm = _read_param(params, ["armLength", "length"], 180)
        arm_w = _read_param(params, ["armWidth", "width"], 18)
        motor = _read_param(params, ["motorMountDiameter", "motorDiameter"], 28)
        primitives.append(_primitive("center-plate", "box", width=70, depth=70, height=8, material="carbon"))
        for index, angle in enumerate([0, math.pi / 2, math.pi, math.pi * 1.5]):
            x = math.cos(angle) * arm / 2
            z = math.sin(angle) * arm / 2
            primitives.append(_primitive(f"arm-{index+1}", "box", position=(x, 0, z), rotation=(0, -angle, 0), width=arm, depth=arm_w, height=8, material="carbon"))
            primitives.append(_primitive(f"motor-mount-{index+1}", "cylinder", position=(math.cos(angle) * arm, 0, math.sin(angle) * arm), radius=motor / 2, height=8, segments=48, material="dark"))
    elif kind == "phone_stand":
        stand_w = _read_param(params, ["width"], 80)
        stand_h = _read_param(params, ["height"], 110)
        angle = math.radians(_read_param(params, ["backAngle", "angle"], 65))
        primitives.append(_primitive("base", "box", width=stand_w, depth=70, height=8, material="plastic"))
        primitives.append(_primitive("back", "box", position=(0, stand_h / 2 * math.sin(angle), -stand_h / 2 * math.cos(angle)), rotation=(angle, 0, 0), width=stand_w, depth=8, height=stand_h, material="plastic"))
        primitives.append(_primitive("lip", "box", position=(0, 8, 32), width=stand_w, depth=10, height=12, material="plastic"))
    elif kind == "cup":
        primitives.append(_primitive("cup-body", "cylinder", outerRadius=max(outer_diameter / 2, 20), innerRadius=max(inner_diameter / 2, outer_diameter / 2 - 3), height=max(height, 70), segments=96, material="ceramic"))
        primitives.append(_primitive("handle", "torus", position=(outer_diameter / 2 + 8, height * 0.05, 0), rotation=(math.pi / 2, 0, 0), radius=max(height * 0.22, 16), tubeRadius=4, material="ceramic"))
    elif kind == "mouse":
        primitives.append(_primitive("mouse-body", "sphere", width=max(width, 95), depth=max(depth, 55), height=max(height, 28), material="plastic"))
        primitives.append(_primitive("scroll", "cylinder", position=(0, 16, -8), rotation=(math.pi / 2, 0, 0), radius=4, height=10, material="dark"))
    elif kind == "pulley":
        primitives.append(_primitive("pulley-body", "cylinder", outerRadius=max(outer_diameter / 2, 20), innerRadius=max(inner_diameter / 2, 4), height=max(face_width, 15), segments=96, material="steel"))
        primitives.append(_primitive("groove", "torus", radius=max(outer_diameter / 2 - 3, 5), tubeRadius=2, material="dark"))
    elif kind == "bearing":
        primitives.append(_primitive("outer-race", "cylinder", outerRadius=max(outer_diameter / 2, 20), innerRadius=max(inner_diameter / 2, outer_diameter / 3), height=max(height, 10), segments=96, material="steel"))
        ball_count = max(6, _read_int_param(params, ["ballCount", "balls"], 10))
        race_radius = (outer_diameter + max(inner_diameter, outer_diameter * 0.45)) / 4
        ball_radius = max((outer_diameter - max(inner_diameter, outer_diameter * 0.45)) / 10, 2)
        for index in range(ball_count):
            angle = index / ball_count * math.tau
            primitives.append(_primitive(f"ball-{index+1}", "sphere", position=(math.cos(angle) * race_radius, 0, math.sin(angle) * race_radius), radius=ball_radius, material="steel"))
    elif kind == "spring":
        primitives.append(_primitive("spring", "spring", radius=max(radius, 20), tubeRadius=_read_param(params, ["wireDiameter", "tubeRadius"], 2), height=max(height, 80), turns=max(2, _read_int_param(params, ["turns", "coils"], 6)), material="steel"))
    elif kind == "fan":
        blade_count = max(2, _read_int_param(params, ["bladeCount", "blades"], 5))
        primitives.append(_primitive("hub", "cylinder", radius=max(radius * 0.18, 8), height=10, segments=48, material="dark"))
        for index in range(blade_count):
            angle = index / blade_count * math.tau
            primitives.append(_primitive(f"blade-{index+1}", "box", position=(math.cos(angle) * radius * 0.48, 0, math.sin(angle) * radius * 0.48), rotation=(0, -angle, 0.25), width=radius * 0.85, depth=max(radius * 0.16, 8), height=5, material="plastic"))
    elif kind == "hinge":
        primitives.append(_primitive("leaf-left", "box", position=(-35, 0, 0), width=60, depth=40, height=3, material="steel"))
        primitives.append(_primitive("leaf-right", "box", position=(35, 0, 0), width=60, depth=40, height=3, material="steel"))
        primitives.append(_primitive("pin", "cylinder", rotation=(math.pi / 2, 0, 0), radius=5, height=85, segments=48, material="steel"))
    elif kind == "clamp":
        primitives.append(_primitive("clamp-body", "box", width=width, depth=depth, height=height, material="steel"))
        primitives.append(_primitive("opening", "hole-marker", position=(0, 0, 0), radius=min(width, depth) * 0.25, height=height * 1.2, material="warning"))
    elif kind == "bolt":
        primitives.append(_primitive("shaft", "cylinder", radius=max(radius * 0.35, 4), height=max(length, 50), segments=48, material="steel"))
        primitives.append(_primitive("head", "hex-prism", position=(0, length / 2 + 5, 0), radius=max(radius, 10), height=10, material="steel"))
    elif kind == "nut":
        primitives.append(_primitive("nut", "hex-prism", radius=max(outer_diameter / 2, 12), innerRadius=max(inner_diameter / 2, 5), height=max(height, 10), material="steel"))
    elif kind == "washer":
        primitives.append(_primitive("washer", "cylinder", outerRadius=max(outer_diameter / 2, 12), innerRadius=max(inner_diameter / 2, 5), height=max(height, 2), segments=72, material="steel"))
    elif kind == "sphere":
        primitives.append(_primitive("sphere", "sphere", radius=max(radius, outer_diameter / 2, 10), material="cad-blue"))
    elif kind == "cone":
        primitives.append(_primitive("cone", "cone", radius=max(radius, outer_diameter / 2, 10), topRadius=_read_param(params, ["topRadius", "exitRadius"], 0), height=max(height, 40), segments=72, material="cad-blue"))
    elif kind == "cylinder":
        primitives.append(_primitive("cylinder", "cylinder", outerRadius=max(outer_diameter / 2, radius, 10), innerRadius=max(inner_diameter / 2, 0), height=max(height, length, 30), segments=72, material="cad-blue"))
    elif kind == "box":
        primitives.append(_primitive("box", "box", width=max(width, 20), depth=max(depth, 20), height=max(height, 20), material="cad-blue"))
    elif kind == "bracket":
        primitives.append(_primitive("base-plate", "box", width=max(width, 80), depth=max(depth, 50), height=8, material="steel"))
        primitives.append(_primitive("vertical-plate", "box", position=(0, max(height, 60) / 2, -max(depth, 50) / 2 + 4), width=max(width, 80), depth=8, height=max(height, 60), material="steel"))
    else:
        primitives.append(_primitive("generic-body", "box", width=max(width, 40), depth=max(depth, 40), height=max(height, 20), material="cad-blue"))

    return {
        "label": project_name,
        "kind": kind,
        "units": "mm",
        "mode": mode,
        "primitives": primitives,
        "size": {
            "width": width,
            "depth": depth,
            "height": height,
            "diameter": outer_diameter,
        },
        "metadata": {
            "source": "cad_worker",
            "prompt": prompt,
            "featureCount": len(features),
            "paramCount": len(params),
        },
    }


def _rotation_matrix(rotation: Sequence[float] | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    rx, ry, rz = (0.0, 0.0, 0.0)
    if rotation and len(rotation) >= 3:
        rx, ry, rz = (_safe_float(rotation[0]), _safe_float(rotation[1]), _safe_float(rotation[2]))
        if max(abs(rx), abs(ry), abs(rz)) > math.tau * 1.25:
            rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    return (
        (cy * cz, -cy * sz, sy),
        (sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy),
        (-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy),
    )


def _transform_vertex(vertex: Vec3, position: Sequence[float] | None, rotation: Sequence[float] | None) -> Vec3:
    matrix = _rotation_matrix(rotation)
    x, y, z = vertex
    tx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z
    ty = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z
    tz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
    px, py, pz = (0.0, 0.0, 0.0)
    if position and len(position) >= 3:
        px, py, pz = _safe_float(position[0]), _safe_float(position[1]), _safe_float(position[2])
    return (tx + px, ty + py, tz + pz)


def _apply_transform(mesh: MeshData, primitive: Mapping[str, Any]) -> MeshData:
    position = primitive.get("position") if isinstance(primitive.get("position"), list) else None
    rotation = primitive.get("rotation") if isinstance(primitive.get("rotation"), list) else None
    return MeshData(vertices=[_transform_vertex(vertex, position, rotation) for vertex in mesh.vertices], faces=list(mesh.faces))


def _box_mesh(width: float, depth: float, height: float) -> MeshData:
    w, d, h = width / 2, depth / 2, height / 2
    vertices = [
        (-w, -h, -d), (w, -h, -d), (w, h, -d), (-w, h, -d),
        (-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return MeshData(vertices, faces)


def _cylinder_mesh(radius: float, height: float, segments: int = 64, inner_radius: float = 0.0) -> MeshData:
    segments = max(12, min(int(segments), 256))
    radius = max(radius, 0.1)
    height = max(height, 0.1)
    inner_radius = max(0.0, min(inner_radius, radius - 0.05))
    y0, y1 = -height / 2, height / 2
    vertices: list[Vec3] = []
    faces: list[Face] = []

    if inner_radius <= 0:
        bottom_center = len(vertices); vertices.append((0, y0, 0))
        top_center = len(vertices); vertices.append((0, y1, 0))
        bottom: list[int] = []
        top: list[int] = []
        for i in range(segments):
            angle = i / segments * math.tau
            x, z = math.cos(angle) * radius, math.sin(angle) * radius
            bottom.append(len(vertices)); vertices.append((x, y0, z))
            top.append(len(vertices)); vertices.append((x, y1, z))
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((bottom_center, bottom[j], bottom[i]))
            faces.append((top_center, top[i], top[j]))
            faces.append((bottom[i], bottom[j], top[j]))
            faces.append((bottom[i], top[j], top[i]))
    else:
        outer_bottom: list[int] = []
        outer_top: list[int] = []
        inner_bottom: list[int] = []
        inner_top: list[int] = []
        for i in range(segments):
            angle = i / segments * math.tau
            co, si = math.cos(angle), math.sin(angle)
            outer_bottom.append(len(vertices)); vertices.append((co * radius, y0, si * radius))
            outer_top.append(len(vertices)); vertices.append((co * radius, y1, si * radius))
            inner_bottom.append(len(vertices)); vertices.append((co * inner_radius, y0, si * inner_radius))
            inner_top.append(len(vertices)); vertices.append((co * inner_radius, y1, si * inner_radius))
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((outer_bottom[i], outer_bottom[j], outer_top[j]))
            faces.append((outer_bottom[i], outer_top[j], outer_top[i]))
            faces.append((inner_bottom[j], inner_bottom[i], inner_top[i]))
            faces.append((inner_bottom[j], inner_top[i], inner_top[j]))
            faces.append((outer_top[i], outer_top[j], inner_top[j]))
            faces.append((outer_top[i], inner_top[j], inner_top[i]))
            faces.append((outer_bottom[j], outer_bottom[i], inner_bottom[i]))
            faces.append((outer_bottom[j], inner_bottom[i], inner_bottom[j]))
    return MeshData(vertices, faces)


def _gear_mesh(outer_radius: float, inner_radius: float, height: float, teeth: int) -> MeshData:
    teeth = max(6, min(teeth, 180))
    segments = teeth * 2
    root_radius = outer_radius * 0.86
    y0, y1 = -height / 2, height / 2
    inner_radius = max(0, min(inner_radius, root_radius - 0.05))
    vertices: list[Vec3] = []
    faces: list[Face] = []
    outer_bottom: list[int] = []
    outer_top: list[int] = []
    inner_bottom: list[int] = []
    inner_top: list[int] = []
    for i in range(segments):
        radius = outer_radius if i % 2 == 0 else root_radius
        angle = i / segments * math.tau
        co, si = math.cos(angle), math.sin(angle)
        outer_bottom.append(len(vertices)); vertices.append((co * radius, y0, si * radius))
        outer_top.append(len(vertices)); vertices.append((co * radius, y1, si * radius))
        if inner_radius > 0:
            inner_bottom.append(len(vertices)); vertices.append((co * inner_radius, y0, si * inner_radius))
            inner_top.append(len(vertices)); vertices.append((co * inner_radius, y1, si * inner_radius))
    if inner_radius <= 0:
        bottom_center = len(vertices); vertices.append((0, y0, 0))
        top_center = len(vertices); vertices.append((0, y1, 0))
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((outer_bottom[i], outer_bottom[j], outer_top[j]))
        faces.append((outer_bottom[i], outer_top[j], outer_top[i]))
        if inner_radius > 0:
            faces.append((inner_bottom[j], inner_bottom[i], inner_top[i]))
            faces.append((inner_bottom[j], inner_top[i], inner_top[j]))
            faces.append((outer_top[i], outer_top[j], inner_top[j]))
            faces.append((outer_top[i], inner_top[j], inner_top[i]))
            faces.append((outer_bottom[j], outer_bottom[i], inner_bottom[i]))
            faces.append((outer_bottom[j], inner_bottom[i], inner_bottom[j]))
        else:
            faces.append((bottom_center, outer_bottom[j], outer_bottom[i]))
            faces.append((top_center, outer_top[i], outer_top[j]))
    return MeshData(vertices, faces)


def _sphere_mesh(radius: float, width: float | None = None, depth: float | None = None, height: float | None = None, segments: int = 32) -> MeshData:
    lat_segments = max(8, segments // 2)
    lon_segments = max(12, segments)
    sx = (width / 2) if width else radius
    sy = (height / 2) if height else radius
    sz = (depth / 2) if depth else radius
    vertices: list[Vec3] = []
    faces: list[Face] = []
    for lat in range(lat_segments + 1):
        theta = lat / lat_segments * math.pi
        for lon in range(lon_segments):
            phi = lon / lon_segments * math.tau
            vertices.append((math.sin(theta) * math.cos(phi) * sx, math.cos(theta) * sy, math.sin(theta) * math.sin(phi) * sz))
    for lat in range(lat_segments):
        for lon in range(lon_segments):
            a = lat * lon_segments + lon
            b = lat * lon_segments + (lon + 1) % lon_segments
            c = (lat + 1) * lon_segments + (lon + 1) % lon_segments
            d = (lat + 1) * lon_segments + lon
            faces.append((a, b, c))
            faces.append((a, c, d))
    return MeshData(vertices, faces)


def _cone_mesh(radius: float, top_radius: float, height: float, segments: int = 48) -> MeshData:
    segments = max(12, min(segments, 128))
    y0, y1 = -height / 2, height / 2
    vertices: list[Vec3] = []
    bottom_center = len(vertices); vertices.append((0, y0, 0))
    top_center = len(vertices); vertices.append((0, y1, 0))
    bottom: list[int] = []
    top: list[int] = []
    for i in range(segments):
        angle = i / segments * math.tau
        co, si = math.cos(angle), math.sin(angle)
        bottom.append(len(vertices)); vertices.append((co * radius, y0, si * radius))
        top.append(len(vertices)); vertices.append((co * top_radius, y1, si * top_radius))
    faces: list[Face] = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((bottom_center, bottom[j], bottom[i]))
        if top_radius > 0:
            faces.append((top_center, top[i], top[j]))
        faces.append((bottom[i], bottom[j], top[j]))
        faces.append((bottom[i], top[j], top[i]))
    return MeshData(vertices, faces)


def _triangular_prism_mesh(length: float, depth: float, height: float) -> MeshData:
    length = max(length, 0.1)
    depth = max(depth, 0.1)
    height = max(height, 0.1)
    d = depth / 2
    vertices: list[Vec3] = [
        (0, 0, -d),
        (length, 0, -d),
        (0, height, -d),
        (0, 0, d),
        (length, 0, d),
        (0, height, d),
    ]
    faces: list[Face] = [
        (0, 1, 2),
        (3, 5, 4),
        (0, 3, 4),
        (0, 4, 1),
        (0, 2, 5),
        (0, 5, 3),
        (1, 4, 5),
        (1, 5, 2),
    ]
    return MeshData(vertices, faces)


def _torus_mesh(radius: float, tube_radius: float, radial_segments: int = 64, tube_segments: int = 12) -> MeshData:
    vertices: list[Vec3] = []
    faces: list[Face] = []
    for i in range(radial_segments):
        phi = i / radial_segments * math.tau
        for j in range(tube_segments):
            theta = j / tube_segments * math.tau
            x = (radius + tube_radius * math.cos(theta)) * math.cos(phi)
            y = tube_radius * math.sin(theta)
            z = (radius + tube_radius * math.cos(theta)) * math.sin(phi)
            vertices.append((x, y, z))
    for i in range(radial_segments):
        for j in range(tube_segments):
            a = i * tube_segments + j
            b = i * tube_segments + (j + 1) % tube_segments
            c = ((i + 1) % radial_segments) * tube_segments + (j + 1) % tube_segments
            d = ((i + 1) % radial_segments) * tube_segments + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return MeshData(vertices, faces)


def _spring_mesh(radius: float, tube_radius: float, height: float, turns: int) -> MeshData:
    path_segments = max(32, turns * 28)
    tube_segments = 8
    vertices: list[Vec3] = []
    faces: list[Face] = []
    for i in range(path_segments + 1):
        t = i / path_segments
        phi = t * turns * math.tau
        center = (math.cos(phi) * radius, -height / 2 + t * height, math.sin(phi) * radius)
        radial = (math.cos(phi), 0, math.sin(phi))
        binormal = (0, 1, 0)
        for j in range(tube_segments):
            theta = j / tube_segments * math.tau
            x = center[0] + tube_radius * (math.cos(theta) * radial[0] + math.sin(theta) * binormal[0])
            y = center[1] + tube_radius * (math.cos(theta) * radial[1] + math.sin(theta) * binormal[1])
            z = center[2] + tube_radius * (math.cos(theta) * radial[2] + math.sin(theta) * binormal[2])
            vertices.append((x, y, z))
    for i in range(path_segments):
        for j in range(tube_segments):
            a = i * tube_segments + j
            b = i * tube_segments + (j + 1) % tube_segments
            c = (i + 1) * tube_segments + (j + 1) % tube_segments
            d = (i + 1) * tube_segments + j
            faces.append((a, b, c))
            faces.append((a, c, d))
    return MeshData(vertices, faces)


def _primitive_to_mesh(primitive: Mapping[str, Any]) -> MeshData:
    kind = _safe_text(primitive.get("kind"), "box").lower()
    if kind in {"box", "plate", "open-box", "cube", "block"}:
        mesh = _box_mesh(_safe_float(primitive.get("width"), 20), _safe_float(primitive.get("depth"), 20), _safe_float(primitive.get("height"), _safe_float(primitive.get("thickness"), 20)))
    elif kind in {"cylinder", "ring", "washer", "pipe", "tube"}:
        outer = _safe_float(primitive.get("outerRadius"), _safe_float(primitive.get("radius"), 10))
        inner = _safe_float(primitive.get("innerRadius"), 0)
        mesh = _cylinder_mesh(outer, _safe_float(primitive.get("height"), 20), _safe_int(primitive.get("segments"), 64), inner)
    elif kind in {"gear", "helical_gear", "spur_gear"}:
        mesh = _gear_mesh(_safe_float(primitive.get("outerRadius"), 30), _safe_float(primitive.get("innerRadius"), 0), _safe_float(primitive.get("height"), 12), _safe_int(primitive.get("teeth"), 24))
    elif kind in {"sphere", "ball", "ellipsoid"}:
        mesh = _sphere_mesh(_safe_float(primitive.get("radius"), 10), _safe_float(primitive.get("width"), 0) or None, _safe_float(primitive.get("depth"), 0) or None, _safe_float(primitive.get("height"), 0) or None)
    elif kind == "cone":
        mesh = _cone_mesh(_safe_float(primitive.get("radius"), 10), _safe_float(primitive.get("topRadius"), 0), _safe_float(primitive.get("height"), 30), _safe_int(primitive.get("segments"), 48))
    elif kind in {"triangular-prism", "triangular_prism", "wedge"}:
        mesh = _triangular_prism_mesh(_safe_float(primitive.get("width"), 30), _safe_float(primitive.get("depth"), 8), _safe_float(primitive.get("height"), 30))
    elif kind == "torus":
        mesh = _torus_mesh(_safe_float(primitive.get("radius"), 20), _safe_float(primitive.get("tubeRadius"), 3))
    elif kind in {"hex-prism", "hex", "nut", "bolt-head"}:
        mesh = _cylinder_mesh(_safe_float(primitive.get("radius"), 12), _safe_float(primitive.get("height"), 10), 6, _safe_float(primitive.get("innerRadius"), 0))
    elif kind == "spring":
        mesh = _spring_mesh(_safe_float(primitive.get("radius"), 20), _safe_float(primitive.get("tubeRadius"), 2), _safe_float(primitive.get("height"), 80), _safe_int(primitive.get("turns"), 6))
    elif kind == "hole-marker":
        mesh = MeshData()
    else:
        mesh = _box_mesh(_safe_float(primitive.get("width"), 20), _safe_float(primitive.get("depth"), 20), _safe_float(primitive.get("height"), 20))
    return _apply_transform(mesh, primitive)


def mesh_from_runtime_preview(preview: Mapping[str, Any]) -> MeshData:
    result = MeshData()
    for primitive in _as_list(preview.get("primitives")):
        if not isinstance(primitive, Mapping):
            continue
        result.extend(_primitive_to_mesh(primitive))
    if result.empty:
        result.extend(_box_mesh(20, 20, 20))
    return result


def _normal(a: Vec3, b: Vec3, c: Vec3) -> Vec3:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def export_mesh_as_stl(mesh: MeshData, name: str = "textocad") -> str:
    lines = [f"solid {_slug(name)}"]
    for face in mesh.faces:
        a, b, c = mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]
        nx, ny, nz = _normal(a, b, c)
        lines.append(f"  facet normal {nx:.8g} {ny:.8g} {nz:.8g}")
        lines.append("    outer loop")
        lines.append(f"      vertex {a[0]:.8g} {a[1]:.8g} {a[2]:.8g}")
        lines.append(f"      vertex {b[0]:.8g} {b[1]:.8g} {b[2]:.8g}")
        lines.append(f"      vertex {c[0]:.8g} {c[1]:.8g} {c[2]:.8g}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {_slug(name)}")
    return "\n".join(lines) + "\n"


def export_mesh_as_obj(mesh: MeshData, name: str = "textocad") -> str:
    lines = [f"# TextoCAD OBJ export", f"o {_slug(name)}"]
    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.8g} {y:.8g} {z:.8g}")
    for a, b, c in mesh.faces:
        lines.append(f"f {a + 1} {b + 1} {c + 1}")
    return "\n".join(lines) + "\n"


def _pad4(data: bytes, pad: bytes = b" ") -> bytes:
    extra = (-len(data)) % 4
    return data + pad * extra


def export_mesh_as_glb(mesh: MeshData, name: str = "textocad") -> bytes:
    position_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in mesh.vertices)
    index_bytes = b"".join(struct.pack("<III", *face) for face in mesh.faces)
    position_offset = 0
    index_offset = len(position_bytes)
    bin_blob = _pad4(position_bytes + index_bytes, b"\x00")

    xs = [v[0] for v in mesh.vertices] or [0]
    ys = [v[1] for v in mesh.vertices] or [0]
    zs = [v[2] for v in mesh.vertices] or [0]

    gltf = {
        "asset": {"version": "2.0", "generator": "TextoCAD cad_worker"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.52, 1.0, 1.0], "metallicFactor": 0.1, "roughnessFactor": 0.55}}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": position_offset, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": index_offset, "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(mesh.vertices), "type": "VEC3", "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 1, "componentType": 5125, "count": len(mesh.faces) * 3, "type": "SCALAR"},
        ],
    }
    json_blob = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    return b"glTF" + struct.pack("<II", 2, total_length) + struct.pack("<I4s", len(json_blob), b"JSON") + json_blob + struct.pack("<I4s", len(bin_blob), b"BIN\x00") + bin_blob


def export_mesh_as_gltf(mesh: MeshData, name: str = "textocad") -> str:
    position_bytes = b"".join(struct.pack("<fff", *vertex) for vertex in mesh.vertices)
    index_bytes = b"".join(struct.pack("<III", *face) for face in mesh.faces)
    bin_blob = position_bytes + index_bytes
    encoded = base64.b64encode(bin_blob).decode("ascii")
    xs = [v[0] for v in mesh.vertices] or [0]
    ys = [v[1] for v in mesh.vertices] or [0]
    zs = [v[2] for v in mesh.vertices] or [0]
    gltf = {
        "asset": {"version": "2.0", "generator": "TextoCAD cad_worker"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "mode": 4}]}],
        "buffers": [{"uri": f"data:application/octet-stream;base64,{encoded}", "byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes), "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(mesh.vertices), "type": "VEC3", "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 1, "componentType": 5125, "count": len(mesh.faces) * 3, "type": "SCALAR"},
        ],
    }
    return json.dumps(gltf, indent=2)


def export_mesh_as_3mf(mesh: MeshData, name: str = "textocad") -> bytes:
    vertices_xml = "".join(f'<vertex x="{x:.8g}" y="{y:.8g}" z="{z:.8g}"/>' for x, y, z in mesh.vertices)
    triangles_xml = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in mesh.faces)
    model_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">{_slug(name)}</metadata>
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>{vertices_xml}</vertices>
        <triangles>{triangles_xml}</triangles>
      </mesh>
    </object>
  </resources>
  <build><item objectid="1"/></build>
</model>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", model_xml)
    return buffer.getvalue()


def _project_backup(payload: Mapping[str, Any], preview: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "app": "TextoCAD",
            "version": "1.2.0",
            "exportedAt": _iso_now(),
            "projectName": payload.get("projectName") or payload.get("project_name") or "TextoCAD Build",
            "prompt": payload.get("prompt"),
            "mode": payload.get("mode") or payload.get("generationMode") or payload.get("generation_mode"),
            "pipeline": payload.get("pipeline") or payload.get("generationPipeline") or payload.get("generation_pipeline"),
            "units": "mm",
            "code": payload.get("code") or "",
            "params": _as_list(payload.get("params")),
            "features": _as_list(payload.get("features")),
            "robotDescription": payload.get("robotDescription") or payload.get("robot_description"),
            "runtimePreview": preview,
        },
        indent=2,
    )


def _placeholder_step(payload: Mapping[str, Any], preview: Mapping[str, Any]) -> str:
    name = _safe_text(payload.get("projectName") or payload.get("project_name"), "TextoCAD Build")
    return f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('TextoCAD placeholder STEP. True STEP requires CadQuery/OpenCascade worker integration.'),'2;1');
FILE_NAME('{_slug(name)}.step','{_iso_now()}',('TextoCAD'),('TextoCAD'),'TextoCAD','TextoCAD','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=PRODUCT('{_slug(name)}','{_slug(name)}','Generated by TextoCAD placeholder worker',(#2));
#2=PRODUCT_CONTEXT('',#3,'mechanical');
#3=APPLICATION_CONTEXT('configuration controlled 3d designs of mechanical parts and assemblies');
ENDSEC;
END-ISO-10303-21;
"""


def _placeholder_iges(payload: Mapping[str, Any]) -> str:
    name = _safe_text(payload.get("projectName") or payload.get("project_name"), "TextoCAD Build")
    return f"TextoCAD placeholder IGES for {name}\nTrue IGES export should be enabled after STEP/OpenCascade is stable.\n"


def _robot_description(payload: Mapping[str, Any]) -> dict[str, Any]:
    existing = payload.get("robotDescription") or payload.get("robot_description")
    if isinstance(existing, Mapping):
        return dict(existing)
    return {
        "format": "textocad_robot_description",
        "version": "0.1.0",
        "links": [
            {"id": "base_link", "name": "base_link", "visual": {"kind": "box", "size": [0.12, 0.08, 0.04]}},
            {"id": "upper_arm_link", "name": "upper_arm_link", "visual": {"kind": "box", "size": [0.22, 0.035, 0.035]}},
            {"id": "forearm_link", "name": "forearm_link", "visual": {"kind": "box", "size": [0.16, 0.03, 0.03]}},
        ],
        "joints": [
            {"id": "shoulder_joint", "name": "shoulder_joint", "jointType": "revolute", "parent": "base_link", "child": "upper_arm_link", "axis": [0, 0, 1], "limit": {"lower": -1.57, "upper": 1.57}},
            {"id": "elbow_joint", "name": "elbow_joint", "jointType": "revolute", "parent": "upper_arm_link", "child": "forearm_link", "axis": [0, 0, 1], "limit": {"lower": -1.57, "upper": 1.57}},
        ],
    }


def export_robot_urdf(payload: Mapping[str, Any]) -> str:
    description = _robot_description(payload)
    robot_name = _slug(_safe_text(payload.get("projectName") or payload.get("project_name"), "textocad_robot"))
    lines = [f'<robot name="{robot_name}">']
    for link in description.get("links", []):
        link_id = _slug(_safe_text(link.get("id") or link.get("name"), "link"))
        lines.append(f'  <link name="{link_id}">')
        lines.append('    <visual><geometry><box size="0.1 0.05 0.05"/></geometry></visual>')
        lines.append('    <collision><geometry><box size="0.1 0.05 0.05"/></geometry></collision>')
        lines.append('    <inertial><mass value="1"/><inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>')
        lines.append('  </link>')
    for joint in description.get("joints", []):
        joint_id = _slug(_safe_text(joint.get("id") or joint.get("name"), "joint"))
        joint_type = _safe_text(joint.get("jointType") or joint.get("joint_type"), "revolute")
        parent = _slug(_safe_text(joint.get("parent"), "base_link"))
        child = _slug(_safe_text(joint.get("child"), "link"))
        axis = joint.get("axis") if isinstance(joint.get("axis"), list) else [0, 0, 1]
        limit = joint.get("limit") if isinstance(joint.get("limit"), Mapping) else {"lower": -1.57, "upper": 1.57}
        lines.append(f'  <joint name="{joint_id}" type="{joint_type}">')
        lines.append(f'    <parent link="{parent}"/>')
        lines.append(f'    <child link="{child}"/>')
        lines.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'    <axis xyz="{_safe_float(axis[0],0)} {_safe_float(axis[1],0)} {_safe_float(axis[2],1)}"/>')
        if joint_type in {"revolute", "prismatic"}:
            lines.append(f'    <limit lower="{_safe_float(limit.get("lower"), -1.57)}" upper="{_safe_float(limit.get("upper"), 1.57)}" effort="10" velocity="1"/>')
        lines.append('  </joint>')
    lines.append('</robot>')
    return "\n".join(lines) + "\n"


def export_robot_xacro(payload: Mapping[str, Any]) -> str:
    urdf = export_robot_urdf(payload)
    return '<?xml version="1.0"?>\n<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="textocad_robot_xacro">\n' + "\n".join(urdf.splitlines()[1:-1]) + "\n</robot>\n"


def export_robot_ros_zip(payload: Mapping[str, Any], urdf_text: str, xacro_text: str) -> bytes:
    robot_name = _slug(_safe_text(payload.get("projectName") or payload.get("project_name"), "textocad_robot"))
    package_xml = f'''<?xml version="1.0"?>
<package format="3">
  <name>{robot_name}</name>
  <version>0.1.0</version>
  <description>TextoCAD generated robot description package.</description>
  <maintainer email="noreply@example.com">TextoCAD</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz2</exec_depend>
</package>'''
    cmake = f'''cmake_minimum_required(VERSION 3.8)
project({robot_name})
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf xacro launch rviz DESTINATION share/${{PROJECT_NAME}})
ament_package()
'''
    launch = f'''from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', output='screen'),
    ])
'''
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{robot_name}/package.xml", package_xml)
        archive.writestr(f"{robot_name}/CMakeLists.txt", cmake)
        archive.writestr(f"{robot_name}/urdf/{robot_name}.urdf", urdf_text)
        archive.writestr(f"{robot_name}/xacro/{robot_name}.urdf.xacro", xacro_text)
        archive.writestr(f"{robot_name}/launch/display.launch.py", launch)
        archive.writestr(f"{robot_name}/metadata/textocad_package.json", json.dumps(_robot_description(payload), indent=2))
    return buffer.getvalue()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _extension_for_format(fmt: ExportFormat) -> str:
    return {"iges": "igs", "ros_zip": "zip"}.get(fmt, fmt)


def _download_url(build_id: str, fmt: ExportFormat) -> str:
    return f"/api/builds/{build_id}/download/{fmt}"


def _preview_url(build_id: str) -> str:
    return f"/api/builds/{build_id}/download/glb"


def _allowed_exports(mode: GenerationMode, requested: Iterable[Any] | None = None) -> list[ExportFormat]:
    allowed = EXPORTS_BY_MODE[mode]
    if requested is None:
        return list(allowed)
    normalized = [_normalize_format(item) for item in requested]
    selected = [item for item in normalized if item in allowed]
    return selected or list(allowed)


def _tessellate_stl_to_glb(stl_path: Path, glb_path: Path, name: str = "textocad") -> None:
    """Convert a real kernel STL into a viewer GLB so the compiled preview shows actual geometry."""
    import trimesh

    mesh = trimesh.load(str(stl_path), file_type="stl", force="mesh")
    if mesh is None or getattr(mesh, "is_empty", True) or len(getattr(mesh, "faces", [])) == 0:
        raise RuntimeError("kernel STL produced an empty mesh")

    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, (1, 0, 0)))
    try:
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=[51, 133, 255, 255])
    except Exception:
        pass
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(mesh.export(file_type="glb"))


def build_cad_model(payload: Mapping[str, Any], output_root: str | Path | None = None) -> WorkerBuildResult:
    mode = _normalize_mode(payload.get("mode") or payload.get("generationMode") or payload.get("generation_mode"))
    pipeline = _normalize_pipeline(payload.get("pipeline") or payload.get("generationPipeline") or payload.get("generation_pipeline"), mode)
    build_id = _safe_text(payload.get("buildId") or payload.get("build_id"), f"build-{uuid.uuid4().hex[:12]}")
    build_dir = get_build_dir(build_id, output_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    project_name = _safe_text(payload.get("projectName") or payload.get("project_name"), "TextoCAD Build")
    requested_exports = _allowed_exports(mode, payload.get("requestedExports") or payload.get("requested_exports"))
    warnings: list[str] = []

    try:
        document, repair_warnings = _payload_cad_document(payload, mode)
        warnings.extend(repair_warnings)
        document_dict = cad_document_to_dict(document)
        canonical_payload = {
            **dict(payload),
            "mode": mode,
            "pipeline": pipeline,
            "cadDocument": document_dict,
            "cad_document": document_dict,
            "params": document_dict.get("params", []),
            "operations": document_dict.get("operations", []),
        }
        preview = create_runtime_preview(canonical_payload)
        mesh = mesh_from_runtime_preview(preview)
        files: dict[ExportFormat, Path] = {}

        engine_metadata: dict[str, Any] = {
            "realOpenScadUsed": False,
            "realCadQueryUsed": False,
            "safeWorker": True,
            "safeFallbackUsed": mode in {"standard", "extended"},
            "directLlmCodeExecution": False,
            "cadKernel": "runtime_mesh" if mode == "fast" else "none-yet",
            "unsupportedOperations": list(document.metadata.get("unsupportedOperations", [])),
            "warnings": [],
        }


        support_pipeline = "cadquery" if mode in {"standard", "extended"} else "robot" if mode == "robot" else "fast"
        support_summary = operation_support_summary(document.operations, support_pipeline)
        engine_metadata.update(
            {
                "operationSupportSummary": support_summary,
                "nativeOperations": support_summary.get("native", []),
                "approximatedOperations": support_summary.get("approximate", []),
                "skippedOperations": support_summary.get("skipped", []),
            }
        )

        if mode in {"standard", "extended"}:
            engine_metadata.update(build_cadquery_artifacts(document, build_dir, mode=mode))

        warnings.extend([str(item) for item in engine_metadata.get("warnings", []) if str(item)])

        robot_mesh_files: list[Path] = []
        if mode == "robot" and document.robot is not None:
            mesh_root = build_dir / "meshes"
            mesh_root.mkdir(parents=True, exist_ok=True)
            real_links = 0
            link_total = len(document.robot.links)
            for link in document.robot.links:
                link_doc = CadDocument(
                    schemaVersion="textocad.v1",
                    title=link.name,
                    units=document.units,
                    modelType="part",
                    params=document.params,
                    operations=link.operations,
                    metadata={"source": "robot_link", "linkName": link.name},
                )
                safe_link = _slug(link.name, "link")
                stl_path = mesh_root / f"{safe_link}.stl"
                glb_path = mesh_root / f"{safe_link}.glb"


                used_real = False
                if link.operations:
                    link_build_dir = build_dir / "links" / safe_link
                    try:
                        link_meta = build_cadquery_artifacts(link_doc, link_build_dir, mode="standard")
                        link_stl = link_build_dir / "result.stl"
                        if (
                            link_meta.get("realCadQueryUsed")
                            and not link_meta.get("safeFallbackUsed", True)
                            and link_stl.exists()
                        ):
                            shutil.copyfile(link_stl, stl_path)
                            try:
                                _tessellate_stl_to_glb(link_stl, glb_path, safe_link)
                            except Exception:
                                fallback_mesh = mesh_from_runtime_preview(
                                    _runtime_preview_from_document(link_doc, "fast")
                                )
                                _write_bytes(glb_path, export_mesh_as_glb(fallback_mesh, safe_link))
                            used_real = True
                            real_links += 1
                    except Exception as exc:
                        warnings.append(
                            f"Link '{link.name}' CadQuery build failed ({exc}); used approximate mesh."
                        )

                if not used_real:
                    link_preview = _runtime_preview_from_document(link_doc, "fast")
                    link_mesh = mesh_from_runtime_preview(link_preview)
                    _write_text(stl_path, export_mesh_as_stl(link_mesh, safe_link))
                    _write_bytes(glb_path, export_mesh_as_glb(link_mesh, safe_link))

                link.visualMesh = f"meshes/{safe_link}.stl"
                link.collisionMesh = f"meshes/{safe_link}.stl"
                robot_mesh_files.extend([stl_path, glb_path])

            robot_metadata = build_urdf_artifacts(document, build_dir, robot_mesh_files)
            engine_metadata.update(robot_metadata)

            engine_metadata["robotLinkCount"] = link_total
            engine_metadata["robotRealLinks"] = real_links
            engine_metadata["realCadQueryUsed"] = bool(real_links)
            engine_metadata["safeFallbackUsed"] = real_links < link_total
            engine_metadata["previewGeometry"] = "real" if real_links == link_total and link_total else "approximate"
            warnings.extend([str(item) for item in robot_metadata.get("warnings", []) if str(item)])

        for fmt in requested_exports:
            filename = f"model.{_extension_for_format(fmt)}"
            path = build_dir / filename
            if fmt == "json":
                _write_text(path, json.dumps({"cadDocument": document_dict, "buildMetadata": engine_metadata}, indent=2))
            elif fmt == "stl":
                real_stl = build_dir / "result.stl"
                if real_stl.exists() and mode in {"standard", "extended"} and not bool(engine_metadata.get("safeFallbackUsed")):
                    shutil.copyfile(real_stl, path)
                else:
                    _write_text(path, export_mesh_as_stl(mesh, project_name))
            elif fmt == "obj":
                _write_text(path, export_mesh_as_obj(mesh, project_name))
            elif fmt == "glb":
                _write_bytes(path, export_mesh_as_glb(mesh, project_name))
            elif fmt == "gltf":
                _write_text(path, export_mesh_as_gltf(mesh, project_name))
            elif fmt == "3mf":
                _write_bytes(path, export_mesh_as_3mf(mesh, project_name))
            elif fmt == "step":
                real_step = build_dir / "result.step"
                if real_step.exists() and mode == "extended" and not bool(engine_metadata.get("safeFallbackUsed")):
                    shutil.copyfile(real_step, path)
                else:
                    _write_text(path, _placeholder_step(canonical_payload, preview))
            elif fmt == "iges":
                _write_text(path, _placeholder_iges(canonical_payload))
            elif fmt == "urdf":
                robot_urdf = build_dir / "model.urdf"
                if robot_urdf.exists():
                    if robot_urdf.resolve() != path.resolve():
                        shutil.copyfile(robot_urdf, path)
                else:
                    _write_text(path, export_robot_urdf(canonical_payload))
            elif fmt == "xacro":
                robot_xacro = build_dir / "model.xacro"
                if robot_xacro.exists():
                    if robot_xacro.resolve() != path.resolve():
                        shutil.copyfile(robot_xacro, path)
                else:
                    _write_text(path, export_robot_xacro(canonical_payload))
            elif fmt == "ros_zip":
                robot_zip = build_dir / "model.zip"
                if robot_zip.exists():
                    if robot_zip.resolve() != path.resolve():
                        shutil.copyfile(robot_zip, path)
                else:
                    urdf = export_robot_urdf(canonical_payload)
                    xacro = export_robot_xacro(canonical_payload)
                    _write_bytes(path, export_robot_ros_zip(canonical_payload, urdf, xacro))
            files[fmt] = path


        real_kernel_used = bool(engine_metadata.get("realCadQueryUsed")) and not bool(
            engine_metadata.get("safeFallbackUsed", True)
        )
        real_stl = build_dir / "result.stl"
        if mode in {"standard", "extended"} and real_kernel_used and real_stl.exists():
            glb_target = build_dir / "model.glb"
            try:
                _tessellate_stl_to_glb(real_stl, glb_target, project_name)
                files["glb"] = glb_target
                engine_metadata["previewGeometry"] = "real"
                engine_metadata["cadKernel"] = "cadquery"
            except Exception as exc:
                warnings.append(
                    f"Real GLB tessellation failed ({exc}); showing approximate mesh preview."
                )
                engine_metadata["previewGeometry"] = "approximate"
        elif mode in {"standard", "extended"}:
            engine_metadata["previewGeometry"] = "approximate"


        if "glb" not in files:
            preview_path = build_dir / "model.glb"
            _write_bytes(preview_path, export_mesh_as_glb(mesh, project_name))
            files["glb"] = preview_path

        glb_path = files.get("glb")
        if glb_path is None or not glb_path.exists() or glb_path.stat().st_size <= 0:
            preview_path = build_dir / "model.glb"
            _write_bytes(preview_path, export_mesh_as_glb(mesh, project_name))
            files["glb"] = preview_path
            glb_path = preview_path

        if not glb_path.exists() or glb_path.stat().st_size <= 0:
            raise RuntimeError("Backend GLB preview could not be generated.")

        result = WorkerBuildResult(
            build_id=build_id,
            status="ready",
            mode=mode,
            pipeline=pipeline,
            preview_source="compiled",
            build_dir=build_dir,
            files=files,
            exports={fmt: _download_url(build_id, fmt) for fmt in files},
            preview_url=_preview_url(build_id),
            runtime_preview=preview,
            warnings=warnings,
            metadata={
                **engine_metadata,
                "projectName": project_name,
                "requestedExports": requested_exports,
                "cadDocument": document_dict,
                "robotJointValues": dict(payload.get("robotJointValues") or payload.get("robot_joint_values") or {})
                if isinstance(payload.get("robotJointValues") or payload.get("robot_joint_values"), Mapping)
                else {},
                "directLlmCodeExecution": False,
                "buildStage": "ready",
            },
        )
        _write_text(build_dir / "build_metadata.json", json.dumps(result.metadata, indent=2))
        save_build_manifest(result)
        return result
    except Exception as exc:
        result = WorkerBuildResult(
            build_id=build_id,
            status="error",
            mode=mode,
            pipeline=pipeline,
            preview_source="none",
            build_dir=build_dir,
            error=str(exc),
            warnings=warnings,
            metadata={
                "safeWorker": True,
                "directLlmCodeExecution": False,
                "cadKernel": "none-yet",
                "buildStage": "error",
            },
        )
        save_build_manifest(result)
        return result


def _relpath_in_build(path: str | Path, build_dir: str | Path) -> str:
    """Store manifest file paths RELATIVE to the build dir so builds stay portable across machines and changes to."""
    raw = Path(path)
    base = Path(build_dir)
    try:
        return str(raw.resolve().relative_to(base.resolve())).replace("\\", "/")
    except Exception:


        return raw.name


def _resolve_manifest_path(value: str, build_dir: Path) -> Path | None:
    """Resolve a manifest file entry to an existing absolute path."""
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        candidates.append(build_dir / raw.name)
    else:
        candidates.append(build_dir / raw)
        candidates.append(build_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _discover_build_files(build_dir: Path) -> dict[ExportFormat, Path]:
    """Recover standard model.<ext> outputs straight from disk so a build still resolves after a backend restart."""
    found: dict[ExportFormat, Path] = {}
    if not build_dir.exists():
        return found
    for fmt in ("glb", "stl", "step", "3mf", "gltf", "obj", "iges", "json", "urdf", "xacro", "ros_zip"):
        candidate = build_dir / f"model.{_extension_for_format(fmt)}"
        if candidate.exists():
            found[fmt] = candidate
    return found


def save_build_manifest(result: WorkerBuildResult) -> None:
    manifest_path = result.build_dir / "build.json"
    data = result.to_api_dict(include_file_paths=True)
    _write_text(manifest_path, json.dumps(data, indent=2))


def load_build_result(build_id: str, output_root: str | Path | None = None) -> WorkerBuildResult | None:
    build_dir = get_build_dir(build_id, output_root)
    manifest_path = build_dir / "build.json"
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        files_raw = data.get("files") if isinstance(data.get("files"), Mapping) else {}
        files: dict[ExportFormat, Path] = {}
        for key, value in files_raw.items():
            fmt = _normalize_format(key)
            if not fmt:
                continue
            resolved = _resolve_manifest_path(str(value), build_dir)
            if resolved is not None:
                files[fmt] = resolved


        for fmt, path in _discover_build_files(build_dir).items():
            files.setdefault(fmt, path)
        return WorkerBuildResult(
            build_id=_safe_text(data.get("buildId") or data.get("build_id"), build_id),
            status=_safe_text(data.get("status"), "ready"),
            mode=_normalize_mode(data.get("mode") or data.get("generationMode")),
            pipeline=_normalize_pipeline(data.get("pipeline") or data.get("generationPipeline"), _normalize_mode(data.get("mode") or data.get("generationMode"))),
            preview_source=_safe_text(data.get("previewSource") or data.get("preview_source"), "compiled"),
            build_dir=build_dir,
            files=files,
            exports=dict(data.get("exports") or {}),
            preview_url=data.get("previewUrl") or data.get("preview_url"),
            runtime_preview=data.get("runtimePreview") or data.get("runtime_preview") or data.get("preview"),
            error=data.get("error"),
            warnings=[str(item) for item in data.get("warnings", [])] if isinstance(data.get("warnings"), list) else [],
            created_at=_safe_text(data.get("createdAt") or data.get("created_at"), _iso_now()),
            updated_at=_safe_text(data.get("updatedAt") or data.get("updated_at"), _iso_now()),
            metadata=dict(data.get("metadata") or {}),
        )
    except Exception:
        return None


def get_export_path(build_id: str, fmt: ExportFormat | str, output_root: str | Path | None = None) -> Path | None:
    normalized = _normalize_format(fmt)
    if normalized is None:
        return None
    result = load_build_result(build_id, output_root)
    if result and normalized in result.files and result.files[normalized].exists():
        return result.files[normalized]
    candidate = get_build_dir(build_id, output_root) / f"model.{_extension_for_format(normalized)}"
    return candidate if candidate.exists() else None


rebuild_cad_model = build_cad_model
build_compiled_model = build_cad_model

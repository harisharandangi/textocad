from __future__ import annotations

import math
import re
from typing import Any, Mapping

from textocad.core.config import settings
from textocad.services.cad_schema import (
    CadDocument,
    SUPPORTED_OPERATION_TYPES,
    cad_document_to_dict,
    snake_key,
    validate_cad_document,
)
from textocad.services.cad_operations import canonicalize_operation_name
from textocad.services.cad_intent import extract_dimension_hints, infer_cad_intent


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
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
    return int(round(_safe_float(value, fallback)))


def _is_record(value: Any) -> bool:
    return isinstance(value, Mapping)


def _param(
    key: str,
    label: str,
    value: float | int | str | bool,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    step: float | None = 1,
    unit: str | None = "mm",
    group: str | None = None,
    description: str | None = None,
    param_type: str = "number",
) -> dict[str, Any]:
    return {
        "key": snake_key(key),
        "label": label,
        "type": param_type,
        "value": value,
        "defaultValue": value,
        "min": min_value,
        "max": max_value,
        "step": step,
        "unit": unit,
        "group": group,
        "description": description or label,
        "locked": False,
    }


def _op(
    op_id: str,
    op_type: str,
    *,
    params: dict[str, Any] | None = None,
    transform: dict[str, Any] | None = None,
    target: str | None = None,
    inputs: list[str] | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": op_id,
        "type": canonicalize_operation_name(op_type)
        if canonicalize_operation_name(op_type) in SUPPORTED_OPERATION_TYPES
        else "unsupported",
        "visible": True,
        "enabled": True,
        "params": params or {},
        "properties": {},
        "inputs": inputs or [],
        "target": target,
        "transform": {
            "translate": [0, 0, 0],
            "rotate": [0, 0, 0],
            "scale": [1, 1, 1],
            **(transform or {}),
        },
        "tags": tags or [],
        "metadata": metadata or {},
    }


def _extract_existing_params(raw_params: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not isinstance(raw_params, list):
        return result
    for item in raw_params:
        if not isinstance(item, Mapping):
            continue
        key = snake_key(item.get("key") or item.get("id") or item.get("name"))
        if key:
            result[key] = item.get("value", item.get("defaultValue"))
    return result


def _value(existing: Mapping[str, Any], key: str, fallback: float) -> float:
    return _safe_float(existing.get(snake_key(key)), fallback)


def _base_document(
    *,
    title: str,
    prompt: str,
    model_type: str = "part",
    params: list[dict[str, Any]],
    operations: list[dict[str, Any]],
    robot: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schemaVersion": "textocad.v1",
        "title": title,
        "units": "mm",
        "modelType": model_type,
        "params": params,
        "operations": operations,
        "metadata": {
            "prompt": prompt,
            "repaired": True,
            "directLlmCodeExecution": False,
            **(metadata or {}),
        },
    }
    if robot is not None:
        document["robot"] = robot
    return document


def _bracket_document(prompt: str, source: Mapping[str, Any]) -> dict[str, Any]:
    dims = extract_dimension_hints(prompt)
    existing = _extract_existing_params(source.get("params"))

    base_length = _value(existing, "base_length", max(dims.get("length", 100), 100))
    base_width = _value(existing, "base_width", max(dims.get("width", 60), 60))
    plate_thickness = _value(existing, "plate_thickness", max(dims.get("thickness", 8), 6))
    vertical_height = _value(existing, "vertical_height", max(dims.get("height", 70), 70))
    hole_diameter = _value(existing, "hole_diameter", 8)
    fillet_radius = _value(existing, "fillet_radius", max(dims.get("fillet", 3), 2))
    gusset_thickness = _value(existing, "gusset_thickness", max(plate_thickness * 0.8, 4))

    params = [
        _param("base_length", "Base Length", base_length, min_value=20, max_value=400, group="Dimensions"),
        _param("base_width", "Base Width", base_width, min_value=20, max_value=250, group="Dimensions"),
        _param("plate_thickness", "Plate Thickness", plate_thickness, min_value=2, max_value=40, group="Dimensions"),
        _param("vertical_height", "Vertical Height", vertical_height, min_value=20, max_value=300, group="Dimensions"),
        _param("hole_diameter", "Hole Diameter", hole_diameter, min_value=1, max_value=30, step=0.5, group="Holes"),
        _param("fillet_radius", "Inside Fillet Radius", fillet_radius, min_value=0, max_value=20, step=0.5, group="Finishing"),
        _param("gusset_thickness", "Gusset Thickness", gusset_thickness, min_value=2, max_value=30, group="Dimensions"),
    ]

    base_hole_x = base_length * 0.28
    base_hole_y = base_width * 0.28
    back_hole_y = base_width * 0.26
    back_hole_z = vertical_height * 0.55

    operations = [
        _op(
            "base_plate",
            "box",
            params={"size": ["base_length", "base_width", "plate_thickness"]},
            transform={"translate": [0, 0, "plate_thickness / 2"]},
            tags=["body", "base"],
        ),
        _op(
            "rear_vertical_plate",
            "box",
            params={"size": ["plate_thickness", "base_width", "vertical_height"]},
            transform={
                "translate": [
                    "-base_length / 2 + plate_thickness / 2",
                    0,
                    "plate_thickness + vertical_height / 2",
                ]
            },
            tags=["body", "back_plate"],
        ),
        _op(
            "base_holes",
            "hole_pattern",
            params={
                "diameter": "hole_diameter",
                "depth": "plate_thickness + 2",
                "axis": [0, 0, 1],
                "positions": [
                    [-base_hole_x, -base_hole_y, "plate_thickness / 2"],
                    [base_hole_x, -base_hole_y, "plate_thickness / 2"],
                    [-base_hole_x, base_hole_y, "plate_thickness / 2"],
                    [base_hole_x, base_hole_y, "plate_thickness / 2"],
                ],
            },
            target="base_plate",
            tags=["holes", "subtract"],
        ),
        _op(
            "back_plate_holes",
            "hole_pattern",
            params={
                "diameter": "hole_diameter",
                "depth": "plate_thickness + 2",
                "axis": [1, 0, 0],
                "positions": [
                    ["-base_length / 2 + plate_thickness / 2", -back_hole_y, f"{plate_thickness + back_hole_z}"],
                    ["-base_length / 2 + plate_thickness / 2", back_hole_y, f"{plate_thickness + back_hole_z}"],
                ],
            },
            target="rear_vertical_plate",
            tags=["holes", "subtract"],
        ),
        _op(
            "left_triangular_gusset",
            "triangular_prism",
            params={
                "length": "base_length * 0.34",
                "height": "vertical_height * 0.72",
                "width": "gusset_thickness",
            },
            transform={
                "translate": [
                    "-base_length / 2 + base_length * 0.17 + plate_thickness",
                    "-base_width * 0.28",
                    "plate_thickness",
                ]
            },
            tags=["gusset", "body"],
        ),
        _op(
            "right_triangular_gusset",
            "triangular_prism",
            params={
                "length": "base_length * 0.34",
                "height": "vertical_height * 0.72",
                "width": "gusset_thickness",
            },
            transform={
                "translate": [
                    "-base_length / 2 + base_length * 0.17 + plate_thickness",
                    "base_width * 0.28",
                    "plate_thickness",
                ]
            },
            tags=["gusset", "body"],
        ),
        _op(
            "inside_transition_fillet",
            "fillet",
            params={"radius": "fillet_radius"},
            target="base_plate",
            inputs=["base_plate", "rear_vertical_plate"],
            tags=["finish"],
            metadata={"approximation": "Fast/OpenSCAD may ignore this; CadQuery attempts edge fillet."},
        ),
    ]

    return _base_document(
        title="L-Bracket",
        prompt=prompt,
        params=params,
        operations=operations,
        metadata={"intent": "bracket"},
    )


def _gear_document(prompt: str, source: Mapping[str, Any], *, helical: bool = False) -> dict[str, Any]:
    dims = extract_dimension_hints(prompt)
    existing = _extract_existing_params(source.get("params"))
    outer = _value(existing, "outer_diameter", max(dims.get("outerDiameter", 60), 20))
    bore = min(_value(existing, "bore_diameter", max(dims.get("boreDiameter", 16), 3)), outer - 2)
    face_width = _value(existing, "face_width", max(dims.get("thickness", 12), 2))
    tooth_count = max(_safe_int(existing.get("tooth_count"), int(dims.get("toothCount", 24))), 6)
    helix_angle = _value(existing, "helix_angle", dims.get("helixAngle", 15))
    root = max(outer * 0.78, bore + 4)

    params = [
        _param("outer_diameter", "Outer Diameter", outer, min_value=8, max_value=400, group="Gear"),
        _param("root_diameter", "Root Diameter", root, min_value=bore + 2, max_value=outer, group="Gear"),
        _param("bore_diameter", "Bore Diameter", bore, min_value=1, max_value=outer - 2, group="Holes"),
        _param("face_width", "Face Width", face_width, min_value=1, max_value=120, group="Gear"),
        _param("tooth_count", "Tooth Count", tooth_count, min_value=6, max_value=160, step=1, unit="", group="Gear"),
        _param("helix_angle", "Helix Angle", helix_angle, min_value=-45, max_value=45, unit="deg", group="Gear"),
    ]
    operations = [
        _op(
            "gear_body",
            "gear_helical_approx" if helical else "gear_spur_approx",
            params={
                "outer_diameter": "outer_diameter",
                "root_diameter": "root_diameter",
                "bore_diameter": "bore_diameter",
                "face_width": "face_width",
                "tooth_count": "tooth_count",
                "helix_angle": "helix_angle",
            },
            tags=["body", "gear"],
        )
    ]
    return _base_document(
        title="Helical Gear" if helical else "Spur Gear",
        prompt=prompt,
        params=params,
        operations=operations,
        metadata={"intent": "helical_gear" if helical else "gear"},
    )


def _robot_document(prompt: str, source: Mapping[str, Any]) -> dict[str, Any]:
    dims = extract_dimension_hints(prompt)
    base_width = max(dims.get("width", 120), 80)
    base_depth = max(dims.get("depth", 80), 60)
    base_height = max(dims.get("thickness", 35), 20)
    upper = max(dims.get("length", 220), 120)
    forearm = upper * 0.75
    link_thickness = 35

    params = [
        _param("base_width", "Base Width", base_width, min_value=40, max_value=400, group="Robot"),
        _param("upper_arm_length", "Upper Arm Length", upper, min_value=60, max_value=700, group="Robot"),
        _param("forearm_length", "Forearm Length", forearm, min_value=60, max_value=700, group="Robot"),
        _param("link_thickness", "Link Thickness", link_thickness, min_value=10, max_value=80, group="Robot"),
    ]

    base_ops = [
        _op("base_body", "box", params={"size": [base_width, base_depth, base_height]}, transform={"translate": [0, 0, base_height / 2]})
    ]
    upper_ops = [
        _op("upper_arm_body", "box", params={"size": [upper, link_thickness, link_thickness]}, transform={"translate": [upper / 2, 0, 0]})
    ]
    forearm_ops = [
        _op("forearm_body", "box", params={"size": [forearm, link_thickness * 0.85, link_thickness * 0.85]}, transform={"translate": [forearm / 2, 0, 0]})
    ]

    robot = {
        "robotName": "textocad_robot_arm",
        "baseLink": "base_link",
        "units": "m",
        "links": [
            {"name": "base_link", "operations": base_ops, "mass": 2.0, "origin": [0, 0, 0], "rpy": [0, 0, 0]},
            {"name": "upper_arm_link", "operations": upper_ops, "mass": 1.0, "origin": [0, 0, base_height], "rpy": [0, 0, 0]},
            {"name": "forearm_link", "operations": forearm_ops, "mass": 0.8, "origin": [upper, 0, base_height], "rpy": [0, 0, 0]},
        ],
        "joints": [
            {
                "name": "shoulder_joint",
                "type": "revolute",
                "parent": "base_link",
                "child": "upper_arm_link",
                "origin": [0, 0, base_height / 1000],
                "rpy": [0, 0, 0],
                "axis": [0, 0, 1],
                "limit": {"lower": -1.57, "upper": 1.57, "effort": 10, "velocity": 1},
            },
            {
                "name": "elbow_joint",
                "type": "revolute",
                "parent": "upper_arm_link",
                "child": "forearm_link",
                "origin": [upper / 1000, 0, 0],
                "rpy": [0, 0, 0],
                "axis": [0, 1, 0],
                "limit": {"lower": -1.57, "upper": 1.57, "effort": 10, "velocity": 1},
            },
        ],
    }

    operations = [
        _op("base_link_preview", "box", params={"size": [base_width, base_depth, base_height]}, transform={"translate": [0, 0, base_height / 2]}, tags=["robot_link"]),
        _op("upper_arm_preview", "box", params={"size": [upper, link_thickness, link_thickness]}, transform={"translate": [upper / 2, 0, base_height + link_thickness / 2]}, tags=["robot_link"]),
        _op("forearm_preview", "box", params={"size": [forearm, link_thickness * 0.85, link_thickness * 0.85]}, transform={"translate": [upper + forearm / 2, 0, base_height + link_thickness / 2]}, tags=["robot_link"]),
    ]

    return _base_document(
        title="Robot Arm",
        prompt=prompt,
        model_type="robot",
        params=params,
        operations=operations,
        robot=robot,
        metadata={"intent": "robot"},
    )


def _generic_document(prompt: str, source: Mapping[str, Any]) -> dict[str, Any]:
    dims = extract_dimension_hints(prompt)
    width = max(dims.get("width", 80), 10)
    depth = max(dims.get("depth", 50), 10)
    height = max(dims.get("height", dims.get("thickness", 20)), 5)
    params = [
        _param("width", "Width", width, min_value=1, max_value=400, group="Dimensions"),
        _param("depth", "Depth", depth, min_value=1, max_value=400, group="Dimensions"),
        _param("height", "Height", height, min_value=1, max_value=300, group="Dimensions"),
    ]
    operations = [
        _op("main_body", "box", params={"size": ["width", "depth", "height"]}),
    ]
    return _base_document(
        title=_safe_text(source.get("title"), "Generated CAD Model"),
        prompt=prompt,
        params=params,
        operations=operations,
        metadata={"intent": "generic"},
    )


def infer_document_from_prompt(prompt: str, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    source = source or {}
    clean = prompt.strip() or _safe_text(source.get("prompt"), "Create a simple CAD part")
    intent = infer_cad_intent(clean)
    lowered = clean.lower()

    if intent == "robot":
        return _robot_document(clean, source)
    if intent in {"gear", "helical_gear"}:
        return _gear_document(clean, source, helical=intent == "helical_gear" or "helical" in lowered)
    if intent == "bracket" or "l-bracket" in lowered or "l bracket" in lowered:
        return _bracket_document(clean, source)
    return _generic_document(clean, source)


def _repair_raw_document(raw: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    document = dict(raw)
    document["schemaVersion"] = "textocad.v1"
    document["title"] = _safe_text(document.get("title"), "TextoCAD Model")
    units = _safe_text(document.get("units"), "mm").lower()
    document["units"] = "inch" if units in {"in", "inch", "inches"} else units if units in {"mm", "cm", "m"} else "mm"
    model_type = _safe_text(document.get("modelType") or document.get("model_type"), "part").lower()
    document["modelType"] = model_type if model_type in {"part", "assembly", "robot"} else "part"

    if not isinstance(document.get("params"), list):
        document["params"] = []
    if not isinstance(document.get("operations"), list):
        document["operations"] = []

    repaired_ops: list[dict[str, Any]] = []
    unsupported: list[str] = []
    for index, item in enumerate(document["operations"]):
        if not isinstance(item, Mapping):
            continue
        op = dict(item)
        original_type = op.get("type")
        clean_type = canonicalize_operation_name(original_type)
        if clean_type not in SUPPORTED_OPERATION_TYPES:
            op["type"] = "unsupported"
            metadata = dict(op.get("metadata") or {})
            metadata["unsupportedType"] = original_type
            metadata["unsupportedReason"] = "Unknown operation type from model JSON."
            op["metadata"] = metadata
            unsupported.append(str(original_type))
        op.setdefault("id", f"op_{index + 1}")
        op.setdefault("visible", True)
        op.setdefault("enabled", True)
        op.setdefault("params", {})
        op.setdefault("properties", {})
        op.setdefault("inputs", [])
        op.setdefault("target", None)
        op.setdefault("transform", {"translate": [0, 0, 0], "rotate": [0, 0, 0], "scale": [1, 1, 1]})
        op.setdefault("tags", [])
        op.setdefault("metadata", {})
        repaired_ops.append(op)

    if not repaired_ops:


        if isinstance(document.get("robot"), Mapping) or document.get("modelType") == "robot":
            metadata = dict(document.get("metadata") or {})
            metadata.setdefault("prompt", prompt)
            metadata.setdefault("directLlmCodeExecution", False)
            metadata["unsupportedOperations"] = unsupported
            document["metadata"] = metadata
            document["operations"] = []
            return document
        return infer_document_from_prompt(prompt, document)

    metadata = dict(document.get("metadata") or {})
    metadata.setdefault("prompt", prompt)
    metadata.setdefault("directLlmCodeExecution", False)
    metadata["unsupportedOperations"] = unsupported
    document["metadata"] = metadata
    document["operations"] = repaired_ops
    return document


def repair_cad_document(raw: Any, *, prompt: str = "", mode: str = "part") -> tuple[CadDocument, list[str]]:
    source = raw if isinstance(raw, Mapping) else {}
    prompt = _safe_text(prompt or source.get("prompt"), "Create a simple CAD model")
    warnings: list[str] = []

    candidate: dict[str, Any]
    if isinstance(source.get("cadDocument"), Mapping):
        candidate = dict(source["cadDocument"])
    elif isinstance(source.get("cad_document"), Mapping):
        candidate = dict(source["cad_document"])
    elif source.get("schemaVersion") == "textocad.v1":
        candidate = _repair_raw_document(source, prompt)
    elif isinstance(source.get("operations"), list):
        candidate = _repair_raw_document(source, prompt)
    else:
        candidate = infer_document_from_prompt(prompt, source)

    if mode == "robot":
        candidate["modelType"] = "robot"
        if not isinstance(candidate.get("robot"), Mapping):
            robot_doc = _robot_document(prompt, source)
            candidate["robot"] = robot_doc["robot"]

    try:
        document = validate_cad_document(candidate)
    except Exception as exc:
        warnings.append(f"Canonical CAD JSON was repaired after validation failed: {exc}")
        document = validate_cad_document(infer_document_from_prompt(prompt, source))

    unsupported = document.metadata.get("unsupportedOperations")
    if isinstance(unsupported, list) and unsupported:
        warnings.append(f"Unsupported operation types were preserved as unsupported: {', '.join(map(str, unsupported))}")

    if len(document.operations) > settings.max_operations:
        document.operations = document.operations[: settings.max_operations]
        warnings.append(f"CAD operations were trimmed to {settings.max_operations}.")

    if len(document.params) > settings.max_params:
        document.params = document.params[: settings.max_params]
        warnings.append(f"CAD params were trimmed to {settings.max_params}.")

    return document, warnings


def repair_cad_document_dict(raw: Any, *, prompt: str = "", mode: str = "part") -> tuple[dict[str, Any], list[str]]:
    document, warnings = repair_cad_document(raw, prompt=prompt, mode=mode)
    return cad_document_to_dict(document), warnings

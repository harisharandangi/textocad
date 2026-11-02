from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from textocad.services.cad_operations import operation_support_summary
from textocad.services.cad_schema import CadDocument, CadOperation, param_context, resolve_number


@dataclass
class OpenScadCompileResult:
    source: str
    unsupported_operations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    support_summary: dict[str, list[str]] = field(default_factory=dict)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _vec(value: Any, params: Mapping[str, float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) < 3:
        return fallback
    return (
        resolve_number(value[0], params, fallback[0]),
        resolve_number(value[1], params, fallback[1]),
        resolve_number(value[2], params, fallback[2]),
    )


def _scad_vec(values: tuple[float, float, float]) -> str:
    return f"[{_fmt(values[0])}, {_fmt(values[1])}, {_fmt(values[2])}]"


def _transform(code: str, op: CadOperation, params: Mapping[str, float]) -> str:
    translate = _vec(op.transform.translate, params, (0, 0, 0))
    rotate = _vec(op.transform.rotate, params, (0, 0, 0))
    scale = _vec(op.transform.scale, params, (1, 1, 1))
    mirror = _vec(op.transform.mirror, params, (0, 0, 0)) if op.transform.mirror else None

    wrapped = code
    if scale != (1, 1, 1):
        wrapped = f"scale({_scad_vec(scale)}) {{\n{_indent(wrapped)}\n}}"
    if rotate != (0, 0, 0):
        wrapped = f"rotate({_scad_vec(rotate)}) {{\n{_indent(wrapped)}\n}}"
    if mirror and mirror != (0, 0, 0):
        wrapped = f"mirror({_scad_vec(mirror)}) {{\n{_indent(wrapped)}\n}}"
    if translate != (0, 0, 0):
        wrapped = f"translate({_scad_vec(translate)}) {{\n{_indent(wrapped)}\n}}"
    return wrapped


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())


def _size(op: CadOperation, params: Mapping[str, float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = op.params.get("size")
    if isinstance(raw, list) and len(raw) >= 3:
        return _vec(raw, params, fallback)
    return (
        resolve_number(op.params.get("width", op.params.get("x")), params, fallback[0]),
        resolve_number(op.params.get("depth", op.params.get("y")), params, fallback[1]),
        resolve_number(op.params.get("height", op.params.get("z", op.params.get("thickness"))), params, fallback[2]),
    )


def _compile_gear(op: CadOperation, params: Mapping[str, float]) -> str:
    outer = resolve_number(op.params.get("outer_diameter"), params, 60)
    root = resolve_number(op.params.get("root_diameter"), params, outer * 0.82)
    bore = resolve_number(op.params.get("bore_diameter"), params, 0)
    width = resolve_number(op.params.get("face_width", op.params.get("width")), params, 12)
    teeth = max(6, int(round(resolve_number(op.params.get("tooth_count"), params, 24))))
    outer_r = outer / 2
    root_r = min(max(root / 2, outer_r * 0.45), outer_r * 0.96)
    points: list[str] = []
    for index in range(teeth * 4):
        angle = math.tau * index / (teeth * 4)
        radius = outer_r if index % 4 in {1, 2} else root_r
        points.append(f"[{_fmt(math.cos(angle) * radius)}, {_fmt(math.sin(angle) * radius)}]")
    polygon = f"linear_extrude(height={_fmt(width)}, center=true) polygon(points=[{', '.join(points)}]);"
    if bore > 0:
        return (
            "difference() {\n"
            f"{_indent(polygon)}\n"
            f"  cylinder(h={_fmt(width + 2)}, d={_fmt(bore)}, center=true, $fn=96);\n"
            "}"
        )
    return polygon


def _compile_triangular_prism(op: CadOperation, params: Mapping[str, float]) -> str:
    length = resolve_number(op.params.get("length"), params, 30)
    width = resolve_number(op.params.get("width"), params, 8)
    height = resolve_number(op.params.get("height"), params, 30)
    points = [
        [0, -width / 2, 0],
        [length, -width / 2, 0],
        [0, -width / 2, height],
        [0, width / 2, 0],
        [length, width / 2, 0],
        [0, width / 2, height],
    ]
    faces = [[0, 1, 2], [3, 5, 4], [0, 3, 4, 1], [0, 2, 5, 3], [1, 4, 5, 2]]
    return f"polyhedron(points={json.dumps(points)}, faces={json.dumps(faces)});"


def compile_positive_operation(
    op: CadOperation,
    params: Mapping[str, float],
    result: OpenScadCompileResult,
) -> str | None:
    if not op.enabled or not op.visible:
        return None

    if op.type in {"hole", "hole_pattern", "fillet", "chamfer"}:
        return None

    code: str | None = None
    if op.type == "box":
        sx, sy, sz = _size(op, params, (20, 20, 20))
        code = f"cube([{_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)}], center=true);"
    elif op.type == "cylinder":
        height = resolve_number(op.params.get("height", op.params.get("length")), params, 20)
        diameter = resolve_number(op.params.get("diameter"), params, 0)
        radius = resolve_number(op.params.get("radius"), params, diameter / 2 if diameter else 10)
        code = f"cylinder(h={_fmt(height)}, r={_fmt(radius)}, center=true, $fn=96);"
    elif op.type == "sphere":
        radius = resolve_number(op.params.get("radius"), params, resolve_number(op.params.get("diameter"), params, 20) / 2)
        code = f"sphere(r={_fmt(radius)}, $fn=64);"
    elif op.type == "cone":
        height = resolve_number(op.params.get("height"), params, 30)
        r1 = resolve_number(op.params.get("bottom_radius", op.params.get("radius1")), params, 15)
        r2 = resolve_number(op.params.get("top_radius", op.params.get("radius2")), params, 5)
        code = f"cylinder(h={_fmt(height)}, r1={_fmt(r1)}, r2={_fmt(r2)}, center=true, $fn=96);"
    elif op.type in {"wedge", "triangular_prism"}:
        code = _compile_triangular_prism(op, params)
    elif op.type == "polyhedron":
        points = op.params.get("points")
        faces = op.params.get("faces")
        if isinstance(points, list) and isinstance(faces, list) and points and faces:
            code = f"polyhedron(points={json.dumps(points)}, faces={json.dumps(faces)});"
        else:
            result.warnings.append(f"{op.id}: polyhedron requires points and faces; skipped.")
    elif op.type in {"gear_spur_approx", "gear_helical_approx"}:
        if op.type == "gear_helical_approx":
            result.warnings.append("OpenSCAD helical gear uses a spur-tooth approximation.")
        code = _compile_gear(op, params)
    elif op.type == "polygon_extrude":
        points = op.params.get("points")
        height = resolve_number(op.params.get("height"), params, 10)
        if isinstance(points, list) and len(points) >= 3:
            code = f"linear_extrude(height={_fmt(height)}, center=true) polygon(points={json.dumps(points)});"
        else:
            result.warnings.append(f"{op.id}: polygon_extrude requires at least three 2D points; skipped.")
    elif op.type == "polygon":
        points = op.params.get("points")
        height = resolve_number(op.params.get("height"), params, 1)
        if isinstance(points, list) and len(points) >= 3:
            code = f"linear_extrude(height={_fmt(height)}, center=true) polygon(points={json.dumps(points)});"
        else:
            result.warnings.append(f"{op.id}: polygon requires points; skipped.")
    elif op.type == "text":
        text = str(op.params.get("text") or op.params.get("value") or "Text").replace("\\", "\\\\").replace('"', '\\"')
        size = resolve_number(op.params.get("size"), params, 10)
        height = resolve_number(op.params.get("height"), params, 1)
        code = f'linear_extrude(height={_fmt(height)}, center=true) text("{text}", size={_fmt(size)}, halign="center", valign="center");'
    elif op.type == "text_emboss":
        text = str(op.params.get("text") or op.params.get("value") or "Text").replace("\\", "\\\\").replace('"', '\\"')
        size = resolve_number(op.params.get("size"), params, 10)
        height = resolve_number(op.params.get("height"), params, 1)
        code = f'linear_extrude(height={_fmt(height)}, center=true) text("{text}", size={_fmt(size)}, halign="center", valign="center");'
    elif op.type == "linear_extrude":
        points = op.params.get("points")
        height = resolve_number(op.params.get("height"), params, 10)
        if isinstance(points, list) and len(points) >= 3:
            code = f"linear_extrude(height={_fmt(height)}, center=true) polygon(points={json.dumps(points)});"
        else:
            result.warnings.append(f"{op.id}: linear_extrude requires polygon points; skipped.")
    elif op.type in {
        "union",
        "combine",
        "subtract",
        "difference",
        "cut",
        "intersect",
        "intersection",
        "translate",
        "rotate",
        "scale",
        "mirror",
        "resize",
        "pattern_linear",
        "pattern_circular",
        "rarray",
        "polar_array",
        "revolve",
        "rotate_extrude",
        "hull",
        "offset",
        "minkowski",
        "import_mesh_placeholder",
        "import_step_placeholder",
        "import_dxf_placeholder",
    }:
        if op.type in {"subtract", "difference", "cut", "intersect", "intersection"}:
            result.warnings.append(f"{op.id}: {op.type} operation has no standalone OpenSCAD geometry; explicit cutters are compiled when supplied.")
        return None
    elif op.type == "unsupported":
        result.unsupported_operations.append(str(op.metadata.get("unsupportedType") or op.id))

    return _transform(code, op, params) if code else None


def _hole_cylinder(position: tuple[float, float, float], axis: tuple[float, float, float], diameter: float, depth: float) -> str:
    body = f"cylinder(h={_fmt(depth)}, d={_fmt(diameter)}, center=true, $fn=64);"
    if abs(axis[0]) >= max(abs(axis[1]), abs(axis[2])):
        body = f"rotate([0, 90, 0]) {body}"
    elif abs(axis[1]) >= max(abs(axis[0]), abs(axis[2])):
        body = f"rotate([90, 0, 0]) {body}"
    return f"translate({_scad_vec(position)}) {body}"


def compile_cut_operation(op: CadOperation, params: Mapping[str, float]) -> list[str]:
    if not op.enabled:
        return []
    if op.type not in {"hole", "hole_pattern"}:
        return []
    diameter = resolve_number(op.params.get("diameter", op.params.get("hole_diameter")), params, 5)
    depth = resolve_number(op.params.get("depth"), params, 20)
    axis = _vec(op.params.get("axis"), params, (0, 0, 1))
    if op.type == "hole":
        position = _vec(op.params.get("position"), params, (0, 0, 0))
        return [_hole_cylinder(position, axis, diameter, depth)]
    positions = op.params.get("positions")
    if not isinstance(positions, list):
        return []
    cutters: list[str] = []
    for item in positions:
        cutters.append(_hole_cylinder(_vec(item, params, (0, 0, 0)), axis, diameter, depth))
    return cutters


def compile_openscad(document: CadDocument) -> OpenScadCompileResult:
    result = OpenScadCompileResult(source="")
    params = param_context(document.params)
    positives: list[str] = []
    cutters: list[str] = []

    for op in document.operations:
        positive = compile_positive_operation(op, params, result)
        if positive:
            positives.append(positive)
        cutters.extend(compile_cut_operation(op, params))
        if op.type in {"fillet", "chamfer"}:
            result.warnings.append(f"{op.id}: {op.type} is skipped in OpenSCAD fallback; CadQuery can attempt a true {op.type}.")

    if not positives:
        positives.append("cube([20, 20, 20], center=true);")
        result.warnings.append("No supported positive operations were found; emitted a safe cube fallback.")

    header = [
        "// TextoCAD deterministic OpenSCAD source",
        "// Generated only from validated canonical CAD JSON.",
        "// directLlmCodeExecution=false",
        "$fn = 96;",
        "",
    ]
    body = "union() {\n" + "\n".join(_indent(item) for item in positives) + "\n}"
    if cutters:
        body = (
            "difference() {\n"
            f"{_indent(body)}\n"
            "  union() {\n"
            + "\n".join(_indent(item, "    ") for item in cutters)
            + "\n  }\n"
            "}"
        )
    result.source = "\n".join(header) + body + "\n"
    result.unsupported_operations = sorted(set(result.unsupported_operations))
    result.support_summary = operation_support_summary(document.operations, "openscad")
    result.support_summary["unsupported"] = sorted(set(result.unsupported_operations))
    return result

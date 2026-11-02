from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from textocad.services.cad_operations import operation_support_summary
from textocad.services.cad_schema import CadDocument, CadOperation, param_context, resolve_number


@dataclass
class CadQueryCompileResult:
    source: str
    unsupported_operations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    support_summary: dict[str, list[str]] = field(default_factory=dict)


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.9f}".rstrip("0").rstrip(".")


def _vec(value: Any, params: Mapping[str, float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) < 3:
        return fallback
    return (
        resolve_number(value[0], params, fallback[0]),
        resolve_number(value[1], params, fallback[1]),
        resolve_number(value[2], params, fallback[2]),
    )


def _size(op: CadOperation, params: Mapping[str, float], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = op.params.get("size")
    if isinstance(raw, list) and len(raw) >= 3:
        return _vec(raw, params, fallback)
    return (
        resolve_number(op.params.get("width", op.params.get("x")), params, fallback[0]),
        resolve_number(op.params.get("depth", op.params.get("y")), params, fallback[1]),
        resolve_number(op.params.get("height", op.params.get("z", op.params.get("thickness"))), params, fallback[2]),
    )


def _resolve_profile_points(op: "CadOperation | None", params: Mapping[str, float]) -> list[tuple[float, float]]:
    """Resolve a 2D sketch profile ([[x, y], ...], values may be param expressions)."""
    if op is None:
        return []
    raw = op.params.get("points")
    if not isinstance(raw, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((resolve_number(item[0], params, 0.0), resolve_number(item[1], params, 0.0)))
    return points


def _translate_line(var_name: str, op: CadOperation, params: Mapping[str, float]) -> list[str]:
    translate = _vec(op.transform.translate, params, (0, 0, 0))
    rotate = _vec(op.transform.rotate, params, (0, 0, 0))
    lines: list[str] = []
    if rotate != (0, 0, 0):
        lines.append(
            f"{var_name} = {var_name}.rotate((0, 0, 0), (1, 0, 0), {_fmt(rotate[0])}).rotate((0, 0, 0), (0, 1, 0), {_fmt(rotate[1])}).rotate((0, 0, 0), (0, 0, 1), {_fmt(rotate[2])})"
        )
    if translate != (0, 0, 0):
        lines.append(f"{var_name} = {var_name}.translate(({_fmt(translate[0])}, {_fmt(translate[1])}, {_fmt(translate[2])}))")
    return lines


def _compile_gear(op: CadOperation, params: Mapping[str, float], var_name: str, result: CadQueryCompileResult) -> list[str]:
    outer = resolve_number(op.params.get("outer_diameter"), params, 60)
    root = resolve_number(op.params.get("root_diameter"), params, outer * 0.82)
    bore = resolve_number(op.params.get("bore_diameter"), params, 0)
    width = resolve_number(op.params.get("face_width", op.params.get("width")), params, 12)
    teeth = max(6, int(round(resolve_number(op.params.get("tooth_count"), params, 24))))
    if op.type == "gear_helical_approx":
        result.warnings.append("CadQuery helical gear uses segmented spur-tooth approximation.")
    outer_r = outer / 2
    root_r = min(max(root / 2, outer_r * 0.45), outer_r * 0.96)
    points: list[tuple[float, float]] = []
    for index in range(teeth * 4):
        angle = math.tau * index / (teeth * 4)
        radius = outer_r if index % 4 in {1, 2} else root_r
        points.append((math.cos(angle) * radius, math.sin(angle) * radius))
    lines = [
        f"{var_name}_pts = {json.dumps(points)}",
        f"{var_name} = cq.Workplane('XY').polyline({var_name}_pts).close().extrude({_fmt(width / 2)}, both=True)",
    ]
    if bore > 0:
        lines.append(f"{var_name} = {var_name}.cut(cq.Workplane('XY').circle({_fmt(bore / 2)}).extrude({_fmt(width + 2)}, both=True))")
    return lines


def _compile_triangular_prism(op: CadOperation, params: Mapping[str, float], var_name: str) -> list[str]:
    length = resolve_number(op.params.get("length"), params, 30)
    width = resolve_number(op.params.get("width"), params, 8)
    height = resolve_number(op.params.get("height"), params, 30)

    return [
        f"{var_name} = cq.Workplane('XZ').polyline([(0, 0), ({_fmt(length)}, 0), (0, {_fmt(height)})]).close().extrude({_fmt(width / 2)}, both=True)",
    ]


def compile_positive_operation(
    op: CadOperation,
    params: Mapping[str, float],
    var_name: str,
    result: CadQueryCompileResult,
    ops_by_id: Mapping[str, CadOperation] | None = None,
) -> list[str]:
    if not op.enabled or not op.visible:
        return []

    if op.type in {"hole", "hole_pattern", "cbore_hole", "csk_hole", "fillet", "chamfer", "shell"}:
        return []

    lines: list[str] = []
    if op.type == "box":
        sx, sy, sz = _size(op, params, (20, 20, 20))
        lines.append(f"{var_name} = make_box({_fmt(sx)}, {_fmt(sy)}, {_fmt(sz)})")
    elif op.type == "cylinder":
        height = resolve_number(op.params.get("height", op.params.get("length")), params, 20)
        diameter = resolve_number(op.params.get("diameter"), params, 0)
        radius = resolve_number(op.params.get("radius"), params, diameter / 2 if diameter else 10)
        lines.append(f"{var_name} = make_cylinder({_fmt(radius)}, {_fmt(height)})")
    elif op.type == "sphere":
        radius = resolve_number(op.params.get("radius"), params, resolve_number(op.params.get("diameter"), params, 20) / 2)
        lines.append(f"{var_name} = make_sphere({_fmt(radius)})")
    elif op.type == "cone":
        height = resolve_number(op.params.get("height"), params, 30)
        r1 = resolve_number(op.params.get("bottom_radius", op.params.get("radius1")), params, 15)
        r2 = resolve_number(op.params.get("top_radius", op.params.get("radius2")), params, 5)
        lines.append(f"{var_name} = make_cone({_fmt(r1)}, {_fmt(r2)}, {_fmt(height)})")
    elif op.type in {"wedge", "triangular_prism"}:
        lines.extend(_compile_triangular_prism(op, params, var_name))
    elif op.type in {"spur_gear", "helical_gear"}:


        module = resolve_number(op.params.get("module"), params, 2)
        teeth = resolve_number(
            op.params.get("teeth") or op.params.get("teeth_count") or op.params.get("number_of_teeth"),
            params, 18,
        )
        width = resolve_number(
            op.params.get("width") or op.params.get("face_width") or op.params.get("thickness"),
            params, 10,
        )
        bore = resolve_number(
            op.params.get("bore_diameter") or op.params.get("bore") or op.params.get("bore_d"),
            params, 0,
        )
        helix = 0.0
        if op.type == "helical_gear":
            helix = resolve_number(op.params.get("helix_angle") or op.params.get("spiral_angle"), params, 20)
        lines.append(
            f"{var_name} = make_spur_gear({_fmt(module)}, {_fmt(teeth)}, {_fmt(width)}, {_fmt(bore)}, {_fmt(helix)})"
        )
    elif op.type in {"gear_spur_approx", "gear_helical_approx"}:
        lines.extend(_compile_gear(op, params, var_name, result))
    elif op.type == "polygon_extrude":
        points = op.params.get("points")
        height = resolve_number(op.params.get("height"), params, 10)
        if isinstance(points, list) and len(points) >= 3:
            lines.append(f"{var_name} = cq.Workplane('XY').polyline({json.dumps(points)}).close().extrude({_fmt(height / 2)}, both=True)")
        else:
            result.warnings.append(f"{op.id}: polygon_extrude requires at least three 2D points; skipped.")
    elif op.type == "linear_extrude":
        points = op.params.get("points")
        height = resolve_number(op.params.get("height"), params, 10)
        if isinstance(points, list) and len(points) >= 3:
            lines.append(f"{var_name} = cq.Workplane('XY').polyline({json.dumps(points)}).close().extrude({_fmt(height / 2)}, both=True)")
        else:
            result.warnings.append(f"{op.id}: linear_extrude requires polygon points; skipped.")
    elif op.type in {"text", "text_emboss"}:
        text = str(op.params.get("text") or op.params.get("value") or "Text").replace("\\", "\\\\").replace("'", "\\'")
        size = resolve_number(op.params.get("size"), params, 10)
        height = resolve_number(op.params.get("height"), params, 1)
        lines.append(f"{var_name} = make_text('{text}', {_fmt(size)}, {_fmt(height)})")
    elif op.type == "revolve":


        lookup = ops_by_id or {}
        profile_op = next(
            (lookup[i] for i in op.inputs if i in lookup and isinstance(lookup[i].params.get("points"), list)),
            None,
        )
        points = _resolve_profile_points(profile_op or op, params)
        if len(points) < 3:
            result.warnings.append(f"{op.id}: revolve profile needs at least three points; skipped.")
            return []
        plane = "XZ"
        plane_src = (profile_op or op).params.get("plane")
        if isinstance(plane_src, str) and plane_src.strip().upper() in {"XY", "XZ", "YZ"}:
            plane = plane_src.strip().upper()
        angle = resolve_number(op.params.get("angle"), params, 360)
        pts_literal = "[" + ", ".join(f"({_fmt(x)}, {_fmt(y)})" for x, y in points) + "]"
        lines.append(
            f"{var_name} = cq.Workplane('{plane}').polyline({pts_literal}).close().revolve({_fmt(angle)})"
        )
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
        "mirror_x",
        "mirror_y",
        "pattern_linear",
        "pattern_circular",
        "rarray",
        "polar_array",
        "sweep",
        "loft",
        "split",
    }:
        if op.type not in {"union", "combine"}:
            result.warnings.append(f"{op.id}: {op.type} needs a valid operation graph target/input; skipped in flat fallback compile.")
        return []
    else:
        result.unsupported_operations.append(str(op.metadata.get("unsupportedType") or op.type))

    if lines:
        lines.extend(_translate_line(var_name, op, params))
    return lines


def _hole_cutter_lines(
    var_name: str,
    position: tuple[float, float, float],
    axis: tuple[float, float, float],
    diameter: float,
    depth: float,
) -> list[str]:
    lines = [f"{var_name} = cq.Workplane('XY').circle({_fmt(diameter / 2)}).extrude({_fmt(depth / 2)}, both=True)"]
    if abs(axis[0]) >= max(abs(axis[1]), abs(axis[2])):
        lines.append(f"{var_name} = {var_name}.rotate((0, 0, 0), (0, 1, 0), 90)")
    elif abs(axis[1]) >= max(abs(axis[0]), abs(axis[2])):
        lines.append(f"{var_name} = {var_name}.rotate((0, 0, 0), (1, 0, 0), 90)")
    lines.append(f"{var_name} = {var_name}.translate(({_fmt(position[0])}, {_fmt(position[1])}, {_fmt(position[2])}))")
    return lines


def _orient_and_place_lines(var_name: str, axis: tuple[float, float, float], position: tuple[float, float, float]) -> list[str]:
    lines: list[str] = []
    if abs(axis[0]) >= max(abs(axis[1]), abs(axis[2])):
        lines.append(f"{var_name} = {var_name}.rotate((0, 0, 0), (0, 1, 0), 90)")
    elif abs(axis[1]) >= max(abs(axis[0]), abs(axis[2])):
        lines.append(f"{var_name} = {var_name}.rotate((0, 0, 0), (1, 0, 0), 90)")
    lines.append(f"{var_name} = {var_name}.translate(({_fmt(position[0])}, {_fmt(position[1])}, {_fmt(position[2])}))")
    return lines


def _cbore_csk_cutter_lines(op: CadOperation, params: Mapping[str, float], var_name: str) -> list[str]:
    diameter = resolve_number(op.params.get("diameter", op.params.get("hole_diameter")), params, 5)
    depth = resolve_number(op.params.get("depth"), params, 20)
    axis = _vec(op.params.get("axis"), params, (0, 0, 1))
    position = _vec(op.params.get("position"), params, (0, 0, 0))
    lines = [f"{var_name} = cq.Workplane('XY').circle({_fmt(diameter / 2)}).extrude({_fmt(depth / 2)}, both=True)"]
    if op.type == "cbore_hole":
        cb_d = resolve_number(op.params.get("cbore_diameter", op.params.get("counterbore_diameter")), params, diameter * 1.8)
        cb_depth = resolve_number(op.params.get("cbore_depth", op.params.get("counterbore_depth")), params, max(depth * 0.4, 1))

        lines.append(
            f"{var_name} = {var_name}.union("
            f"cq.Workplane('XY').circle({_fmt(cb_d / 2)}).extrude({_fmt(cb_depth)})"
            f".translate((0, 0, {_fmt(depth / 2 - cb_depth)})))"
        )
    else:
        cs_d = resolve_number(op.params.get("csk_diameter", op.params.get("countersink_diameter")), params, diameter * 1.8)
        cs_depth = resolve_number(op.params.get("csk_depth"), params, max((cs_d - diameter) / 2, 0.5))
        lines.append(
            f"{var_name} = {var_name}.union("
            f"cq.Solid.makeCone({_fmt(diameter / 2)}, {_fmt(cs_d / 2)}, {_fmt(max(cs_depth, 0.5))})"
            f".translate(cq.Vector(0, 0, {_fmt(depth / 2 - cs_depth)})))"
        )
    lines.extend(_orient_and_place_lines(var_name, axis, position))
    return lines


def compile_cut_operation(op: CadOperation, params: Mapping[str, float], index: int) -> list[str]:
    if not op.enabled:
        return []

    if op.type in {"cbore_hole", "csk_hole"}:
        var_name = f"cut_{index}_0"
        lines = _cbore_csk_cutter_lines(op, params, var_name)
        lines.append(f"result = result.cut({var_name})")
        return lines

    if op.type not in {"hole", "hole_pattern"}:
        return []
    diameter = resolve_number(op.params.get("diameter", op.params.get("hole_diameter")), params, 5)
    depth = resolve_number(op.params.get("depth"), params, 20)
    axis = _vec(op.params.get("axis"), params, (0, 0, 1))
    positions = [op.params.get("position")] if op.type == "hole" else op.params.get("positions")
    if not isinstance(positions, list):
        return []
    lines: list[str] = []
    for pos_index, raw_pos in enumerate(positions):
        var_name = f"cut_{index}_{pos_index}"
        lines.extend(_hole_cutter_lines(var_name, _vec(raw_pos, params, (0, 0, 0)), axis, diameter, depth))
        lines.append(f"result = result.cut({var_name})")
    return lines


def compile_cadquery(document: CadDocument, export_step: bool = True) -> CadQueryCompileResult:
    result = CadQueryCompileResult(source="")
    params = param_context(document.params)
    lines = [
        "# TextoCAD deterministic CadQuery source",
        "# Generated only from validated canonical CAD JSON.",
        "# directLlmCodeExecution=false",
        "import cadquery as cq",
        "from pathlib import Path",
        "",
        "def make_box(x, y, z):",
        "    return cq.Workplane('XY').box(x, y, z)",
        "",
        "def make_cylinder(radius, height):",
        "    return cq.Workplane('XY').circle(radius).extrude(height / 2, both=True)",
        "",
        "def make_sphere(radius):",
        "    return cq.Workplane('XY').sphere(radius)",
        "",
        "def make_cone(radius1, radius2, height):",
        "    # Wrap the low-level Solid in a Workplane so it exposes the same",
        "    # boolean/fillet/edge API as the other primitives (.union, .cut, .edges...).",
        "    cone = cq.Solid.makeCone(radius1, radius2, height).translate(cq.Vector(0, 0, -height / 2))",
        "    return cq.Workplane('XY').add(cone)",
        "",
        "def make_text(text, size, height):",
        "    try:",
        "        return cq.Workplane('XY').text(text, size, height, combine=True)",
        "    except Exception:",
        "        return cq.Workplane('XY').box(max(len(text) * size * 0.55, size), size * 0.15, height)",
        "",
        "def make_spur_gear(module, teeth, width, bore, helix_angle=0.0):",
        "    from cq_gears import SpurGear",
        "    g = SpurGear(module=module, teeth_number=int(round(teeth)), width=width,",
        "                 bore_d=max(bore, 0.0), helix_angle=helix_angle)",
        "    return cq.Workplane('XY').gear(g)",
        "",
        "def apply_transform(obj, translate=(0, 0, 0), rotate=(0, 0, 0)):",
        "    if rotate != (0, 0, 0):",
        "        obj = obj.rotate((0, 0, 0), (1, 0, 0), rotate[0]).rotate((0, 0, 0), (0, 1, 0), rotate[1]).rotate((0, 0, 0), (0, 0, 1), rotate[2])",
        "    if translate != (0, 0, 0):",
        "        obj = obj.translate(translate)",
        "    return obj",
        "",
        "def apply_boolean(base, tool, op):",
        "    if base is None:",
        "        return tool",
        "    if op == 'cut':",
        "        return base.cut(tool)",
        "    if op == 'intersect':",
        "        return base.intersect(tool)",
        "    return base.union(tool)",
        "",
        "def apply_fillet(obj, radius):",
        "    # Robust edge selection: try all edges, then axis-aligned subsets. CRITICAL",
        "    # (C4): a requested fillet that cannot apply must RAISE, never silently no-op",
        "    # and return the unfilleted solid - a silent skip is the failure this gate",
        "    # exists to prevent. The verification gate independently confirms rounds exist.",
        "    last_exc = None",
        "    for selector in (None, '|Z', '|X', '|Y'):",
        "        try:",
        "            edges = obj.edges() if selector is None else obj.edges(selector)",
        "            if edges.size() == 0:",
        "                continue",
        "            out = edges.fillet(radius)",
        "            # Reject a result that 'succeeded' but produced an invalid solid",
        "            # (OCCT can return a self-intersecting body without raising).",
        "            if not out.val().isValid():",
        "                last_exc = 'fillet produced an invalid solid'",
        "                continue",
        "            return out",
        "        except Exception as exc:",
        "            last_exc = exc",
        "            continue",
        "    raise ValueError('fillet could not be applied (no edge set accepted radius '",
        "                     + str(radius) + '): ' + str(last_exc))",
        "",
        "def apply_chamfer(obj, distance):",
        "    last_exc = None",
        "    for selector in (None, '|Z', '|X', '|Y'):",
        "        try:",
        "            edges = obj.edges() if selector is None else obj.edges(selector)",
        "            if edges.size() == 0:",
        "                continue",
        "            out = edges.chamfer(distance)",
        "            if not out.val().isValid():",
        "                last_exc = 'chamfer produced an invalid solid'",
        "                continue",
        "            return out",
        "        except Exception as exc:",
        "            last_exc = exc",
        "            continue",
        "    raise ValueError('chamfer could not be applied (no edge set accepted distance '",
        "                     + str(distance) + '): ' + str(last_exc))",
        "",
        "def apply_shell(obj, thickness):",
        "    for selector in ('>Z', '<Z', '>Y', '>X'):",
        "        try:",
        "            return obj.faces(selector).shell(-abs(thickness))",
        "        except Exception:",
        "            continue",
        "    try:",
        "        return obj.shell(-abs(thickness))",
        "    except Exception:",
        "        print('TextoCAD warning: shell skipped (geometry too complex)')",
        "        return obj",
        "",
        "def apply_hole(obj, cutter):",
        "    return obj.cut(cutter)",
        "",
        "def apply_pattern(items):",
        "    result = None",
        "    for item in items:",
        "        result = item if result is None else result.union(item)",
        "    return result",
        "",
        "result = None",
        "objects = {}",
    ]

    ops_by_id = {op.id: op for op in document.operations}
    solid_index = 0
    for op in document.operations:
        op_lines = compile_positive_operation(op, params, f"solid_{solid_index}", result, ops_by_id)
        if not op_lines:
            continue
        lines.extend(op_lines)
        if solid_index == 0:
            lines.append("result = solid_0")
        else:
            lines.append(f"result = result.union(solid_{solid_index})")
        lines.append(f"objects[{json.dumps(op.id)}] = solid_{solid_index}")
        solid_index += 1

    if solid_index == 0:
        result.warnings.append("No supported positive operations were found; emitted a safe cube fallback.")
        lines.append("result = cq.Workplane('XY').box(20, 20, 20)")

    for index, op in enumerate(document.operations):
        lines.extend(compile_cut_operation(op, params, index))
        if op.type in {"fillet", "chamfer"}:
            radius = resolve_number(op.params.get("radius", op.params.get("distance")), params, 0)
            if radius > 0:
                method = "fillet" if op.type == "fillet" else "chamfer"


                lines.append(f"result = apply_{method}(result, {_fmt(radius)})")
            else:
                result.warnings.append(f"{op.id}: {op.type} radius/distance was not positive; skipped.")
        elif op.type == "shell":
            thickness = resolve_number(
                op.params.get("thickness", op.params.get("distance", op.params.get("wall"))), params, 0
            )
            if thickness > 0:
                lines.append("try:")
                lines.append(f"    result = apply_shell(result, {_fmt(thickness)})")
                lines.append("except Exception as exc:")
                lines.append(f"    print('TextoCAD warning: {op.id} shell failed:', exc)")
            else:
                result.warnings.append(f"{op.id}: shell thickness was not positive; skipped.")

    export_lines = [
        "",
        "build_dir = Path(__file__).resolve().parent",
    ]
    if export_step:
        export_lines.append("cq.exporters.export(result, str(build_dir / 'result.step'))")
    export_lines.append("cq.exporters.export(result, str(build_dir / 'result.stl'))")
    export_lines.append("print('TextoCAD CadQuery build completed')")
    lines.extend(export_lines)
    result.source = "\n".join(lines) + "\n"
    result.unsupported_operations = sorted(set(result.unsupported_operations))
    result.support_summary = operation_support_summary(document.operations, "cadquery")
    result.support_summary["unsupported"] = sorted(set(result.unsupported_operations))
    return result

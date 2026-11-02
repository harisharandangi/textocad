"""Robotics: a robot is planned joints-first, then modelled link by link."""

from __future__ import annotations

import ast
import json
import math
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textocad.services.codegen_sandbox import ExecResult
from textocad.services.provider_errors import ProviderError

FORMAT = "textocad.robot.v1"
KERNELS = ("openscad", "cadquery")
MAX_LINKS = 16
MOVABLE = {"revolute", "continuous", "prismatic"}
JOINT_TYPES = MOVABLE | {"fixed"}
DENSITY_KG_M3 = 1240.0


def _log(message: str) -> None:
    print(f"[textocad.robot] {message}", flush=True)


_FUNCS = {
    "sin": lambda d: math.sin(math.radians(d)), "cos": lambda d: math.cos(math.radians(d)),
    "tan": lambda d: math.tan(math.radians(d)), "sqrt": math.sqrt, "abs": abs,
    "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "PI": math.pi}


def eval_expr(expr: Any, params: dict[str, float]) -> float:
    """A number from a number or an arithmetic expression over parameter names."""
    if isinstance(expr, (int, float)) and not isinstance(expr, bool):
        return float(expr)
    text = str(expr).strip()
    if not text:
        return 0.0
    if len(text) > 200:
        raise ValueError(f"expression too long: {text[:40]}...")
    node = ast.parse(text, mode="eval")

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            if n.id in params:
                return float(params[n.id])
            if n.id in _CONSTS:
                return _CONSTS[n.id]
            raise ValueError(f"unknown name '{n.id}' in '{text}'")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b if b else 0.0
            if isinstance(n.op, ast.Mod):
                return a % b if b else 0.0
            if abs(b) > 8:
                raise ValueError("exponent too large")
            return a ** b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return float(_FUNCS[n.func.id](*[ev(a) for a in n.args]))
        raise ValueError(f"unsupported expression '{text}'")

    value = ev(node)
    if not math.isfinite(value):
        raise ValueError(f"expression '{text}' is not finite")
    return value


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: list[Any]
    rpy: list[Any]
    axis: list[float]
    lower: Any = None
    upper: Any = None


@dataclass
class RobotSpec:
    name: str
    params: list[dict[str, Any]]
    links: list[dict[str, Any]]
    joints: list[Joint]
    root: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def defaults(self) -> dict[str, float]:
        return {p["key"]: float(p.get("value", 0)) for p in self.params}


def _slug(value: Any, fallback: str) -> str:
    clean = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    clean = clean[:40] or fallback
    return f"{fallback}_{clean}" if clean[0].isdigit() else clean


def parse_spec(data: dict[str, Any]) -> RobotSpec:
    """Validate and normalise a planner spec."""
    errors: list[str] = []
    if not isinstance(data, dict):
        raise ValueError("robot spec must be a JSON object")
    params: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for raw in data.get("params") or []:
        if not isinstance(raw, dict):
            continue
        key = _slug(raw.get("key"), "p")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        try:
            value = float(raw.get("value", 0))
        except (TypeError, ValueError):
            errors.append(f"param {key} has a non-numeric value")
            continue
        lo = float(raw.get("min", min(0.0, value)))
        hi = float(raw.get("max", max(value * 2, value + 10)))
        params.append({
            "key": key, "label": str(raw.get("label") or key.replace("_", " ").capitalize()),
            "type": "number", "value": value, "min": lo, "max": hi,
            "step": float(raw.get("step") or (1.0 if hi - lo >= 10 else 0.1)),
            "unit": str(raw.get("unit") or "mm"), "group": str(raw.get("group") or "Robot"),
        })
    links: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in (data.get("links") or [])[: MAX_LINKS + 1]:
        name = _slug(raw.get("name") if isinstance(raw, dict) else raw, "link")
        if name in names:
            errors.append(f"duplicate link name {name}")
            continue
        names.add(name)
        links.append({
            "name": name,
            "description": str((raw or {}).get("description") or "") if isinstance(raw, dict) else "",
            "color": (raw or {}).get("color") if isinstance(raw, dict) else None,
        })
    if len(links) > MAX_LINKS:
        errors.append(f"at most {MAX_LINKS} links")
    if not links:
        errors.append("no links")
    joints: list[Joint] = []
    jnames: set[str] = set()
    children: dict[str, str] = {}
    for raw in data.get("joints") or []:
        if not isinstance(raw, dict):
            continue
        name = _slug(raw.get("name"), "joint")
        jtype = str(raw.get("type") or "fixed").lower()
        parent, child = _slug(raw.get("parent"), "link"), _slug(raw.get("child"), "link")
        if name in jnames:
            errors.append(f"duplicate joint name {name}")
        jnames.add(name)
        if jtype not in JOINT_TYPES:
            errors.append(f"joint {name}: type must be one of {sorted(JOINT_TYPES)}")
        if parent not in names or child not in names:
            errors.append(f"joint {name}: parent/child must be declared links")
        if child in children:
            errors.append(f"link {child} has two parent joints ({children[child]}, {name})")
        children[child] = name
        origin = raw.get("origin") or {}
        xyz = list(origin.get("xyz") or [0, 0, 0])[:3]
        rpy = list(origin.get("rpy") or [0, 0, 0])[:3]
        while len(xyz) < 3:
            xyz.append(0)
        while len(rpy) < 3:
            rpy.append(0)
        axis_raw = raw.get("axis") or [0, 0, 1]
        try:
            axis = [float(a) for a in list(axis_raw)[:3]]
            norm = math.sqrt(sum(a * a for a in axis))
            if norm < 1e-9:
                raise ValueError
            axis = [a / norm for a in axis]
        except (TypeError, ValueError):
            errors.append(f"joint {name}: axis must be 3 numbers, not all zero")
            axis = [0.0, 0.0, 1.0]
        limit = raw.get("limit") or {}
        joints.append(Joint(
            name=name, type=jtype, parent=parent, child=child, xyz=xyz, rpy=rpy, axis=axis,
            lower=limit.get("lower"), upper=limit.get("upper"),
        ))
    roots = [l["name"] for l in links if l["name"] not in children]
    if len(roots) != 1:
        errors.append(f"exactly one root link is required, found {roots or 'none'}")

    parent_of = {j.child: j.parent for j in joints}
    for l in links:
        cur, hops = l["name"], 0
        while cur in parent_of and hops <= len(links):
            cur, hops = parent_of[cur], hops + 1
        if hops > len(links):
            errors.append(f"joint loop through {l['name']}")
            break
    spec = RobotSpec(
        name=_slug(data.get("name"), "robot"), params=params, links=links, joints=joints,
        root=roots[0] if len(roots) == 1 else "", raw=data,
    )
    try:
        values = spec.defaults()
        for j in joints:
            for e in j.xyz + j.rpy + [j.lower, j.upper]:
                if e is not None:
                    eval_expr(e, values)
    except ValueError as exc:
        errors.append(str(exc))
    if errors:
        raise ValueError("robot spec problems: " + "; ".join(dict.fromkeys(errors)))
    return spec


def _rpy_matrix(r: float, p: float, y: float):
    import numpy as np

    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: list[float], angle: float):
    import numpy as np

    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def joint_origin_matrix(j: Joint, values: dict[str, float]):
    """Parent link frame -> joint frame (mm, degrees in the spec)."""
    import numpy as np

    m = np.eye(4)
    m[:3, :3] = _rpy_matrix(*[math.radians(eval_expr(a, values)) for a in j.rpy])
    m[:3, 3] = [eval_expr(a, values) for a in j.xyz]
    return m


def joint_motion_matrix(j: Joint, q: float):
    """Motion about/along the joint axis: q in degrees (revolute) or mm (prismatic)."""
    import numpy as np

    m = np.eye(4)
    if j.type in ("revolute", "continuous"):
        m[:3, :3] = _axis_angle(j.axis, math.radians(q))
    elif j.type == "prismatic":
        m[:3, 3] = [a * q for a in j.axis]
    return m


def forward_kinematics(spec: RobotSpec, values: dict[str, float], q: dict[str, float] | None = None):
    """World transform of every link frame."""
    import numpy as np

    q = q or {}
    world = {spec.root: np.eye(4)}
    by_parent: dict[str, list[Joint]] = {}
    for j in spec.joints:
        by_parent.setdefault(j.parent, []).append(j)
    stack = [spec.root]
    while stack:
        link = stack.pop()
        for j in by_parent.get(link, []):
            world[j.child] = world[link] @ joint_origin_matrix(j, values) @ joint_motion_matrix(j, q.get(j.name, 0.0))
            stack.append(j.child)
    return world


def joint_limits(j: Joint, values: dict[str, float]) -> tuple[float, float]:
    if j.type == "continuous":
        return -180.0, 180.0
    lo = eval_expr(j.lower, values) if j.lower is not None else (-90.0 if j.type == "revolute" else 0.0)
    hi = eval_expr(j.upper, values) if j.upper is not None else (90.0 if j.type == "revolute" else 50.0)
    return (min(lo, hi), max(lo, hi))


_PLAN_SYS = """You are a robotics engineer. Plan a robot's KINEMATIC STRUCTURE before any
geometry exists. Reply with ONLY a JSON object:

{
  "name": "snake_case_robot_name",
  "design_notes": "1-3 sentences: what the robot is, its DOF and key sizes",
  "params": [ {"key":"base_height","label":"Base height","value":40,"min":20,"max":120,"step":1,"unit":"mm"} ],
  "links": [ {"name":"base_link","description":"round base, 80 mm dia, 40 mm tall","color":"#4a7bd0"} ],
  "joints": [
    {"name":"shoulder_pan","type":"revolute","parent":"base_link","child":"turret",
     "origin":{"xyz":["0","0","base_height"],"rpy":[0,0,0]},
     "axis":[0,0,1],"limit":{"lower":-170,"upper":170}}
  ]
}

RULES
- Units: millimetres and DEGREES (prismatic limits in mm). Z is up. The root link sits
  on the ground (z = 0 is its bottom).
- "origin" is where the CHILD link's frame sits, expressed in the PARENT link's frame.
  The child link will be modelled with that joint at its own origin, so origins must
  be exactly where the physical pivot/axle/slide is. xyz entries may be numbers or
  arithmetic over param keys ("upper_arm_length", "base_height/2 + 5").
- Every dimension that sets a joint position is a param, so a slider moves the joint
  and the geometry together. 3-12 params.
- Exactly one root link; every other link has exactly one parent joint (a tree).
- type: revolute (with limits), continuous (wheels), prismatic (slides, limits in mm),
  fixed (rigidly attached pieces such as a sensor or gripper finger base).
- axis is in the joint frame: [0,0,1] spins about Z, [0,1,0] pitches, [1,0,0] rolls.
- 2-12 links. Give each link a distinct colour and a description with rough sizes.
- Wheels: continuous joints whose axis is the axle; mobile bases put wheel joints at
  the axle centres.
"""

_GEOM_SCAD = """You model robot links in OpenSCAD for 3D printing. You are given a robot's
kinematic plan. Reply with ONE fenced ```openscad block and nothing else.

CONTRACT
- Do NOT declare the parameters; the host prepends them. Use the parameter names as
  variables directly.
- For EVERY link write `// @part <link_name> #rrggbb <Label>` and then
  `module part_<link_name>() { ... }` (no arguments). Do not call them at top level.
- Model each link in ITS OWN FRAME: the joint that connects it to its parent is at the
  origin [0,0,0], with the joint axis exactly as given. For the root link, the origin
  is the centre of its bottom face on the ground.
- Child joints of a link are listed with their positions in that link's frame. Put
  the mounting feature (a boss, a fork, an axle seat, a flange) exactly there so the
  child touches this link at the joint. Leave ~0.5 mm clearance between a link and
  its children so they do not interpenetrate, but do NOT leave visible gaps.
- Rotating parts must clear their neighbours through the whole joint range: keep
  material away from the swept volume (round housings around revolute axes help).
- Printable: walls >= 1.5 mm, everything a closed solid, $fn <= 96.
- Only library allowed: include <BOSL2/std.scad>. No import()/surface().
"""

_GEOM_CQ = """You model robot links with CadQuery. You are given a robot's kinematic plan.
Reply with ONLY a fenced ```python block defining build(params):

```python
def build(params):
    import cadquery as cq
    ...
    return {"base_link": (base, "#4a7bd0"), "upper_arm": (arm, "#e07b39")}
```

CONTRACT
- Return a dict with EVERY link name as a key; values are (Workplane, "#rrggbb").
- Read sizes from params['key'] (the plan's parameter keys).
- Model each link in ITS OWN FRAME: the joint that connects it to its parent is at the
  origin (0,0,0), with the joint axis exactly as given. For the root link, the origin
  is the centre of its bottom face on the ground.
- Child joints of a link are listed with their positions in that link's frame. Put
  the mounting feature (a boss, a fork, an axle seat, a flange) exactly there so the
  child touches this link at the joint. Leave ~0.5 mm clearance so links do not
  interpenetrate, but do NOT leave visible gaps.
- Rotating parts must clear their neighbours through the whole joint range.
- import only cadquery (as cq), cq_gears, math. No file access, no top-level code
  other than the def and imports. Wrap each fillet in try/except; never wrap holes.
"""


def _llm_json(system: str, user: str, image: dict[str, Any] | None = None) -> dict[str, Any]:
    from textocad.services.codegen_pipeline import _extract_json
    from textocad.services.llm_service import multimodal_generate

    raw = multimodal_generate(system, user, image=image, temperature=0.2)
    return _extract_json(raw)


def plan_robot(
    prompt: str,
    *,
    prior: dict[str, Any] | None = None,
    error: str | None = None,
    image: dict[str, Any] | None = None,
) -> RobotSpec:
    user = prompt
    if prior:
        user = (
            "You are EDITING an existing robot. Current plan:\n"
            f"{json.dumps(prior.get('spec'), indent=1)}\n\nRequest: {prompt}\n"
            "Return the COMPLETE updated plan. Keep existing link/joint/param names."
        )
    if error:
        user += f"\n\nThe previous plan was rejected:\n{error}\nFix exactly these problems."
    data = _llm_json(_PLAN_SYS, user, image=image)
    return parse_spec(data)


def _describe_for_geometry(spec: RobotSpec) -> str:
    values = spec.defaults()
    lines = [f"Robot: {spec.name}", "Parameters (use these names):"]
    for p in spec.params:
        lines.append(f"  {p['key']} = {p['value']:g} {p['unit']}  ({p['label']})")
    lines.append("Links (model each in its own frame, parent joint at the origin):")
    by_parent: dict[str, list[Joint]] = {}
    parent_joint = {j.child: j for j in spec.joints}
    for j in spec.joints:
        by_parent.setdefault(j.parent, []).append(j)
    for link in spec.links:
        name = link["name"]
        pj = parent_joint.get(name)
        how = (
            f"attached to {pj.parent} by {pj.type} joint '{pj.name}' (axis {pj.axis} at its origin)"
            if pj else "ROOT link, origin = centre of its bottom face on the ground"
        )
        lines.append(f"- {name} [{link.get('color') or 'pick a colour'}]: {link['description']}; {how}")
        for cj in by_parent.get(name, []):
            xyz = ", ".join(str(a) for a in cj.xyz)
            at = ", ".join(f"{eval_expr(a, values):g}" for a in cj.xyz)
            rpy = ", ".join(str(a) for a in cj.rpy)
            lo_hi = ""
            if cj.type in MOVABLE:
                lo, hi = joint_limits(cj, values)
                lo_hi = f", range {lo:g}..{hi:g}"
            lines.append(
                f"    child joint '{cj.name}' ({cj.type}) -> {cj.child} at [{xyz}] (= [{at}] mm now), "
                f"rpy [{rpy}] deg, axis {cj.axis}{lo_hi}"
            )
    return "\n".join(lines)


def _scad_header(spec: RobotSpec) -> str:
    rows = ["/* [Robot] */"]
    for p in spec.params:
        rows.append(f"{p['key']} = {p['value']:g}; // [{p['min']:g}:{p['step']:g}:{p['max']:g}] {p['label']}")
    rows.append("/* [Hidden] */")
    rows.append("$fn = 64;")
    return "\n".join(rows) + "\n"


def write_geometry(
    spec: RobotSpec, kernel: str, prompt: str, *, prior_geometry: str | None = None,
    error: str | None = None,
) -> str:
    from textocad.services.codegen_pipeline import _extract_scad
    from textocad.services.codegen_sandbox import validate_script
    from textocad.services.llm_service import multimodal_generate

    system = _GEOM_SCAD if kernel == "openscad" else _GEOM_CQ
    user = f"Request: {prompt}\n\n{_describe_for_geometry(spec)}"
    if prior_geometry:
        lang = "openscad" if kernel == "openscad" else "python"
        user += f"\n\nCurrent geometry (update it, keep what still fits):\n```{lang}\n{prior_geometry}\n```"
    if error:
        user += f"\n\nThe previous geometry had these problems:\n{error}\nFix them exactly."
    raw = multimodal_generate(system, user, temperature=0.2)
    if kernel == "openscad":
        script = _extract_scad(raw)
        script = re.sub(r"^\s*/\*\s*\[Robot\]\s*\*/.*?\$fn\s*=\s*\d+\s*;\s*\n", "", script, flags=re.S)
        from textocad.services.openscad_engine import validate_scad

        errs = validate_scad(_scad_header(spec) + script)
        if errs:
            raise ValueError("geometry failed validation: " + "; ".join(errs))
        return script
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", raw, re.S)
    script = (m.group(1) if m else raw).strip()
    v = validate_script(script, "precision")
    if not v.ok:
        raise ValueError("geometry failed AST validation: " + "; ".join(v.errors))
    return script


def _compile_links(kernel: str, spec: RobotSpec, geometry: str, values: dict[str, float], work: Path) -> ExecResult:
    if kernel == "openscad":
        from textocad.services.openscad_engine import compile_model

        source = _scad_header(spec) + geometry
        return compile_model(source, values, work, schema=spec.params, keep_placement=False)
    from textocad.services.codegen_sandbox import execute_script

    return execute_script(geometry, values, "precision", build_dir=str(work), timeout_s=180)


def _sample_points(mesh, n: int = 2500):
    import numpy as np

    if len(mesh.faces) == 0:
        return np.zeros((0, 3))
    pts, _ = mesh.sample(n, return_index=True) if hasattr(mesh, "sample") else (mesh.vertices, None)
    return np.vstack([pts, mesh.vertices[: min(len(mesh.vertices), 2000)]])


def _min_distance(a, b) -> float:
    import numpy as np

    if len(a) == 0 or len(b) == 0:
        return float("inf")
    try:
        from scipy.spatial import cKDTree

        d, _ = cKDTree(b).query(a, k=1)
        return float(d.min())
    except Exception:
        best = float("inf")
        for chunk in np.array_split(a, max(1, len(a) // 500)):
            dd = np.linalg.norm(chunk[:, None, :] - b[None, :, :], axis=2)
            best = min(best, float(dd.min()))
        return best


def _overlap_volume(a, b) -> float:
    """Volume of a ∩ b (mm³). 0 when it cannot be computed."""
    try:
        import manifold3d as mf
        import numpy as np

        def to_manifold(m):
            mesh = mf.Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32),
                           tri_verts=np.asarray(m.faces, dtype=np.uint32))
            return mf.Manifold(mesh)

        inter = to_manifold(a) ^ to_manifold(b)
        vol = inter.volume() if hasattr(inter, "volume") else inter.get_volume()
        return float(vol)
    except Exception:
        return 0.0


def assemble_and_check(spec_raw: dict[str, Any], values: dict[str, float], parts_dir: str, out_dir: str) -> dict[str, Any]:
    """Runs in an isolated child."""
    import numpy as np
    import trimesh

    from textocad.services.part_export import normalize_color, threemf_bytes

    spec = parse_spec(spec_raw)
    pdir, out = Path(parts_dir), Path(out_dir)
    manifest = {p["name"]: p for p in json.loads((pdir / "parts.json").read_text("utf-8"))}
    meshes: dict[str, Any] = {}
    issues: list[str] = []
    for i, link in enumerate(spec.links):
        name = link["name"]
        f = pdir / f"{name}.stl"
        if not f.exists():
            issues.append(f"link {name} was not modelled (no part_{name})")
            continue
        m = trimesh.load(str(f), force="mesh")
        meshes[name] = m
        link["color"] = normalize_color(manifest.get(name, {}).get("color") or link.get("color"), i, len(spec.links))
    missing = [l["name"] for l in spec.links if l["name"] not in meshes]
    if missing:
        raise ValueError("links without geometry: " + ", ".join(missing))

    world = forward_kinematics(spec, values)
    placed = {n: meshes[n].copy().apply_transform(world[n]) for n in meshes}
    size = max(float(np.max(trimesh.util.concatenate(list(placed.values())).extents)), 1.0)
    touch_tol = max(1.5, 0.02 * size)
    samples = {n: _sample_points(placed[n]) for n in placed}

    parent_of = {j.child: j for j in spec.joints}

    for j in spec.joints:
        gap = _min_distance(samples[j.child], samples[j.parent])
        if gap > touch_tol:
            issues.append(
                f"link {j.child} floats {gap:.1f} mm away from {j.parent}; it must touch "
                f"{j.parent} at joint '{j.name}' (its origin)"
            )
        origin_w = world[j.child][:3, 3][None, :]
        d_child = _min_distance(origin_w, samples[j.child])
        d_parent = _min_distance(origin_w, samples[j.parent])
        if d_child > touch_tol * 2:
            issues.append(
                f"link {j.child} is {d_child:.1f} mm from its own origin; model it so its "
                f"pivot/mount for joint '{j.name}' is at [0,0,0]"
            )
        if d_parent > touch_tol * 2:
            issues.append(
                f"{j.parent} has no material near joint '{j.name}' ({d_parent:.1f} mm away); "
                f"add the mount at the joint position"
            )

    names = list(placed)
    vols = {n: max(abs(float(placed[n].volume)), 1e-6) for n in names}
    adjacent = {frozenset((j.parent, j.child)) for j in spec.joints}

    def check_pose(pl: dict[str, Any], label: str) -> None:
        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                a, b = names[a_i], names[b_i]
                ov = _overlap_volume(pl[a], pl[b])
                small = min(vols[a], vols[b])
                if frozenset((a, b)) in adjacent:
                    if ov > 0.15 * small and ov > 5:
                        issues.append(f"{a} and {b} interpenetrate by {ov:.0f} mm³{label}; add clearance at their joint")
                elif ov > max(1.0, 0.01 * small):
                    issues.append(f"{a} collides with {b} ({ov:.0f} mm³){label}")

    check_pose(placed, "")

    for j in spec.joints:
        if j.type not in MOVABLE or len(issues) > 12:
            continue
        lo, hi = joint_limits(j, values)
        for q in (lo, hi):
            if abs(q) < 1e-6:
                continue
            w = forward_kinematics(spec, values, {j.name: q})
            pl = {n: meshes[n].copy().apply_transform(w[n]) for n in meshes}
            before = len(issues)
            check_pose(pl, f" when {j.name} = {q:g}{'°' if j.type != 'prismatic' else ' mm'}")


            issues[before:] = [x for x in issues[before:] if "interpenetrate" not in x]
    issues = list(dict.fromkeys(issues))


    out.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene(base_frame="world")
    scene.graph.update(frame_from="world", frame_to=spec.root, matrix=np.eye(4))
    by_parent: dict[str, list[Joint]] = {}
    for j in spec.joints:
        by_parent.setdefault(j.parent, []).append(j)

    from textocad.services.part_export import _colored

    color = {l["name"]: l["color"] for l in spec.links}
    stack = [spec.root]
    while stack:
        link = stack.pop()
        scene.add_geometry(
            _colored(meshes[link], color[link], link), node_name=f"{link}__mesh",
            geom_name=link, parent_node_name=link,
        )
        for j in by_parent.get(link, []):
            jn = f"joint__{j.name}"
            scene.graph.update(frame_from=link, frame_to=jn, matrix=joint_origin_matrix(j, values))
            scene.graph.update(frame_from=jn, frame_to=j.child, matrix=np.eye(4))
            stack.append(j.child)
    (out / "result.glb").write_bytes(scene.export(file_type="glb"))


    assembled_dir = out / "parts"
    if assembled_dir.exists():
        shutil.rmtree(assembled_dir)
    assembled_dir.mkdir()
    facts = []
    for n in names:
        placed[n].export(str(assembled_dir / f"{n}.stl"))
        ext = placed[n].extents
        facts.append({
            "name": n, "label": n.replace("_", " ").capitalize(), "color": color[n],
            "volume": round(vols[n], 3), "bbox": [round(float(v), 3) for v in ext],
            "triangles": int(len(placed[n].faces)), "watertight": bool(placed[n].is_watertight),

            "hasStep": False,
        })
    (assembled_dir / "parts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    whole = trimesh.util.concatenate([placed[n] for n in names])
    whole.export(str(out / "result.stl"))
    (out / "result.3mf").write_bytes(threemf_bytes([(n, color[n], placed[n]) for n in names]))

    urdf = write_robot_package(spec, values, meshes, color, out)
    ext = whole.extents
    joints_out = []
    for j in spec.joints:
        lo, hi = joint_limits(j, values)
        o = joint_origin_matrix(j, values)
        joints_out.append({
            "name": j.name, "type": j.type, "parent": j.parent, "child": j.child,
            "axis": j.axis,
            "origin": {"xyz": [float(v) for v in o[:3, 3]],
                       "rpy": [math.radians(eval_expr(a, values)) for a in j.rpy]},
            "limit": ({"lower": math.radians(lo), "upper": math.radians(hi)} if j.type in ("revolute", "continuous")
                      else {"lower": lo, "upper": hi} if j.type == "prismatic" else None),
            "node": f"joint__{j.name}",
        })
    return {
        "issues": issues,
        "parts": facts,
        "bbox": [float(ext[0]), float(ext[1]), float(ext[2])],
        "volume": float(sum(vols.values())),
        "robot": {
            "robotName": spec.name, "baseLink": spec.root, "units": "mm",
            "links": [{"name": l["name"], "color": color[l["name"]]} for l in spec.links],
            "joints": joints_out,
        },
        **urdf,
    }


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def write_robot_package(spec: RobotSpec, values: dict[str, float], meshes: dict[str, Any], color: dict[str, str], out: Path) -> dict[str, Any]:
    """URDF/Xacro + ROS 2 package."""
    import numpy as np

    pkg = spec.name
    root = out / "ros_package"
    if root.exists():
        shutil.rmtree(root)
    (root / "meshes").mkdir(parents=True)
    (root / "urdf").mkdir()
    (root / "launch").mkdir()
    link_xml: list[str] = []
    for l in spec.links:
        n = l["name"]
        m = meshes[n]
        m.export(str(root / "meshes" / f"{n}.stl"))
        hull = m.convex_hull
        hull.export(str(root / "meshes" / f"{n}_collision.stl"))

        try:
            mm = m.copy()
            mm.apply_scale(0.001)
            mm.density = DENSITY_KG_M3
            mass = max(float(mm.mass), 1e-4) if mm.is_watertight else max(float(abs(mm.volume)) * DENSITY_KG_M3, 1e-4)
            com = mm.center_mass if mm.is_watertight else mm.centroid
            inertia = mm.moment_inertia if mm.is_watertight else np.eye(3) * mass * (max(mm.extents) ** 2) / 6
        except Exception:
            mass, com, inertia = 0.05, np.zeros(3), np.eye(3) * 1e-5
        r, g, b = (int(color[n][i:i + 2], 16) / 255 for i in (1, 3, 5))
        link_xml.append(
            f'  <link name="{n}">\n'
            f'    <visual>\n      <geometry><mesh filename="package://{pkg}/meshes/{n}.stl" scale="0.001 0.001 0.001"/></geometry>\n'
            f'      <material name="{n}_color"><color rgba="{r:.3f} {g:.3f} {b:.3f} 1"/></material>\n    </visual>\n'
            f'    <collision>\n      <geometry><mesh filename="package://{pkg}/meshes/{n}_collision.stl" scale="0.001 0.001 0.001"/></geometry>\n    </collision>\n'
            f'    <inertial>\n      <origin xyz="{_fmt(com[0])} {_fmt(com[1])} {_fmt(com[2])}" rpy="0 0 0"/>\n'
            f'      <mass value="{_fmt(mass)}"/>\n'
            f'      <inertia ixx="{_fmt(inertia[0][0])}" ixy="{_fmt(inertia[0][1])}" ixz="{_fmt(inertia[0][2])}" '
            f'iyy="{_fmt(inertia[1][1])}" iyz="{_fmt(inertia[1][2])}" izz="{_fmt(inertia[2][2])}"/>\n'
            f'    </inertial>\n  </link>'
        )
    joint_xml: list[str] = []
    for j in spec.joints:
        xyz = [eval_expr(a, values) / 1000.0 for a in j.xyz]
        rpy = [math.radians(eval_expr(a, values)) for a in j.rpy]
        body = (
            f'  <joint name="{j.name}" type="{j.type}">\n'
            f'    <parent link="{j.parent}"/>\n    <child link="{j.child}"/>\n'
            f'    <origin xyz="{" ".join(_fmt(v) for v in xyz)}" rpy="{" ".join(_fmt(v) for v in rpy)}"/>\n'
        )
        if j.type != "fixed":
            body += f'    <axis xyz="{" ".join(_fmt(v) for v in j.axis)}"/>\n'
        if j.type == "revolute":
            lo, hi = joint_limits(j, values)
            body += f'    <limit lower="{_fmt(math.radians(lo))}" upper="{_fmt(math.radians(hi))}" effort="5" velocity="2"/>\n'
        elif j.type == "prismatic":
            lo, hi = joint_limits(j, values)
            body += f'    <limit lower="{_fmt(lo / 1000)}" upper="{_fmt(hi / 1000)}" effort="20" velocity="0.1"/>\n'
        joint_xml.append(body + "  </joint>")
    header = (
        f"<!-- Generated by TextoCAD. Units: metres, radians. Each mesh is in its link's own\n"
        f"     frame (millimetres, scaled by 0.001); joint origins come from the robot plan.\n"
        f"     Masses/inertias are estimates for PLA ({DENSITY_KG_M3:g} kg/m^3). -->\n"
    )
    urdf = (
        '<?xml version="1.0"?>\n' + header + f'<robot name="{pkg}">\n'
        + "\n".join(link_xml) + "\n" + "\n".join(joint_xml) + "\n</robot>\n"
    )
    props = "\n".join(
        f'  <xacro:property name="{p["key"]}" value="{_fmt(values.get(p["key"], p["value"]))}"/>'
        for p in spec.params
    )
    xacro = (
        '<?xml version="1.0"?>\n' + header
        + f'<robot name="{pkg}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
        + "  <!-- Design parameters (mm). The geometry below was built with these values. -->\n"
        + props + "\n" + "\n".join(link_xml) + "\n" + "\n".join(joint_xml) + "\n</robot>\n"
    )
    _validate_urdf(urdf, spec)
    (root / "urdf" / f"{pkg}.urdf").write_text(urdf, encoding="utf-8")
    (root / "urdf" / f"{pkg}.urdf.xacro").write_text(xacro, encoding="utf-8")
    (root / "package.xml").write_text(
        f'<?xml version="1.0"?>\n<package format="3">\n  <name>{pkg}</name>\n  <version>0.1.0</version>\n'
        f"  <description>{pkg} robot description generated by TextoCAD</description>\n"
        "  <maintainer email=\"user@example.com\">TextoCAD user</maintainer>\n  <license>Proprietary</license>\n"
        "  <buildtool_depend>ament_cmake</buildtool_depend>\n  <exec_depend>robot_state_publisher</exec_depend>\n"
        "  <exec_depend>joint_state_publisher_gui</exec_depend>\n  <exec_depend>rviz2</exec_depend>\n  <exec_depend>xacro</exec_depend>\n"
        "  <export><build_type>ament_cmake</build_type></export>\n</package>\n",
        encoding="utf-8",
    )
    (root / "CMakeLists.txt").write_text(
        f"cmake_minimum_required(VERSION 3.8)\nproject({pkg})\nfind_package(ament_cmake REQUIRED)\n"
        "install(DIRECTORY urdf meshes launch DESTINATION share/${PROJECT_NAME})\nament_package()\n",
        encoding="utf-8",
    )
    (root / "launch" / "display.launch.py").write_text(
        "import os\nfrom ament_index_python.packages import get_package_share_directory\n"
        "from launch import LaunchDescription\nfrom launch_ros.actions import Node\n\n\n"
        "def generate_launch_description():\n"
        f"    urdf = os.path.join(get_package_share_directory('{pkg}'), 'urdf', '{pkg}.urdf')\n"
        "    with open(urdf, encoding='utf-8') as fh:\n        description = fh.read()\n"
        "    return LaunchDescription([\n"
        "        Node(package='robot_state_publisher', executable='robot_state_publisher',\n"
        "             parameters=[{'robot_description': description}]),\n"
        "        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui'),\n"
        "        Node(package='rviz2', executable='rviz2'),\n    ])\n",
        encoding="utf-8",
    )
    (out / "robot.urdf").write_text(urdf, encoding="utf-8")
    (out / "robot.xacro").write_text(xacro, encoding="utf-8")
    zip_path = out / "robot_ros_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, f"{pkg}/{f.relative_to(root).as_posix()}")
    shutil.rmtree(root, ignore_errors=True)
    return {"urdf_path": str(out / "robot.urdf"), "xacro_path": str(out / "robot.xacro"), "zip_path": str(zip_path)}


def _validate_urdf(urdf: str, spec: RobotSpec) -> None:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf)
    links = {e.get("name") for e in root.findall("link")}
    joints = root.findall("joint")
    if len(links) != len(spec.links) or len(joints) != len(spec.joints):
        raise ValueError("URDF link/joint count does not match the plan")
    for j in joints:
        if j.find("parent").get("link") not in links or j.find("child").get("link") not in links:
            raise ValueError(f"URDF joint {j.get('name')} references an unknown link")


@dataclass
class RobotBuild:
    script: str
    title: str
    param_schema: list[dict[str, Any]]
    exec: ExecResult
    robot: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    urdf_path: str | None = None
    xacro_path: str | None = None
    zip_path: str | None = None
    kernel: str = "openscad"
    notes: list[str] = field(default_factory=list)


def _build_once(spec: RobotSpec, kernel: str, geometry: str, values: dict[str, float], build_dir: Path) -> tuple[ExecResult, dict[str, Any] | None]:
    from textocad.services.cancellation import check as cancel_check
    from textocad.services.geometry_isolation import IsolationError, run_isolated

    cancel_check()
    work = Path(tempfile.mkdtemp(prefix="robot_", dir=str(build_dir)))
    try:
        res = _compile_links(kernel, spec, geometry, values, work)
        if not res.ok:
            return res, None
        try:
            out = run_isolated(assemble_and_check, spec.raw, values, str(work / "parts"), str(build_dir), timeout_s=300)
        except (IsolationError, ValueError) as exc:
            return ExecResult(ok=False, error=f"could not assemble the robot: {exc}"), None
        return ExecResult(
            ok=True,
            stl_path=str(build_dir / "result.stl"),
            glb_path=str(build_dir / "result.glb"),
            threemf_path=str(build_dir / "result.3mf"),
            bbox=tuple(out["bbox"]),
            volume=out["volume"],
            parts=out["parts"],
        ), out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _package_script(kernel: str, spec: RobotSpec, geometry: str) -> str:
    return json.dumps({"format": FORMAT, "kernel": kernel, "spec": spec.raw, "geometry": geometry}, indent=1)


def load_script(script: str) -> dict[str, Any] | None:
    try:
        data = json.loads(script)
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get("format") == FORMAT else None


def generate_robot(
    prompt: str,
    kernel: str,
    build_dir: str,
    *,
    prior_script: str | None = None,
    image: dict[str, Any] | None = None,
    max_geometry_attempts: int = 3,
) -> RobotBuild:
    """Plan -> geometry -> build/check, with repair."""
    kernel = kernel if kernel in KERNELS else "openscad"
    bdir = Path(build_dir)
    prior = load_script(prior_script) if prior_script else None

    spec: RobotSpec | None = None
    plan_error: str | None = None
    for attempt in range(3):
        try:
            spec = plan_robot(prompt, prior=prior, error=plan_error, image=image if attempt == 0 else None)
            break
        except ValueError as exc:
            plan_error = str(exc)
            _log(f"plan attempt {attempt + 1} rejected: {plan_error[:200]}")
    if spec is None:
        raise ValueError(f"could not plan the robot: {plan_error}")
    _log(f"plan ok: {len(spec.links)} links, {len(spec.joints)} joints")

    values = spec.defaults()
    prior_geometry = prior.get("geometry") if prior and prior.get("kernel") == kernel else None
    best: tuple[ExecResult, dict[str, Any], str] | None = None
    error: str | None = None
    for attempt in range(max_geometry_attempts):
        try:
            geometry = write_geometry(spec, kernel, prompt, prior_geometry=prior_geometry, error=error)
        except ProviderError:
            raise
        except Exception as exc:
            error = str(exc)
            _log(f"geometry attempt {attempt + 1} rejected: {error[:200]}")
            continue
        res, out = _build_once(spec, kernel, geometry, values, bdir)
        if not res.ok or out is None:
            error = res.error
            prior_geometry = geometry
            _log(f"geometry attempt {attempt + 1} failed to build: {error[:200]}")
            continue
        issues = out["issues"]
        if best is None or len(issues) < len(best[1]["issues"]):
            best = (res, out, geometry)
        if not issues:
            break
        _log(f"geometry attempt {attempt + 1}: {len(issues)} alignment issue(s)")
        error = "Alignment check found:\n- " + "\n- ".join(issues[:10])
        prior_geometry = geometry
    if best is None:
        raise ValueError(f"could not build the robot geometry: {error}")
    res, out, geometry = best

    final_res, final_out = _build_once(spec, kernel, geometry, values, bdir)
    if final_res.ok and final_out is not None:
        res, out = final_res, final_out
    notes = [f"Robot: {len(spec.links)} links, {len(spec.joints)} joints ({kernel})"]
    if out["issues"]:
        notes.append("Alignment notes: " + "; ".join(out["issues"][:6]))
    else:
        notes.append("Alignment checked: every link meets its parent and no collisions across joint ranges")
    return RobotBuild(
        script=_package_script(kernel, spec, geometry),
        title=spec.name.replace("_", " ").capitalize(),
        param_schema=spec.params, exec=res, robot=out["robot"], issues=out["issues"],
        urdf_path=out.get("urdf_path"), xacro_path=out.get("xacro_path"), zip_path=out.get("zip_path"),
        kernel=kernel, notes=notes,
    )


def rebuild_robot(script: str, params: dict[str, Any], build_dir: str) -> RobotBuild:
    """Slider rebuild: same spec + geometry, new parameter values. No LLM."""
    data = load_script(script)
    if data is None:
        raise ValueError("not a robot script")
    spec = parse_spec(data["spec"])
    kernel = data.get("kernel") if data.get("kernel") in KERNELS else "openscad"
    values = spec.defaults()
    for k, v in (params or {}).items():
        if k in values:
            values[k] = float(v)
    res, out = _build_once(spec, kernel, data["geometry"], values, Path(build_dir))
    if not res.ok or out is None:
        raise ValueError(res.error or "robot rebuild failed")
    return RobotBuild(
        script=script, title=spec.name.replace("_", " ").capitalize(), param_schema=spec.params,
        exec=res, robot=out["robot"], issues=out["issues"], urdf_path=out.get("urdf_path"),
        xacro_path=out.get("xacro_path"), zip_path=out.get("zip_path"), kernel=kernel,
        notes=(["Alignment notes: " + "; ".join(out["issues"][:6])] if out["issues"] else []),
    )

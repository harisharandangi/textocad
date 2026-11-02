"""Organic generation via signed-distance fields (SDF) + marching cubes."""

from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np
import trimesh
from skimage import measure


_MAX_PRIMITIVES = 64
_GRID_N = 96
_TARGET_SIZE_MM = 120.0

_PRIM_TYPES = {"sphere", "ellipsoid", "box", "capsule", "torus", "cylinder"}
_OPS = {
    "union",
    "smooth_union",
    "subtract",
    "smooth_subtract",
    "intersect",
    "smooth_intersect",
}


_SDF_SYS = """You are a 3D organic modeling engine. Describe the requested shape as a
signed-distance-field (SDF) scene built from smooth-blended primitives. This is how
organic, fleshy, smooth forms are made (smooth-union fuses blobs seamlessly).

Return ONLY JSON of this exact shape:
{
  "design_notes": "1-2 sentences: what this is and the overall proportions you chose",
  "title": "short name",
  "material": {"color": [r, g, b]},               // 0..1 floats
  "primitives": [
    {
      "type": "sphere|ellipsoid|box|capsule|torus|cylinder",
      "op": "smooth_union|union|subtract|smooth_subtract|intersect",
      "blend": 0.4,                                  // smoothing radius (world units)
      "position": [x, y, z],
      "rotation": [rx, ry, rz],                      // euler DEGREES, optional
      "size": [a, b, c],                             // meaning per type, see below
      "color": [r, g, b]                             // optional per-part color 0..1
    }
  ]
}

SIZE meaning per type:
- sphere:    [radius]
- ellipsoid: [rx, ry, rz]
- box:       [half_x, half_y, half_z]   (rounded by blend)
- capsule:   [radius, height]           (long axis = Y)
- torus:     [major_radius, tube_radius]
- cylinder:  [radius, height]           (axis = Y)

RULES (follow exactly):
- Incomplete prompt? Infer sensible real-world proportions yourself. "a mouse"
  (animal) -> rounded body ~2 long, head, 2 ears, tail; "a vase" -> tapered
  revolve-like stack of blended discs. Never produce a single bare sphere for a
  named object — capture its recognizable silhouette.
- Work in a coordinate space roughly within [-3, 3] on each axis. Keep the whole
  model centered near the origin.
- The FIRST primitive is the base body (use op "union"). Add features with
  "smooth_union" (blend 0.2-0.7) so they fuse organically. Use "subtract" for
  hollows/mouths/eye sockets.
- Build recognizable forms: a creature = body + head + limbs + ears/eyes, smooth-
  unioned. A mouse (animal) = rounded body + head + 2 ears + tail. An object = a
  few blended blobs that match the silhouette.
- Use 4 to 24 primitives. More primitives + gentle blends = richer shape.
- Numbers only (no expressions). Do NOT output code. Output ONLY the JSON object."""


def _num(value: Any, fallback: float = 0.0) -> float:
    try:
        f = float(value)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _vec3(value: Any, fallback: tuple[float, float, float]) -> list[float]:
    if isinstance(value, (list, tuple)) and value:
        out = [
            _num(value[i] if i < len(value) else fallback[i], fallback[i])
            for i in range(3)
        ]
        return out
    return list(fallback)


def _color(value: Any, fallback: tuple[float, float, float]) -> list[float]:
    c = _vec3(value, fallback)
    return [min(max(x, 0.0), 1.0) for x in c]


def _normalize_primitive(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    ptype = str(raw.get("type") or "sphere").strip().lower()
    if ptype not in _PRIM_TYPES:
        ptype = "sphere"
    op = str(raw.get("op") or "smooth_union").strip().lower()
    if op not in _OPS:
        op = "smooth_union"
    size = _vec3(raw.get("size"), (1.0, 1.0, 1.0))

    size = [max(abs(s), 1e-3) for s in size]
    return {
        "type": ptype,
        "op": op,
        "blend": max(_num(raw.get("blend"), 0.3), 0.0),
        "position": _vec3(raw.get("position"), (0.0, 0.0, 0.0)),
        "rotation": _vec3(raw.get("rotation"), (0.0, 0.0, 0.0)),
        "size": size,
        "color": _color(raw.get("color"), (0.85, 0.55, 0.45)),
    }


def normalize_scene(raw: Any, *, prompt: str = "") -> dict[str, Any]:
    """Coerce arbitrary LLM output into a valid SDF scene."""
    data = raw if isinstance(raw, dict) else {}
    prims_raw = data.get("primitives")
    prims: list[dict[str, Any]] = []
    if isinstance(prims_raw, list):
        for item in prims_raw[:_MAX_PRIMITIVES]:
            p = _normalize_primitive(item)
            if p is not None:
                prims.append(p)
    if not prims:
        return _fallback_scene(prompt)

    prims[0]["op"] = "union"
    return {
        "title": str(data.get("title") or (prompt[:40] or "Organic model")),
        "material": {"color": _color((data.get("material") or {}).get("color"), (0.85, 0.55, 0.45))},
        "primitives": prims,
    }


def _fallback_scene(prompt: str = "") -> dict[str, Any]:
    """A friendly deterministic blob creature so organic generation never fails."""
    return {
        "title": (prompt[:40] or "Organic blob"),
        "material": {"color": [0.85, 0.55, 0.45]},
        "primitives": [
            {"type": "ellipsoid", "op": "union", "blend": 0.0,
             "position": [0, 0, 0], "rotation": [0, 0, 0], "size": [1.6, 1.2, 1.2],
             "color": [0.85, 0.55, 0.45]},
            {"type": "sphere", "op": "smooth_union", "blend": 0.6,
             "position": [0, 1.1, 0.5], "rotation": [0, 0, 0], "size": [0.9, 0, 0],
             "color": [0.85, 0.55, 0.45]},
            {"type": "sphere", "op": "smooth_union", "blend": 0.3,
             "position": [-0.5, 1.8, 0.5], "rotation": [0, 0, 0], "size": [0.35, 0, 0],
             "color": [0.85, 0.55, 0.45]},
            {"type": "sphere", "op": "smooth_union", "blend": 0.3,
             "position": [0.5, 1.8, 0.5], "rotation": [0, 0, 0], "size": [0.35, 0, 0],
             "color": [0.85, 0.55, 0.45]},
            {"type": "capsule", "op": "smooth_union", "blend": 0.4,
             "position": [0, -1.0, 0], "rotation": [0, 0, 0], "size": [0.45, 1.0],
             "color": [0.85, 0.55, 0.45]},
        ],
    }


def _rot_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


def _prim_sdf(prim: dict[str, Any], pts: np.ndarray) -> np.ndarray:
    """SDF of one primitive over points pts (M,3) in world space -> (M,)."""
    pos = np.array(prim["position"], dtype=float)
    rot = _rot_matrix(*prim["rotation"])

    p = (pts - pos) @ rot
    size = prim["size"]
    t = prim["type"]

    if t == "sphere":
        return np.linalg.norm(p, axis=1) - size[0]

    if t == "ellipsoid":
        r = np.array([size[0], size[1], size[2]], dtype=float)
        k0 = np.linalg.norm(p / r, axis=1)
        k1 = np.linalg.norm(p / (r * r), axis=1)
        return k0 * (k0 - 1.0) / np.where(k1 == 0, 1e-9, k1)

    if t == "box":
        b = np.array([size[0], size[1], size[2]], dtype=float)
        q = np.abs(p) - b
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    if t == "capsule":
        radius, height = size[0], size[1]
        py = np.clip(p[:, 1], -height / 2.0, height / 2.0)
        d = p.copy()
        d[:, 1] -= py
        return np.linalg.norm(d, axis=1) - radius

    if t == "torus":
        major, tube = size[0], size[1]
        qx = np.linalg.norm(p[:, [0, 2]], axis=1) - major
        return np.sqrt(qx * qx + p[:, 1] * p[:, 1]) - tube

    if t == "cylinder":
        radius, height = size[0], size[1]
        d_xz = np.linalg.norm(p[:, [0, 2]], axis=1) - radius
        d_y = np.abs(p[:, 1]) - height / 2.0
        outside = np.sqrt(
            np.maximum(d_xz, 0.0) ** 2 + np.maximum(d_y, 0.0) ** 2
        )
        inside = np.minimum(np.maximum(d_xz, d_y), 0.0)
        return outside + inside


    return np.linalg.norm(p, axis=1) - max(size[0], 1e-3)


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 1e-6:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


def _smax(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -_smin(-a, -b, k)


def _scene_sdf(scene: dict[str, Any], pts: np.ndarray) -> np.ndarray:
    acc: np.ndarray | None = None
    for prim in scene["primitives"]:
        d = _prim_sdf(prim, pts)
        k = float(prim.get("blend", 0.0))
        op = prim["op"]
        if acc is None:
            acc = d
            continue
        if op == "union":
            acc = np.minimum(acc, d)
        elif op == "smooth_union":
            acc = _smin(acc, d, k)
        elif op == "subtract":
            acc = np.maximum(acc, -d)
        elif op == "smooth_subtract":
            acc = _smax(acc, -d, k)
        elif op == "intersect":
            acc = np.maximum(acc, d)
        elif op == "smooth_intersect":
            acc = _smax(acc, d, k)
        else:
            acc = _smin(acc, d, k)
    if acc is None:

        return np.full((pts.shape[0],), 1e3)
    return acc


def _scene_bounds(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([1e9, 1e9, 1e9])
    hi = np.array([-1e9, -1e9, -1e9])
    for prim in scene["primitives"]:
        pos = np.array(prim["position"], dtype=float)
        reach = max(prim["size"]) + float(prim.get("blend", 0.0))
        lo = np.minimum(lo, pos - reach)
        hi = np.maximum(hi, pos + reach)
    if not np.all(np.isfinite(lo)):
        lo, hi = np.array([-2.0, -2.0, -2.0]), np.array([2.0, 2.0, 2.0])
    pad = 0.15 * np.maximum(hi - lo, 1e-3) + 0.5
    return lo - pad, hi + pad


def scene_to_mesh(scene: dict[str, Any], *, grid_n: int = _GRID_N) -> trimesh.Trimesh:
    """Evaluate the SDF on a grid and extract the isosurface (marching cubes)."""
    lo, hi = _scene_bounds(scene)
    xs = np.linspace(lo[0], hi[0], grid_n)
    ys = np.linspace(lo[1], hi[1], grid_n)
    zs = np.linspace(lo[2], hi[2], grid_n)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    field = _scene_sdf(scene, pts).reshape((grid_n, grid_n, grid_n))


    if field.min() > 0 or field.max() < 0:
        scene = _fallback_scene(scene.get("title", ""))
        return scene_to_mesh(scene, grid_n=grid_n)

    spacing = (
        (hi[0] - lo[0]) / (grid_n - 1),
        (hi[1] - lo[1]) / (grid_n - 1),
        (hi[2] - lo[2]) / (grid_n - 1),
    )
    verts, faces, normals, _ = measure.marching_cubes(field, level=0.0, spacing=spacing)

    verts = verts + lo

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)


    if mesh.body_count > 1:
        parts = mesh.split(only_watertight=False)
        if parts:
            mesh = max(parts, key=lambda m: m.volume if m.volume else len(m.faces))


    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=8)
    except Exception:
        pass


    mesh.apply_translation(-mesh.bounding_box.centroid)
    extent = float(max(mesh.extents)) or 1.0
    mesh.apply_scale(_TARGET_SIZE_MM / extent)


    color = scene.get("material", {}).get("color", [0.85, 0.55, 0.45])
    rgba = [int(min(max(c, 0.0), 1.0) * 255) for c in color] + [255]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=rgba)
    return mesh


def _extract_json(text: str) -> Any:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    start, end = t.find("{"), t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        from json_repair import repair_json

        return json.loads(repair_json(t))


def generate_sdf_scene(
    prompt: str,
    *,
    prior_scene_json: str | None = None,
    image: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask the LLM for an SDF scene."""
    from textocad.services.llm_service import multimodal_generate

    user = prompt
    if prior_scene_json and prior_scene_json.strip():
        user = (
            "You are EDITING an existing organic SDF scene. Current scene JSON:\n\n"
            f"{prior_scene_json.strip()}\n\n"
            f"New request: {prompt}\n\n"
            "Return the COMPLETE updated scene JSON, keeping existing primitives and "
            "adding/adjusting to satisfy the new request."
        )
    if image:
        user = (
            f"{user}\n\nA REFERENCE IMAGE is attached — match its silhouette, "
            "proportions and pose as closely as the primitives allow."
        )

    raw = multimodal_generate(_SDF_SYS, user, image=image, temperature=0.4)
    try:
        data = _extract_json(raw)
    except Exception:
        data = {}
    return normalize_scene(data, prompt=prompt)


def build_organic_sdf(
    prompt: str,
    *,
    build_dir: str,
    prior_scene_json: str | None = None,
    image: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate an organic model from a prompt and export STL + GLB into build_dir."""
    import os

    scene = generate_sdf_scene(prompt, prior_scene_json=prior_scene_json, image=image)
    mesh = scene_to_mesh(scene)

    stl_path = os.path.join(build_dir, "result.stl")
    glb_path = os.path.join(build_dir, "result.glb")
    mesh.export(stl_path)
    mesh.export(glb_path)

    ext = mesh.extents
    return scene, {
        "stl_path": stl_path,
        "glb_path": glb_path,
        "bbox": [float(ext[0]), float(ext[1]), float(ext[2])],
        "volume": float(abs(mesh.volume)),
    }

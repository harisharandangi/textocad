"""Named, coloured parts -> the files every engine ships."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PALETTE = (
    "#4a7bd0",
    "#e07b39",
    "#3f9e6e",
    "#c9504b",
    "#8a63c7",
    "#d4a72c",
    "#3aa3b5",
    "#b5577f",
    "#6f7d8c",
    "#9c6b3f",
)
SINGLE_PART_COLOR = "#9aa1aa"
PARTS_DIR = "parts"

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_NAMED = {
    "red": "#c9504b", "green": "#3f9e6e", "blue": "#4a7bd0", "orange": "#e07b39",
    "yellow": "#e3c341", "purple": "#8a63c7", "teal": "#3aa3b5", "pink": "#e58fb0",
    "magenta": "#b5577f", "brown": "#9c6b3f", "black": "#2b2d31", "white": "#eeeeee",
    "gray": "#9aa1aa", "grey": "#9aa1aa", "silver": "#c0c4c8", "gold": "#d4a72c",
    "cyan": "#3aa3b5", "navy": "#2d3e70", "olive": "#7d8a3a", "beige": "#d8c8a8",
}


def safe_part_name(name: Any, used: set[str] | None = None, fallback: str = "part") -> str:
    """A filename- and glTF-node-safe slug that is unique within ``used``."""
    clean = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")
    clean = clean[:48] or fallback
    if clean[0].isdigit():
        clean = f"{fallback}_{clean}"
    if used is None:
        return clean
    candidate, n = clean, 2
    while candidate in used:
        candidate = f"{clean}_{n}"
        n += 1
    used.add(candidate)
    return candidate


def part_label(name: str) -> str:
    return re.sub(r"[_\s]+", " ", str(name or "")).strip().capitalize() or "Part"


def normalize_color(value: Any, index: int = 0, count: int = 1) -> str:
    """'#rrggbb' from a hex string, a colour name, or an (r, g, b[, a]) tuple in 0-1 or 0-255; otherwise the."""
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in _NAMED:
            return _NAMED[raw]
        m = _HEX_RE.match(raw)
        if m:
            return "#" + m.group(1).lower()
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            rgb = [float(v) for v in value[:3]]
            if all(0.0 <= v <= 1.0 for v in rgb):
                rgb = [v * 255.0 for v in rgb]
            return "#" + "".join(f"{max(0, min(255, int(round(v)))):02x}" for v in rgb)
        except (TypeError, ValueError):
            pass
    if count <= 1:
        return SINGLE_PART_COLOR
    return PALETTE[index % len(PALETTE)]


def _rgba(color: str) -> list[int]:
    h = normalize_color(color).lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255]


def _load_mesh(path: str | Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    return mesh


def _colored(mesh, color: str, name: str):
    import trimesh

    out = mesh.copy()
    out.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name=f"{name}_material",
            baseColorFactor=_rgba(color),
            metallicFactor=0.1,
            roughnessFactor=0.6,
        )
    )
    return out


def write_glb(items: list[tuple[str, str, Any]], path: str | Path) -> str:
    """items: (name, colour, trimesh). One scene node per part."""
    import trimesh

    scene = trimesh.Scene()
    for name, color, mesh in items:
        scene.add_geometry(_colored(mesh, color, name), node_name=name, geom_name=name)
    data = scene.export(file_type="glb")
    Path(path).write_bytes(data)
    return str(path)


def threemf_bytes(items: list[tuple[str, str, Any]], unit: str = "millimeter") -> bytes:
    """A 3MF package: one object per part, each with its colour as a base material."""
    materials = "".join(
        f'<base name="{escape(name)}" displaycolor="{normalize_color(color).upper()}FF"/>'
        for name, color, _ in items
    )
    objects: list[str] = []
    build: list[str] = []
    for idx, (name, _color, mesh) in enumerate(items):
        oid = idx + 2
        verts = "".join(
            f'<vertex x="{v[0]:.5f}" y="{v[1]:.5f}" z="{v[2]:.5f}"/>' for v in mesh.vertices
        )
        tris = "".join(
            f'<triangle v1="{int(f[0])}" v2="{int(f[1])}" v3="{int(f[2])}"/>' for f in mesh.faces
        )
        objects.append(
            f'<object id="{oid}" name="{escape(name)}" type="model" pid="1" pindex="{idx}">'
            f"<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh></object>"
        )
        build.append(f'<item objectid="{oid}"/>')
    model = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<model unit="{unit}" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        '<metadata name="Application">TextoCAD</metadata>'
        f'<resources><basematerials id="1">{materials}</basematerials>{"".join(objects)}</resources>'
        f'<build>{"".join(build)}</build></model>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
        'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model)
    return buf.getvalue()


def write_bundle(parts: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Any]:
    """parts: [{name, label?, color?, stl_path}] (STL in mm, Z-up)."""
    import trimesh

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str, Any]] = []
    facts: list[dict[str, Any]] = []
    for p in parts:
        mesh = _load_mesh(p["stl_path"])
        if mesh.is_empty or len(mesh.faces) == 0:
            raise ValueError(f"part '{p['name']}' has no geometry")
        name, color = p["name"], normalize_color(p.get("color"), len(items), len(parts))
        items.append((name, color, mesh))
        ext = mesh.extents
        facts.append(
            {
                "name": name,
                "label": p.get("label") or part_label(name),
                "color": color,
                "volume": round(float(abs(mesh.volume)), 3),
                "bbox": [round(float(v), 3) for v in ext],
                "triangles": int(len(mesh.faces)),
                "watertight": bool(mesh.is_watertight),
                "hasStep": bool(p.get("step_path")),
            }
        )
    combined = trimesh.util.concatenate([m for _, _, m in items])
    stl_path = out / "result.stl"
    combined.export(str(stl_path))
    glb_path = write_glb(items, out / "result.glb")
    threemf_path = out / "result.3mf"
    threemf_path.write_bytes(threemf_bytes(items))
    (out / "parts.json").write_text(json.dumps(facts, indent=1), encoding="utf-8")
    ext = combined.extents
    return {
        "stl_path": str(stl_path),
        "glb_path": glb_path,
        "threemf_path": str(threemf_path),
        "bbox": [float(ext[0]), float(ext[1]), float(ext[2])],
        "volume": float(sum(f["volume"] for f in facts)),
        "parts": facts,
    }


SELECTION_FORMATS = {"stl", "3mf", "glb", "obj", "step"}


def read_parts_manifest(build_dir: str | Path) -> list[dict[str, Any]]:
    try:
        data = json.loads((Path(build_dir) / PARTS_DIR / "parts.json").read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def export_selection(
    build_dir: str,
    names: list[str],
    fmt: str,
    layout: str,
    colors: dict[str, str] | None = None,
) -> str:
    """Write the chosen parts as one file (``layout='merged'``) or a zip with one file per part."""
    bdir = Path(build_dir)
    pdir = bdir / PARTS_DIR
    manifest = {p["name"]: p for p in read_parts_manifest(bdir)}
    chosen = [n for n in names if n in manifest]
    if not chosen:
        raise ValueError("none of the requested parts exist in this build")
    if fmt not in SELECTION_FORMATS:
        raise ValueError(f"unsupported format {fmt}")
    colors = colors or {}
    out_dir = bdir / "selections"
    out_dir.mkdir(exist_ok=True)
    tag = "_".join(chosen)[:60]
    ext = "step" if fmt == "step" else fmt

    def color_of(name: str) -> str:
        return normalize_color(colors.get(name) or manifest[name].get("color"))

    if fmt == "step":
        import cadquery as cq

        missing = [n for n in chosen if not (pdir / f"{n}.step").exists()]
        if missing:
            raise ValueError("STEP is not available for: " + ", ".join(missing))
        if layout == "separate":
            target = out_dir / f"{tag}_step.zip"
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
                for n in chosen:
                    zf.write(pdir / f"{n}.step", f"{n}.step")
            return str(target)
        assy = cq.Assembly(name="selection")
        for n in chosen:
            shape = cq.importers.importStep(str(pdir / f"{n}.step"))
            r, g, b, _ = _rgba(color_of(n))
            assy.add(shape, name=n, color=cq.Color(r / 255, g / 255, b / 255, 1))
        target = out_dir / f"{tag}.step"
        assy.export(str(target))
        return str(target)

    import trimesh

    items = [(n, color_of(n), _load_mesh(pdir / f"{n}.stl")) for n in chosen]

    def one_file(sub: list[tuple[str, str, Any]]) -> bytes:
        if fmt == "3mf":
            return threemf_bytes(sub)
        if fmt == "glb":
            buf = out_dir / "_tmp.glb"
            write_glb(sub, buf)
            return buf.read_bytes()
        mesh = trimesh.util.concatenate([m for _, _, m in sub])
        if fmt == "obj":
            return mesh.export(file_type="obj").encode("utf-8")
        return mesh.export(file_type="stl")

    if layout == "separate" and len(items) > 1:
        target = out_dir / f"{tag}_{fmt}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in items:
                zf.writestr(f"{item[0]}.{ext}", one_file([item]))
        return str(target)
    target = out_dir / f"{tag}.{ext}"
    target.write_bytes(one_file(items))
    return str(target)


def _surface_points(mesh, n: int = 2000):
    import numpy as np

    if mesh.is_empty or len(mesh.faces) == 0:
        return np.zeros((0, 3))
    pts = mesh.sample(n)
    return np.vstack([pts, mesh.vertices[: min(len(mesh.vertices), 1500)]])


def check_assembly(parts_dir: str) -> list[str]:
    """Plain-language problems with how the parts sit together: parts that float away from every other part, or."""
    import numpy as np

    pdir = Path(parts_dir)
    manifest = read_parts_manifest(pdir.parent) or [
        {"name": f.stem, "label": part_label(f.stem)} for f in sorted(pdir.glob("*.stl"))
    ]
    meshes = {}
    for p in manifest:
        f = pdir / f"{p['name']}.stl"
        if f.exists():
            meshes[p["name"]] = _load_mesh(f)
    names = list(meshes)
    if len(names) < 2:
        return []
    import trimesh

    whole = trimesh.util.concatenate([meshes[n] for n in names])
    size = max(float(max(whole.extents)), 1.0)
    tol = max(1.0, 0.015 * size)
    pts = {n: _surface_points(meshes[n]) for n in names}
    try:
        from scipy.spatial import cKDTree

        trees = {n: cKDTree(pts[n]) for n in names}

        def gap(a: str, b: str) -> float:
            d, _ = trees[b].query(pts[a], k=1)
            return float(d.min())
    except Exception:

        def gap(a: str, b: str) -> float:
            best = float("inf")
            for chunk in np.array_split(pts[a], max(1, len(pts[a]) // 400)):
                best = min(best, float(np.linalg.norm(chunk[:, None] - pts[b][None], axis=2).min()))
            return best

    touching = {n: set() for n in names}
    nearest = {n: (float("inf"), "") for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = gap(a, b)
            if g < nearest[a][0]:
                nearest[a] = (g, b)
            if g < nearest[b][0]:
                nearest[b] = (g, a)
            if g <= tol:
                touching[a].add(b)
                touching[b].add(a)

    groups: list[set[str]] = []
    seen: set[str] = set()
    for n in names:
        if n in seen:
            continue
        stack, group = [n], set()
        while stack:
            cur = stack.pop()
            if cur in group:
                continue
            group.add(cur)
            stack.extend(touching[cur] - group)
        seen |= group
        groups.append(group)
    if len(groups) <= 1:
        return []
    labels = {p["name"]: p.get("label") or part_label(p["name"]) for p in manifest}
    main = max(groups, key=lambda g: sum(abs(float(meshes[n].volume)) for n in g))
    issues = []
    for group in groups:
        if group is main:
            continue
        for n in sorted(group):
            g, other = nearest[n]
            issues.append(
                f"'{labels.get(n, n)}' floats {g:.1f} mm away from the rest of the model "
                f"(nearest: '{labels.get(other, other)}'); move it so it touches where it joins"
            )
    return issues[:8]


STAMP = "TextoCAD (https://textocad.com)"


def _stamp_glb(path: Path) -> None:
    import struct

    data = path.read_bytes()
    if data[:4] != b"glTF":
        return
    size, kind = struct.unpack_from("<I4s", data, 12)
    if kind != b"JSON":
        return
    doc = json.loads(data[20:20 + size])
    doc.setdefault("asset", {"version": "2.0"})["generator"] = STAMP
    chunk = json.dumps(doc, separators=(",", ":")).encode()
    chunk += b" " * (-len(chunk) % 4)
    rest = data[20 + size:]
    body = struct.pack("<I4s", len(chunk), b"JSON") + chunk + rest
    path.write_bytes(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)


def _stamp_stl(path: Path) -> None:
    data = path.read_bytes()
    if len(data) >= 84 and len(data) == 84 + 50 * int.from_bytes(data[80:84], "little"):
        header = f"Made with {STAMP}".encode().ljust(80, b" ")[:80]
        path.write_bytes(header + data[80:])
        return
    text = data.decode("ascii", errors="ignore")
    lines = text.splitlines()
    if not lines or not lines[0].lstrip().startswith("solid"):
        return
    lines[0] = "solid TextoCAD"
    for i in range(len(lines) - 1, 0, -1):
        if lines[i].lstrip().startswith("endsolid"):
            lines[i] = "endsolid TextoCAD"
            break
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _stamp_step(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    note = f"/* Generated by {STAMP} */"
    if note in text or "HEADER;" not in text:
        return
    path.write_text(text.replace("HEADER;", f"HEADER;\n{note}", 1), encoding="utf-8", errors="surrogateescape")


def stamp_exports(paths) -> None:
    """Mark downloadable files as made by TextoCAD (STEP comment, STL header, GLB generator)."""
    stampers = {".glb": _stamp_glb, ".stl": _stamp_stl, ".step": _stamp_step, ".stp": _stamp_step}
    for p in paths:
        if not p:
            continue
        path = Path(p)
        fn = stampers.get(path.suffix.lower())
        if fn and path.is_file():
            try:
                fn(path)
            except (OSError, ValueError):
                pass

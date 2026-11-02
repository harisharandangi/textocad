"""Phase 1 — Verification gate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from textocad.services.requirement_spec import RequirementSpec


MIN_LINEAR_TOL = 0.5


@dataclass
class GeometrySample:
    """Everything the gate needs to read back. Pipeline-agnostic."""

    solid: Any = None
    mesh: Any = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckFailure:
    check: str
    target: str | None
    expected: Any
    got: Any
    message: str

    def as_delta(self) -> dict[str, Any]:
        return {"check": self.check, "target": self.target, "expected": self.expected, "got": self.got}


@dataclass
class VerificationReport:
    passed: bool
    failures: list[CheckFailure]
    understood_spec: str

    def summary(self) -> str:
        if self.passed:
            return f"PASS — {self.understood_spec}"
        lines = [f"REJECT — {len(self.failures)} hard failure(s); {self.understood_spec}"]
        for f in self.failures:
            lines.append(f"  • [{f.check}] {f.message}  (expected={f.expected!r}, got={f.got!r})")
        return "\n".join(lines)


def _mesh_bounds(mesh: Any) -> tuple[float, float, float]:
    lo, hi = mesh.bounds
    return (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))


def _radius_profile(points_xy, center, nbins: int = 1440):
    """Max radius per angular bin around `center`. Returns list[nbins] (0 where empty)."""
    prof = [0.0] * nbins
    cx, cy = center
    for x, y in points_xy:
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)
        ang = math.atan2(dy, dx)
        b = int((ang + math.pi) / (2 * math.pi) * nbins) % nbins
        if r > prof[b]:
            prof[b] = r
    return prof


def _dominant_period_count(prof, lo: int = 3, hi: int | None = None) -> int:
    """Number of teeth = dominant angular frequency of the radius profile, via FFT."""
    import numpy as np

    arr = np.asarray(prof, dtype=float)
    filled = arr[arr > 0]
    if filled.size == 0 or float(filled.max() - filled.min()) < 1e-6:
        return 0
    med = float(np.median(filled))
    arr = np.where(arr > 0, arr, med) - arr.mean()
    spec = np.abs(np.fft.rfft(arr))
    spec[0] = 0.0
    hi = hi or (len(spec) - 1)
    band = spec[lo:hi]
    if band.size == 0:
        return 0
    peak = float(band.max())

    for k in range(lo, hi):
        if spec[k] >= 0.40 * peak:
            return k
    return int(band.argmax()) + lo


def count_teeth(mesh: Any) -> int:
    """Measure tooth count from the mesh, robustly for real involute gears."""
    import numpy as np

    v = mesh.vertices


    z = v[:, 2]
    zc = float((z.min() + z.max()) / 2.0)
    slab = max(float(z.max() - z.min()) * 0.12, 0.3)
    band = np.abs(z - zc) <= slab
    if int(band.sum()) >= 50:
        v = v[band]
    cx = float(v[:, 0].mean())
    cy = float(v[:, 1].mean())
    r = np.hypot(v[:, 0] - cx, v[:, 1] - cy)
    rmax = float(r.max())
    if rmax <= 0:
        return 0
    mask = r >= 0.55 * rmax
    if int(mask.sum()) < 12:
        return 0
    ang = np.arctan2(v[mask, 1] - cy, v[mask, 0] - cx)
    rr = r[mask]
    nb = 360
    b = ((ang + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    prof = np.full(nb, np.nan)
    for bi, rv in zip(b, rr):
        if np.isnan(prof[bi]) or rv > prof[bi]:
            prof[bi] = rv
    good = ~np.isnan(prof)
    if int(good.sum()) < 8:
        return 0
    idx = np.arange(nb)
    prof = np.interp(idx, idx[good], prof[good], period=nb)
    arr = prof - prof.mean()
    if float(arr.max() - arr.min()) < 1e-6:
        return 0
    sp = np.abs(np.fft.rfft(arr))
    sp[0] = 0
    band = sp[5:nb // 2]
    if band.size == 0:
        return 0
    return int(np.argmax(band)) + 5


def _slice_profile(mesh: Any, z: float, center, nbins: int = 1440):
    """Radius profile at a given Z plane (vertices within a thin slab)."""
    verts = mesh.vertices
    lo, hi = mesh.bounds
    slab = max((hi[2] - lo[2]) * 0.06, 0.5)
    pts = [(float(v[0]), float(v[1])) for v in verts if abs(float(v[2]) - z) <= slab]
    if len(pts) < 8:
        pts = [(float(v[0]), float(v[1])) for v in verts]
    return _radius_profile(pts, center, nbins)


def measure_twist_deg(mesh: Any) -> float:
    """Angular twist (deg) of the tooth pattern from bottom to top slice, via circular cross-correlation."""
    verts = mesh.vertices
    lo, hi = mesh.bounds
    cx = float(verts[:, 0].mean())
    cy = float(verts[:, 1].mean())
    nbins = 720
    p0 = _slice_profile(mesh, lo[2] + (hi[2] - lo[2]) * 0.15, (cx, cy), nbins)
    p1 = _slice_profile(mesh, lo[2] + (hi[2] - lo[2]) * 0.85, (cx, cy), nbins)
    import numpy as np

    a0 = np.asarray(p0, dtype=float)
    a1 = np.asarray(p1, dtype=float)
    scores = np.array([float(np.dot(a0, np.roll(a1, lag))) for lag in range(nbins)])
    peak = float(scores.max())


    signed = [(lag if lag <= nbins // 2 else lag - nbins) for lag in range(nbins)]
    candidates = [signed[lag] for lag in range(nbins) if scores[lag] >= 0.98 * peak]
    lag = min(candidates, key=abs) if candidates else 0
    return lag * (360.0 / nbins)


def _curved_face_radii(solid: Any) -> list[float]:
    """Radii of cylindrical/toroidal faces in the solid (the signature of an applied fillet/round)."""
    radii: list[float] = []
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Torus

        faces = solid.faces() if hasattr(solid, "faces") else []
        face_objs = faces.objects if hasattr(faces, "objects") else faces
        for f in face_objs:
            try:
                surf = BRepAdaptor_Surface(f.wrapped)
                t = surf.GetType()
                if t == GeomAbs_Cylinder:
                    radii.append(float(surf.Cylinder().Radius()))
                elif t == GeomAbs_Torus:
                    radii.append(float(surf.Torus().MinorRadius()))
            except Exception:
                continue
    except Exception:

        try:
            face_objs = solid.faces().objects
            for f in face_objs:
                if f.geomType() in {"CYLINDER", "TORUS"}:
                    radii.append(-1.0)
        except Exception:
            pass
    return radii


def _solid_is_valid(solid: Any) -> bool | None:
    """Authoritative closed/valid-solid check via OCCT."""
    if solid is None:
        return None
    try:
        val = solid.val() if hasattr(solid, "val") else solid
        return bool(val.isValid())
    except Exception:
        return None


def count_holes(solid: Any) -> int:
    """Measure hole count FROM the B-rep (don't trust the input)."""
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
    except Exception:
        return -1

    faces = solid.faces() if hasattr(solid, "faces") else None
    face_objs = faces.objects if faces is not None and hasattr(faces, "objects") else None
    if not face_objs:
        return -1

    groups: set[tuple] = set()
    for f in face_objs:
        try:
            surf = BRepAdaptor_Surface(f.wrapped)
            if surf.GetType() != GeomAbs_Cylinder:
                continue
            axis = surf.Cylinder().Axis()
            loc = axis.Location()
            dirn = axis.Direction()
            dx, dy, dz = dirn.X(), dirn.Y(), dirn.Z()
            if (dx, dy, dz) < (0.0, 0.0, 0.0):
                dx, dy, dz = -dx, -dy, -dz

            groups.add((round(dx, 2), round(dy, 2), round(dz, 2),
                        round(loc.X(), 1), round(loc.Y(), 1)))
        except Exception:
            continue
    return len(groups)


def verify(sample: GeometrySample, spec: RequirementSpec) -> VerificationReport:
    failures: list[CheckFailure] = []
    mesh = sample.mesh


    if spec.topology.watertight:
        solid_validity = _solid_is_valid(sample.solid)
        if solid_validity is False:
            failures.append(CheckFailure(
                "watertight", None, True, False,
                "result is NOT a valid closed solid (B-rep self-intersecting or open; "
                "e.g. a fillet/feature corrupted the body)",
            ))
        elif solid_validity is None and mesh is not None and not bool(mesh.is_watertight):
            n_open = -1
            try:
                import numpy as np

                _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
                n_open = int((counts < 2).sum())
            except Exception:
                pass
            failures.append(CheckFailure(
                "watertight", None, True, False,
                "result is NOT watertight (mesh has open boundary edges; not a closed solid)"
                + (f"; ~{n_open} open edges" if n_open >= 0 else ""),
            ))
    if spec.topology.solids is not None and mesh is not None:
        try:
            got_solids = int(mesh.body_count)
        except Exception:
            got_solids = 1
        if got_solids != spec.topology.solids:
            failures.append(CheckFailure(
                "solids", None, spec.topology.solids, got_solids,
                f"expected {spec.topology.solids} solid body, found {got_solids}",
            ))


    if mesh is not None:
        ext = _mesh_bounds(mesh)
        axis_idx = {"x": 0, "y": 1, "z": 2}
        if spec.bbox is not None:
            for axis in ("x", "y", "z"):
                rng = getattr(spec.bbox, axis)
                if rng is None:
                    continue
                got = ext[axis_idx[axis]]
                if got < rng[0] - MIN_LINEAR_TOL or got > rng[1] + MIN_LINEAR_TOL:
                    failures.append(CheckFailure(
                        "bbox", axis, list(rng), round(got, 3),
                        f"{axis}-extent {got:.2f}mm is outside allowed [{rng[0]},{rng[1]}]mm",
                    ))
        for d in spec.dimensions:
            if d.measure == "bbox" and d.axis:
                got = ext[axis_idx[d.axis]]
                tol = max(d.tol, MIN_LINEAR_TOL)
                if abs(got - d.value) > tol:
                    failures.append(CheckFailure(
                        "bbox", d.name, d.value, round(got, 3),
                        f"{d.name} ({d.axis}) is {got:.2f}mm, expected {d.value}±{tol}mm",
                    ))


    for feat in spec.features:
        tol = feat.tol if feat.tol is not None else 0.5

        if feat.kind == "gear_teeth" and feat.count is not None and mesh is not None:
            got = count_teeth(mesh)
            if got != feat.count:
                failures.append(CheckFailure(
                    "gear_teeth", feat.of, feat.count, got,
                    f"measured {got} teeth from geometry, but spec requires {feat.count}",
                ))

        elif feat.kind == "spiral" and feat.angle_deg is not None and mesh is not None:
            twist = measure_twist_deg(mesh)

            lo, hi = mesh.bounds
            face_w = float(hi[2] - lo[2]) or 1.0
            verts = mesh.vertices
            pitch_r = float(((verts[:, 0] ** 2 + verts[:, 1] ** 2) ** 0.5).mean()) or 1.0
            arc = math.radians(abs(twist)) * pitch_r
            measured_angle = math.degrees(math.atan2(arc, face_w))
            if abs(measured_angle - feat.angle_deg) > max(tol, 5.0):
                failures.append(CheckFailure(
                    "spiral_angle", feat.of, feat.angle_deg, round(measured_angle, 2),
                    f"stated spiral angle {feat.angle_deg}° but geometry twists ~{twist:.1f}° "
                    f"(≈{measured_angle:.1f}° spiral) — teeth do not match the stated spiral",
                ))

        elif feat.kind == "fillet" and feat.present and sample.solid is not None:
            radii = _curved_face_radii(sample.solid)
            if not radii:
                failures.append(CheckFailure(
                    "fillet", feat.of, f"r={feat.r}", "none",
                    f"fillet r={feat.r}mm requested on '{feat.of}' but NOT applied — "
                    f"solid is all-planar (sharp edges remain, no rounded faces)",
                ))
            elif feat.r is not None:
                known = [r for r in radii if r > 0]
                if known and not any(abs(r - feat.r) <= max(tol, 0.25) for r in known):
                    failures.append(CheckFailure(
                        "fillet", feat.of, f"r={feat.r}", [round(r, 3) for r in known],
                        f"fillet present but no rounded face matches r={feat.r}mm "
                        f"(found radii {[round(r, 2) for r in known]}mm)",
                    ))

        elif feat.kind == "hole" and feat.count is not None:


            got = count_holes(sample.solid) if sample.solid is not None else -1
            source = "geometry"
            if got < 0:
                got = int(sample.provenance.get("hole_count", -1))
                source = "build"
            if got >= 0 and got != feat.count:
                failures.append(CheckFailure(
                    "hole_count", feat.of, feat.count, got,
                    f"measured {got} holes from {source}, but spec requires {feat.count}",
                ))

    return VerificationReport(
        passed=len(failures) == 0,
        failures=failures,
        understood_spec=spec.describe(),
    )

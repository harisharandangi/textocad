"""Codegen pipeline — prompt -> {param_schema + build(params) script} -> sandbox."""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any

from textocad.core.config import settings
from textocad.services.codegen_sandbox import ExecResult, execute_script, validate_script
from textocad.services.provider_errors import ProviderError


def _log(message: str) -> None:
    """Stage logging for the generation pipeline."""
    print(f"[textocad.codegen] {message}", flush=True)


_ORGANIC_HINTS = (
    "organic", "blob", "vase", "sculpt", "smooth", "freeform", "free-form",
    "lattice", "terrain", "character", "figurine", "statue", "animal", "creature",
    "bottle", "vessel", "ergonomic grip", "flowing", "knurl", "voronoi",
)
_PRECISION_HINTS = (
    "gear", "teeth", "bracket", "hole", "bore", "thread", "flange", "shaft",
    "bearing", "screw", "bolt", "plate", "fillet", "chamfer", "counterbore",
    "mount", "mechanical", "tolerance", "mm", "spur", "helical", "rib", "boss",
)


_CAD_FILE_RE = re.compile(
    r"\b(step|stp|iges|cnc|machin\w*|milling|lathe|injection[- ]mou?ld\w*|sheet[- ]metal|"
    r"b-?rep|solidworks|fusion ?360|tolerances?)\b|step\s*格式|装配模型",
    re.IGNORECASE,
)


_ORGANIC_SUBJECT_RE = re.compile(
    r"\b(animal|dog|puppy|cat|kitten|wolf|fox|bear|horse|cow|pig|sheep|lion|tiger|"
    r"rabbit|bunny|bird|owl|duck|eagle|fish|shark|whale|dolphin|dragon|dinosaur|"
    r"t-?rex|unicorn|monster|creature|character|figurine|figure|statue|sculpture|"
    r"bust|face|head of|person|human|man|woman|girl|boy|baby|mascot|toy animal|"
    r"teddy|plush|elephant|giraffe|monkey|frog|turtle|snake|octopus)\b",
    re.IGNORECASE,
)


def wants_cad_file(prompt: str) -> bool:
    return bool(_CAD_FILE_RE.search(prompt or ""))


def route_mode(prompt: str) -> str:
    """Auto-route when the caller chose no mode."""
    low = prompt.lower()
    o = sum(1 for h in _ORGANIC_HINTS if h in low)
    p = sum(1 for h in _PRECISION_HINTS if h in low)
    if o > p or (_ORGANIC_SUBJECT_RE.search(low) and p == 0):
        return "organic"
    if wants_cad_file(low):
        return "precision"
    return "printing"


def refine_printing_mode(prompt: str) -> tuple[str, str | None]:
    """3D Printing is the default mode, so it also receives requests another engine serves far better."""
    low = (prompt or "").lower()
    if wants_cad_file(low):
        return "precision", "You asked for a CAD file (STEP), so this was built with Precision."
    p = sum(1 for h in _PRECISION_HINTS if h in low)
    if _ORGANIC_SUBJECT_RE.search(low) and p == 0:
        return "organic", "Built with Organic for a smooth, natural shape. It still downloads as a printable STL."
    return "printing", None


@dataclass
class CodegenPackage:
    mode: str
    title: str
    param_schema: list[dict[str, Any]]
    script: str
    prompt: str = ""
    notes: list[str] = field(default_factory=list)


    escalated_to: str = ""

    def default_params(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in self.param_schema:
            key = p.get("key")
            if key:
                out[key] = p.get("value", p.get("defaultValue", 0))
        return out


_PRECISION_SYS = """You are a CAD engineer. Output a JSON object describing a parametric mechanical part.

Return ONLY JSON of this exact shape:
{
  "design_notes": "1-3 sentences: which part this is and the key dimensions you chose (note which were GIVEN by the user vs ASSUMED from standard engineering defaults)",
  "title": "short name",
  "param_schema": [
    {"key":"snake_case","label":"Label","type":"number","value":<num>,"min":<num>,"max":<num>,"step":<num>,"unit":"mm","group":"Dimensions"}
  ],
  "script": "<python source defining build(params)>"
}

ENGINEERING APPROACH — fill in "design_notes" FIRST, reasoning before you write code:
1. Identify exactly which part is requested (gear, bolt, screw, nut, washer, cam,
   L-bracket, V-bracket/V-block, flange, shaft, pulley, dumbbell, impeller, cylinder,
   etc.).
2. Enumerate EVERY dimension that part needs to be fully defined.
3. Use any dimensions the user gave. For every dimension they did NOT specify, choose a
   realistic, manufacturable STANDARD engineering value (see defaults below). NEVER leave
   a part under-specified — a vague prompt like "create a gear" must still yield a
   complete, correct, realistic part.
4. Put ALL of those dimensions in param_schema with sensible min/max/step and correct
   units, grouped logically, so the user can fine-tune them with sliders.
5. build(params) must read every dimension from params['key'].

SENSIBLE DEFAULTS when unspecified (the user's explicit values always override these):
- Spur gear: module 2 mm, 20 teeth, 20° pressure angle, face width 10 mm, bore 6 mm.
- Bolt/screw: ISO metric M8, pitch 1.25, shank length 30 mm, hex head (13 mm across flats, 5.3 mm tall).
- Nut: ISO metric M8, 13 mm across flats, 6.5 mm thick. Washer: ID 8.4, OD 16, thick 1.6 mm.
- L-bracket: legs 50 x 50 mm, width 40 mm, thickness 4 mm, two 5 mm holes per leg.
- V-block: 50 mm body, 90° V-groove, 30 mm thick. Cam: base circle 30 mm, lift 10 mm, width 12 mm, bore 8 mm.
- Cylinder: dia 20 mm, height 40 mm. Pulley: OD 50 mm, bore 8 mm, width 15 mm.
- Dumbbell: bar dia 25 mm x 150 mm long, two end weights dia 60 mm. Impeller: hub 20 mm, 6 blades, OD 80 mm, height 25 mm, bore 8 mm.
Pick proportionate values for anything not listed; keep everything to real, printable mm scale.

The "script" MUST:
- define exactly one top-level function: def build(params):
- import only: cadquery (as cq), cq_gears, math
- read dimensions from params via params['key'] (keys MUST match param_schema)
- return a cadquery Workplane/Solid (a single valid solid) for a one-piece part, OR a
  dict of NAMED PARTS for anything made of several separate pieces:
      return {"base": base_wp, "lid": (lid_wp, "#e07b39"), "hinge_pin": pin_wp}
  Each value is a Workplane (or (Workplane, "#rrggbb") to give the part a colour).
  Every part is positioned where it belongs in the assembled object. Use snake_case
  part names. The user can then show, hide, recolour and download parts one by one.
- NOT import os/sys/subprocess/socket, NOT touch files
- NEVER use any of these builtins (they are hard-banned and will reject the script):
  open, eval, exec, compile, __import__, getattr, setattr, hasattr, delattr,
  globals, locals, vars, dir, input. Call the documented APIs directly instead of
  defensive hasattr()/getattr() checks (e.g. just build the trimesh, do not probe it).
- NOT export anything (the host exports). No top-level code except the def and imports.

For gears use cq_gears (true involute). The installed cq_gears provides ONLY the
SpurGear class — there is NO HelicalGear, BevelGear, or RingGear class, importing
them raises ImportError. For a helical gear, use SpurGear with a non-zero
helix_angle (radians). NEVER import anything other than SpurGear from cq_gears.
  from cq_gears import SpurGear
  g = SpurGear(module=params['module'], teeth_number=int(params['teeth']), width=params['width'], bore_d=params['bore'], helix_angle=params.get('helix_angle', 0.0))
  return cq.Workplane('XY').gear(g)

WORKED EXAMPLES — copy these CadQuery API idioms (all verified to build):

# L-bracket: L-profile extruded along Y; inner-corner fillet via NearestToPointSelector; hole through a leg.
def build(params):
    import cadquery as cq
    L = params['leg_length']; W = params['width']; T = params['thickness']
    pts = [(0, 0), (L, 0), (L, T), (T, T), (T, L), (0, L)]
    b = cq.Workplane('XZ').polyline(pts).close().extrude(W)
    b = b.edges('|Y').edges(cq.NearestToPointSelector((T, W / 2.0, T))).fillet(params['fillet_radius'])
    b = b.faces('>Z').workplane(centerOption='CenterOfBoundBox').pushPoints([(L * 0.6, 0)]).hole(params['hole_diameter'])
    return b

# Flange: disc + center bore + a bolt circle of N holes via polarArray.
def build(params):
    import cadquery as cq
    f = cq.Workplane('XY').circle(params['outer_d'] / 2.0).extrude(params['thickness'])
    f = f.faces('>Z').workplane().hole(params['bore_d'])
    f = f.faces('>Z').workplane().polarArray(params['bolt_circle_d'] / 2.0, 0, 360, int(params['bolt_count'])).hole(params['bolt_d'])
    return f

Idioms to reuse: build a 2D profile with polyline(...).close() then .extrude(); select edges with
'|X'/'|Y'/'|Z' (parallel to axis) or NearestToPointSelector((x,y,z)); put a hole on a face via
.faces('>Z').workplane().hole(d); pattern holes with .polarArray(radius, start_deg, 360, count).hole(d).

ROBUST FILLETS & CHAMFERS — these are the #1 cause of build failure. The OCCT kernel
raises "StdFail_NotDone: BRep_API: command not done" when a fillet/chamfer radius is too
large for an edge, or it self-intersects. Follow ALL of these:
- Build the COMPLETE valid solid FIRST (walls, cuts, holes). Add fillets/chamfers LAST.
- Keep every radius well under HALF the thinnest adjacent wall/feature (e.g. for a 2 mm
  wall use <= 0.8 mm; never fillet an edge with a radius near or above the wall thickness).
- Apply fillets to SPECIFIC selected edges (e.g. .edges('>Z'), NearestToPointSelector),
  NOT a blanket .edges().fillet() over the whole body.
- Make EVERY fillet/chamfer FAIL-SAFE: wrap it in try/except so a fillet the kernel
  rejects is SKIPPED and you keep the valid (un-filleted) solid instead of failing the
  whole model:
      try:
          result = result.edges('>Z').fillet(min(r, wall_thickness * 0.4))
      except Exception:
          pass   # kernel could not apply this fillet — keep the valid solid
- Skip fillets entirely on tiny internal corners of thin-walled compartments; they almost
  always fail.
- try/except is for FILLETS/CHAMFERS ONLY. NEVER wrap primary features — holes, cutouts,
  slots, bores, pockets, teeth — in try/except. If a hole fails, the model must FAIL so it
  gets repaired, not silently ship without the hole.

PARAMS MUST MATCH THE GEOMETRY. Every entry in param_schema MUST be read by the script and
visibly drive the geometry. If param_schema has hole_dia and a hole count, the built solid
MUST contain those holes; a fillet_radius param means the fillet must be attempted. Never
emit a param for a feature the script does not build — either build the feature or drop
the param. A model whose sliders promise holes/fillets that do not exist is WRONG.

Make geometry that MATCHES the prompt exactly (a comb is a spine + teeth array, NOT a gear).
A one-piece part is a single connected solid; separate physical pieces are separate
named parts in the returned dict - never fuse pieces that are really separate."""

_ORGANIC_SYS = """You are a 3D artist. Output a JSON object describing an organic/visual mesh.

Return ONLY JSON of this exact shape:
{
  "title": "short name",
  "param_schema": [ {"key":"snake_case","label":"Label","type":"number","value":<num>,"min":<num>,"max":<num>,"step":<num>,"unit":"mm","group":"Shape"} ],
  "script": "<python source defining build(params)>"
}

The "script" MUST:
- define exactly one top-level function: def build(params):
- import only: trimesh, numpy (as np), math   (PREFER trimesh; do not use manifold3d)
- read params via params['key'] (keys MUST match param_schema)
- return a single non-empty trimesh.Trimesh with volume > 0
- NOT import os/sys/subprocess/socket, NOT touch files
- NEVER use any of these builtins (they are hard-banned and will reject the script):
  open, eval, exec, compile, __import__, getattr, setattr, hasattr, delattr,
  globals, locals, vars, dir, input. Call the documented APIs directly instead of
  defensive hasattr()/getattr() checks (e.g. just build the trimesh, do not probe it).
- NOT export anything. No top-level code except the def and imports.

USE ONLY THESE VERIFIED trimesh APIs (do not invent kwargs):
- Solid of revolution from a 2D profile (BEST for vases/bottles/bowls):
    # profile is an (N,2) array of [radius, height] points, radius>=0, going bottom->top.
    # Close the profile to the axis at both ends (start and end radius = 0) for a solid.
    import numpy as np, trimesh
    prof = np.array([[0,0],[r1,0],[r2,h2],[r3,h3],[0,H]], dtype=float)
    mesh = trimesh.creation.revolve(linestring=prof, sections=96)
- Primitives you may union with boolean: trimesh.creation.icosphere(subdivisions=3, radius=R),
  trimesh.creation.cylinder(radius=R, height=H, sections=96),
  trimesh.creation.capsule(radius=R, height=H, count=[32,32]).
- Boolean union of a list of meshes (for bumpy/lumpy organic forms):
    blob = trimesh.boolean.union([m1, m2, m3])   # returns a Trimesh
- Smooth displacement: build an icosphere, then perturb mesh.vertices with a smooth
  function of their angle/height, then rebuild trimesh.Trimesh(vertices=v, faces=f).

Rules: prefer trimesh.creation.revolve for vessels. Ensure the profile starts and ends
on the axis (radius 0) so the solid is closed and has positive volume. Use sections>=64
for smoothness. Return the Trimesh directly."""


_PRINTING_SYS = """You are a design engineer who models parts for 3D printing in OpenSCAD.

Reply with ONE fenced OpenSCAD code block and nothing else:

```openscad
// title: <short name>
/* [Dimensions] */
width = 60;        // [20:1:200] Width (mm)
wall = 2;          // [0.8:0.2:6] Wall thickness (mm)
/* [Hidden] */
$fn = 64;
include <BOSL2/std.scad>   // optional

// @part base #4a7bd0 Base
module part_base() {
    ...
}
// @part lid #e07b39 Lid
module part_lid() {
    ...
}
```

RULES
- Line 1 is `// title: ...`.
- Every adjustable dimension is a top-level number with a Customizer comment
  `// [min:step:max] Label (mm)`, grouped under `/* [Group] */` headers, BEFORE any
  module. Values the user should not touch go under `/* [Hidden] */`. Every
  parameter MUST be used by the geometry.
- Geometry lives ONLY in modules named `part_<snake_name>()` with NO arguments. Each
  module is one physical printed piece, positioned where it sits in the assembled
  object. A one-piece object has a single module (e.g. `part_body()`).
- Directly above each part module write `// @part <snake_name> #rrggbb <Label>` to
  give it a colour and a display name. Give separate pieces distinct colours.
- Put each part where it belongs in the finished object: either translate/rotate
  INSIDE its part module, or call it at top level with a transform
  (`translate([0,0,40]) part_head();`) - both are kept.
- ASSEMBLED objects must be ONE connected object: every part touches or overlaps
  the part it attaches to by 0.5-1 mm (a head sits ON the neck, legs meet the
  body, wheels sit on their axles). Nothing floats. Only lay parts apart when the
  user asks for separate / exploded parts.
- Units are millimetres, Z is up, and each printed piece should sit on the bed
  (z >= 0) in its assembled position where that makes sense.
- Only library allowed: `include <BOSL2/std.scad>` (also BOSL2/gears.scad,
  BOSL2/threading.scad, BOSL2/screws.scad). No import(), no surface(), no other files.
- Keep $fn <= 128. Avoid minkowski()/hull() on high-resolution shapes (slow).

PRINTABILITY (this is a 3D-printing engine - design for FDM unless told otherwise)
- Walls >= 1.2 mm, small pins >= 3 mm diameter, text/emboss depth >= 0.6 mm.
- Prefer chamfers/fillets on bed edges and 45-degree overhangs; avoid unsupported
  horizontal overhangs wider than ~5 mm unless the part is meant to be printed flipped.
- Parts that fit together get clearance: 0.2-0.3 mm per side for sliding fits,
  0.1 mm for press fits. Holes for screws: M3 = 3.4 mm, M4 = 4.5 mm, M5 = 5.5 mm.
- Every part must be a closed solid with positive volume.

ENGINEERING APPROACH: decide which object is requested, list every dimension it needs,
use the user's numbers, and choose realistic printable defaults for the rest. A vague
prompt must still produce a complete, recognisable, useful object.

USEFUL BOSL2 IDIOMS (verified):
  cuboid([x,y,z], rounding=2, edges="Z", anchor=BOTTOM);
  cyl(h=10, d=20, chamfer=1, anchor=BOTTOM);
  tube(h=10, od=20, id=16, anchor=BOTTOM);
  spur_gear(mod=2, teeth=20, thickness=8, shaft_diam=5);
  threaded_rod(d=10, l=30, pitch=1.5);
Plain OpenSCAD (cube/cylinder/sphere/difference/union/linear_extrude/rotate_extrude/
polygon/offset/hull) is always fine and is preferred when it is just as simple.

RECOGNISABLE SHAPES: model the real proportions of the object. Smooth rounded
forms come from hull() of spheres/cylinders, scale()d spheres and rotate_extrude
profiles - not from stacked boxes. A REFERENCE IMAGE is a guide to the object's
shape and layout; reproduce its main features, not its background or paper."""


_CQ_GEARS_BAD_CLASSES = ("HelicalGear", "HerringboneGear")


def _fix_known_codegen_pitfalls(script: str, mode: str) -> str:
    """Deterministically rewrite known-bad generated code so a recurring LLM mistake self-heals instead of."""
    if mode != "precision" or "cq_gears" not in script:
        return script
    fixed = script
    for bad in _CQ_GEARS_BAD_CLASSES:
        if bad in fixed:
            fixed = re.sub(rf"\b{bad}\b", "SpurGear", fixed)
    return fixed


_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def _sanitize_json_escapes(s: str) -> str:
    """Escape any backslash that is NOT part of a valid JSON escape sequence."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in _VALID_JSON_ESCAPES:
                out.append(c)
                out.append(nxt)
                i += 2
                continue

            out.append("\\\\")
            i += 1
            continue
        if c == "\\" and i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_json(text: str) -> dict[str, Any]:
    """Parse the model's JSON object."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()


    unsliced = t

    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        t = t[start : end + 1]

    parsed: Any = None
    for attempt in (
        lambda: json.loads(t),

        lambda: json.loads(_sanitize_json_escapes(t)),


        lambda: json.loads(__import__("json_repair").repair_json(t)),
    ):
        try:
            parsed = attempt()
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed


    for layer in (parsed, unsliced):
        if not isinstance(layer, str):
            continue
        text_layer = layer
        for attempt in (
            lambda: json.loads(text_layer),
            lambda: json.loads(_sanitize_json_escapes(text_layer)),
            lambda: json.loads(__import__("json_repair").repair_json(text_layer)),
        ):
            try:
                inner = attempt()
            except Exception:
                continue
            if isinstance(inner, dict):
                return inner

            if isinstance(inner, str):
                try:
                    deeper = json.loads(inner)
                except Exception:
                    deeper = None
                if isinstance(deeper, dict):
                    return deeper

    if parsed is not None and not isinstance(parsed, dict):
        raise CodegenFormatError(
            "Model returned JSON that is not an object "
            f"(got {type(parsed).__name__}). Return {{title, param_schema, script}}.",
            raw=text or "",
        )
    raise CodegenFormatError(
        "Model did not return parseable JSON. Return ONLY a JSON object with "
        "keys: title, param_schema, script.",
        raw=text or "",
    )


class CodegenFormatError(ValueError):
    """The model answered in the wrong SHAPE, not with bad geometry."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw or ""


_FENCE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.S)


_BUILD_DEF = re.compile(r"^\s*def\s+build\s*\(", re.M)

_CODE_START = re.compile(r"^\s*(?:import\s+\w|from\s+\w|def\s+\w|class\s+\w)", re.M)


def _salvage_script(text: str) -> str | None:
    """Pull a build(params) script out of a malformed response, locally."""
    body = text or ""
    if not body.strip():
        return None


    for block in _FENCE.findall(body):
        if _BUILD_DEF.search(block):
            return block.strip()


    if not _BUILD_DEF.search(body):
        return None

    start = _CODE_START.search(body)
    candidate = body[start.start() :] if start else body
    return candidate.strip() or None


def _extract_package_json(raw: str, *, prompt: str, mode: str) -> dict[str, Any]:
    """_extract_json, with a local salvage pass behind it."""
    try:
        return _extract_json(raw)
    except ValueError as exc:


        script = _salvage_script(raw)
        if script is None:
            if isinstance(exc, CodegenFormatError):
                raise
            raise CodegenFormatError(str(exc), raw=raw) from exc
        _log(f"SALVAGED a build() from a malformed envelope ({len(script)} chars)")


        return {"title": (prompt or "").strip()[:40], "param_schema": [], "script": script}


_INTENT_SYS = """You decide whether a new CAD instruction MODIFIES the previous design or
asks for a NEW, unrelated design. Return ONLY JSON: {"intent":"edit"} or {"intent":"new"}.
- "edit": tweaks/adds/removes/resizes the SAME object (e.g. previous "a gear", new
  "make it bigger" / "add a bore" / "now round the edges" / "add a hub").
- "new": a different, unrelated object (e.g. previous "a computer mouse", new "a pen with
  separate parts"; previous "a bracket", new "a phone stand").
Choose "edit" only when the instruction clearly refers to / builds on the previous object;
otherwise choose "new"."""


def classify_design_intent(previous_prompt: str, new_prompt: str) -> str:
    """Return "edit" or "new"."""
    if not (previous_prompt or "").strip():
        return "new"
    try:
        from textocad.core.config import settings
        from textocad.services.llm_models import helper_tier
        from textocad.services.llm_service import GenerationConfig, _get_client

        contents = (
            f"Previous design request: {previous_prompt}\n"
            f"New instruction: {new_prompt}\n"
            "Is the new instruction an edit of the previous design, or a new design?"
        )


        with helper_tier("classify"):
            client = _get_client()
            response = client.models.generate_content(
                contents=contents,
                config=GenerationConfig(
                    system_instruction=_INTENT_SYS,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
        data = _extract_json(response.text or "")
        intent = str((data or {}).get("intent") or "").strip().lower()
        return "new" if intent == "new" else "edit"
    except Exception:
        return "edit"


_SEPARATE_HINTS = (
    "each part separate", "each part separately", "parts separate", "separate parts",
    "all parts separate", "every part separate", "exploded", "parts laid out",
    "lay out the parts", "laid out", "individual parts", "as separate parts",
    "each component separate",
)


_MULTI_PART_HINTS = (
    "assembly", "assemble", "parts", "components", "each part", "separate",
    "exploded", "multi-part", "multipart", "set of", "and a ", "with a separate",
    "mechanism", "linkage", "gearbox", "assembled",
)


_PICTURE_WORDS = (
    "isometric projection", "clean white background", "white background",
    "crisp lines", "line weights", "dimension arrows", "dimensional arrows",
    "photorealistic", "photo-realistic", "rendering of", "3d render",
    "catalog rendering", "blueprint style", "patent-style", "patent style",
    "annotations", "title block",
)


def _asks_for_a_picture(prompt: str) -> bool:
    low = (prompt or "").lower()
    return sum(1 for w in _PICTURE_WORDS if w in low) >= 2


def _looks_multi_part(prompt: str) -> bool:
    low = (prompt or "").lower()

    if any(s in low for s in ("single ", "one piece", "one-piece", "model one", "a single")):
        return False
    return any(h in low for h in _MULTI_PART_HINTS)


def _wants_separate_layout(prompt: str) -> bool:
    """True when the user explicitly wants the parts laid out separately (exploded) rather than assembled."""
    low = prompt.lower()
    return any(h in low for h in _SEPARATE_HINTS)


_ASSEMBLY_PLAN_SYS = """You plan how a CAD request should be modeled.
Decide whether it is a SINGLE part or a MULTI-PART model — an object naturally made of
several distinct physical parts, or a combination of several sub-models.

Return ONLY JSON:
{
  "assembly": true | false,
  "layout": "assembled" | "separate",
  "title": "short name",
  "parts": [ {"name": str, "description": str, "position": str} ]
}

Rules:
- "assembly": true ONLY if the object is genuinely several distinct parts (e.g. a click
  pen = barrel + tip cone + ink tube + push button + clip + spring; a desk lamp = base +
  arm + head; a gearbox = housing + gears + shafts; a robot = body + head + arms + legs).
  false for a single monolithic part (a bracket, a gear, a phone stand, one box even with
  internal compartments).
- "layout": "separate" if the user asked for the parts SEPARATELY / exploded / laid out /
  as individual parts; otherwise "assembled" (the DEFAULT) — parts positioned in their
  correct relative locations so together they form the real, working object.
- "parts": every distinct physical part. Each has a short "description" INCLUDING rough
  dimensions in mm, and "position": where it sits relative to the whole assembly (e.g.
  "centered at the bottom", "on top of the base pointing up", "inside the front of the
  barrel"). Use 2 to 12 parts. If "assembly" is false, return "parts": [].
Output ONLY the JSON."""


def _plan_assembly(prompt: str) -> dict[str, Any] | None:
    """Classify a request as single vs multi-part and (if multi) return the parts + layout."""
    from textocad.core.config import settings
    from textocad.services.llm_service import GenerationConfig, _get_client

    client = _get_client()
    response = client.models.generate_content(
        contents=prompt,
        config=GenerationConfig(
            system_instruction=_ASSEMBLY_PLAN_SYS,
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )
    data = _extract_json(response.text or "")
    if not isinstance(data, dict) or not data.get("assembly"):
        return None
    raw_parts = data.get("parts") if isinstance(data.get("parts"), list) else []
    parts: list[dict[str, Any]] = []
    for p in raw_parts[:12]:
        if isinstance(p, dict) and str(p.get("name") or "").strip():
            parts.append(
                {
                    "name": str(p.get("name")).strip(),
                    "description": str(p.get("description") or "").strip(),
                    "position": str(p.get("position") or "").strip(),
                }
            )
    if len(parts) < 2:
        return None
    layout = "separate" if str(data.get("layout") or "").strip().lower() == "separate" else "assembled"
    return {"parts": parts, "layout": layout}


def _plan_assembly_safe(prompt: str) -> dict[str, Any] | None:
    """Plan an assembly; None if unavailable or single-part (then we fall back to single-solid generation, which."""
    from textocad.services.llm_models import helper_tier

    try:


        with helper_tier("plan"):
            return _plan_assembly(prompt)
    except Exception:
        return None


def generate_codegen_package(
    prompt: str,
    mode: str | None = None,
    *,
    prior_error: str | None = None,
    prior_script: str | None = None,
    prior_prompt: str | None = None,
    assembly_parts: list[dict[str, Any]] | None = None,
    assembly_layout: str = "assembled",
    image: dict[str, Any] | None = None,
) -> CodegenPackage:
    """Ask the model for a {param_schema + script}."""
    from textocad.services.llm_service import multimodal_generate

    resolved_mode = mode or route_mode(prompt)
    printing = resolved_mode == "printing"
    system = (
        _PRINTING_SYS if printing
        else _ORGANIC_SYS if resolved_mode == "organic"
        else _PRECISION_SYS
    )


    one_result = (
        "one `module part_<name>()` per part (each with its own `// @part` colour line)"
        if printing
        else "a dict of NAMED parts returned from build(), e.g. "
        "{'base': base, 'lid': (lid, '#e07b39')} - do NOT union them"
    )

    if assembly_parts:
        plist = "\n".join(
            (
                f"- {p['name']}: {p.get('description', '')}"
                + (f" [position: {p['position']}]" if p.get("position") else "")
            ).rstrip(": ")
            for p in assembly_parts
        )
        if assembly_layout == "separate":
            user = (
                "Build this as a MULTI-PART model with the parts SEPARATE (exploded). "
                "Model EACH part below as its own solid, then LAY THEM OUT in a row "
                "spaced apart along X (translate each part by an increasing multiple of a "
                "'layout_gap' param) so no two parts touch. Return them as "
                f"{one_result}. Detail every part realistically; expose each part's "
                "key dimensions as params. Add 'layout_gap' as a parameter.\n\n"
                f"Parts to model separately:\n{plist}\n\n"
                f"Original request: {prompt}"
            )
        else:
            user = (
                "Build this as ONE ASSEMBLED model made of the parts below. Model EACH "
                "part as its own solid and POSITION each in its correct relative location "
                "(use its [position] note) so the parts fit together and form the real, "
                "working object — parts should mate/touch as they would in reality, NOT "
                f"be spaced apart. Return them as {one_result}, each already in its "
                "assembled position. Detail every part "
                "realistically; expose each part's key dimensions as params.\n\n"
                f"Parts (model each, then assemble in place):\n{plist}\n\n"
                f"Original request: {prompt}"
            )
    elif prior_script and prior_script.strip():
        prev = f"Previous request: {prior_prompt}\n" if prior_prompt else ""
        lang = "openscad" if printing else "python"
        what = (
            "the COMPLETE updated OpenSCAD script in one fenced block" if printing
            else "the COMPLETE updated JSON (title, param_schema, script)"
        )
        user = (
            "You are EDITING an existing design, not starting over. Here is the "
            "current script:\n\n"
            f"```{lang}\n{prior_script.strip()}\n```\n\n"
            f"{prev}New request: {prompt}\n\n"
            f"Return {what}. KEEP the "
            "existing geometry/features and ADD the new request on top, combining them "
            "into one connected solid where it makes sense (e.g. place the new part "
            "relative to the existing one, or as a new named part). Preserve existing "
            "param keys and part names and add new "
            "params only as needed. Do not drop earlier features unless the new request "
            "explicitly replaces them."
        )
    else:
        user = prompt
    if image:
        user = (
            f"{user}\n\nA REFERENCE IMAGE is attached. Match its shape/proportions/"
            "features. If it is a dimensioned drawing, READ and use those dimensions; "
            "otherwise treat it as a shape reference and keep any dimensions stated in "
            "the text above."
        )


    if _asks_for_a_picture(prompt):
        user = (
            f"{user}\n\nNOTE: this request is phrased as a drawing or rendering. "
            "You cannot produce images. Model the 3D object it describes as a "
            "solid, and ignore everything about presentation - views, "
            "backgrounds, line weights, annotations and dimension arrows."
        )

    if prior_error:
        low = prior_error.lower()
        hint = ""
        if printing and ("produced no geometry" in low or "no geometry" in low):
            hint = (
                " A part module drew nothing. Every part_<name>() module must "
                "produce a closed solid; delete empty modules."
            )
        elif printing:
            hint = (
                " Fix the OpenSCAD problem above. Keep to the contract: one fenced "
                "```openscad block, Customizer parameters at the top, geometry only "
                "inside no-argument `part_<name>()` modules, no top-level part calls, "
                "only BOSL2 includes."
            )
        elif "stdfail" in low or "brep_api" in low or "command not done" in low:
            hint = (
                " This is an OpenCASCADE geometry failure — almost always a "
                "fillet/chamfer the kernel could not apply. REDUCE the radii (keep them "
                "well under half the wall thickness), apply fillets only to a few "
                "specific edges, and WRAP each fillet in try/except so a failed one is "
                "skipped while the rest of the (valid) solid is kept. If a fillet keeps "
                "failing, drop it entirely and return the un-filleted solid."
            )
        elif (
            "ast validation" in low
            or "build(params)" in low
            or "not an object" in low
            or "parseable json" in low
        ):


            hint = (
                " This was NOT a geometry failure - the response was the wrong "
                "SHAPE. Reply with a single raw JSON object and nothing else: no "
                "prose, no markdown fence, and not a JSON string containing JSON. "
                "It must have exactly the keys title, param_schema and script. "
                "`script` is Python source defining ONE top-level function "
                "`def build(params):` that returns the solid. You cannot produce "
                "images, renderings or drawings - if the request asks for one, "
                "model the 3D object it describes and ignore the presentation."
            )
        user = (
            f"{user}\n\nThe previous attempt FAILED with:\n{prior_error}\n"
            "Fix the script so it builds a valid result and matches the request." + hint
        )

    raw = multimodal_generate(system, user, image=image, temperature=0.2)
    if printing:
        return _printing_package(raw, prompt)
    data = _extract_package_json(raw, prompt=prompt, mode=resolved_mode)
    script = _fix_known_codegen_pitfalls(str(data.get("script") or ""), resolved_mode)
    title = str(data.get("title") or prompt[:40])
    schema = data.get("param_schema") or []
    if not isinstance(schema, list):
        schema = []

    v = validate_script(script, resolved_mode)
    if not v.ok:
        raise ValueError("generated script failed AST validation: " + "; ".join(v.errors))

    return CodegenPackage(
        mode=resolved_mode, title=title, param_schema=schema, script=script, prompt=prompt
    )


_SCAD_FENCE = re.compile(r"```(?:openscad|scad)?[ \t]*\n(.*?)```", re.S | re.I)


def _extract_scad(raw: str) -> str:
    """The OpenSCAD script from the model's reply: the fenced block, else the whole reply when it already looks."""
    text = raw or ""
    blocks = [b for b in _SCAD_FENCE.findall(text) if "module" in b or ";" in b]
    if blocks:
        return max(blocks, key=len).strip()
    if re.search(r"\bmodule\s+part_\w+\s*\(", text):
        return text.strip()
    raise CodegenFormatError(
        "The reply had no OpenSCAD code block. Reply with ONE fenced ```openscad block "
        "containing the whole script and nothing else.",
        raw=text,
    )


def _printing_package(raw: str, prompt: str) -> CodegenPackage:
    from textocad.services.openscad_engine import parse_model, validate_scad

    script = _extract_scad(raw)
    errors = validate_scad(script)
    if errors:
        raise ValueError("generated OpenSCAD failed validation: " + "; ".join(errors))
    model = parse_model(script)
    title = model.title if model.title != "3D print model" else prompt[:40]
    return CodegenPackage(
        mode="printing", title=title, param_schema=model.params, script=script, prompt=prompt
    )


def run_codegen(
    package: CodegenPackage,
    params: dict[str, Any] | None = None,
    *,
    build_dir: str | None = None,
    timeout_s: int = 60,
) -> ExecResult:
    """Execute a package's cached script with given params (slider edits call this directly; NO LLM)."""
    from textocad.services.cancellation import check as cancel_check

    cancel_check()
    p = dict(package.default_params())
    if params:
        p.update(params)
    if package.mode == "printing":
        from textocad.services.openscad_engine import compile_model, parse_params

        schema = package.param_schema or parse_params(package.script)
        return compile_model(
            package.script, p, build_dir or tempfile.mkdtemp(prefix="scad_"), schema=schema
        )
    return execute_script(
        package.script, p, package.mode, build_dir=build_dir, timeout_s=timeout_s
    )


_VERIFY_SIGNALS = (
    "teeth", "tooth", "gear", "hole", "bore", "fillet", "chamfer", "thread",
    "spiral", "bolt", "screw", "nut", "washer", "bracket", "flange", "spur",
    "counterbore", "countersink", "diameter", "radius", "mm", "cm",
)


def _worth_verifying(prompt: str) -> bool:
    low = (prompt or "").lower()
    return any(s in low for s in _VERIFY_SIGNALS) or any(c.isdigit() for c in low)


def _extract_spec_safe(prompt: str, image: dict[str, Any] | None = None) -> Any | None:
    """Extract a checkable RequirementSpec from the prompt (and reference image, so the gate verifies what the."""
    from textocad.services.llm_models import helper_tier

    try:
        from textocad.services.requirement_spec import extract_requirement_spec


        with helper_tier("extract", has_image=bool(image)):
            return extract_requirement_spec(prompt, image=image)
    except Exception:
        return None


def _load_geometry_sample(exec_res: ExecResult) -> Any | None:
    """Build a GeometrySample from a successful build's exported files: the mesh (STL -> trimesh, for."""
    try:
        import trimesh

        from textocad.services.verification import GeometrySample
    except Exception:
        return None
    if not exec_res.stl_path:
        return None
    try:
        mesh = trimesh.load(exec_res.stl_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(geoms) if len(geoms) > 1 else geoms[0]
    except Exception:
        return None
    solid = None
    if exec_res.step_path:
        try:
            import cadquery as cq

            solid = cq.importers.importStep(exec_res.step_path)
        except Exception:
            solid = None
    return GeometrySample(solid=solid, mesh=mesh)


def _verify_geometry_inprocess(spec: Any, exec_res: ExecResult) -> Any | None:
    """Load the built geometry and check it against the spec, in THIS process."""
    sample = _load_geometry_sample(exec_res)
    if sample is None:
        return None
    try:
        from textocad.services.verification import verify

        return verify(sample, spec)
    except Exception:
        return None


def _verify_geometry(spec: Any, exec_res: ExecResult) -> Any | None:
    """Measure the built geometry back and assert it against the spec."""
    from textocad.services.geometry_isolation import IsolationError, run_isolated

    try:
        return run_isolated(_verify_geometry_inprocess, spec, exec_res)
    except IsolationError as exc:
        _log(f"verification skipped: {exc}")
        return None


def _assembly_issues(exec_res: ExecResult) -> list[str]:
    """Parts of an assembled object that float free (see part_export.check_assembly)."""
    if len(exec_res.parts or []) < 2 or not exec_res.stl_path:
        return []
    from pathlib import Path

    from textocad.services.geometry_isolation import IsolationError, run_isolated
    from textocad.services.part_export import check_assembly

    parts_dir = Path(exec_res.stl_path).parent / "parts"
    try:
        return list(run_isolated(check_assembly, str(parts_dir), timeout_s=120) or [])
    except (IsolationError, Exception) as exc:
        _log(f"assembly check skipped: {exc}")
        return []


def generate_and_build(
    prompt: str,
    mode: str | None = None,
    *,
    build_dir: str | None = None,
    max_attempts: int = 3,
    timeout_s: int = 60,
    prior_script: str | None = None,
    prior_prompt: str | None = None,
    image: dict[str, Any] | None = None,
    extra_details: bool = False,
) -> tuple[CodegenPackage | None, ExecResult]:
    """Generate -> build, with an optional "Extra Details" deep path."""
    resolved_mode = mode or route_mode(prompt)


    plan = (
        _plan_assembly_safe(prompt)
        if (
            resolved_mode in ("precision", "printing")
            and not prior_script
            and (extra_details or _looks_multi_part(prompt))
        )
        else None
    )
    assembly_parts = plan["parts"] if plan else None
    assembly_layout = (
        "separate"
        if (plan and (plan["layout"] == "separate" or _wants_separate_layout(prompt)))
        else "assembled"
    )


    spec = (
        _extract_spec_safe(prompt, image)
        if (
            (extra_details or settings.verification_gate_enabled)
            and resolved_mode in ("precision", "printing")
            and not prior_script
            and not assembly_parts
            and _worth_verifying(prompt)
        )
        else None
    )

    prior: str | None = None
    last: ExecResult = ExecResult(ok=False, error="no attempt made")
    pkg: CodegenPackage | None = None


    best_pkg: CodegenPackage | None = None
    best_last: ExecResult | None = None

    def _notes_for(report: Any | None) -> list[str]:
        notes: list[str] = []
        if assembly_parts:
            word = "laid out separately" if assembly_layout == "separate" else "assembled"
            notes.append(
                f"Multi-part model ({word}) — {len(assembly_parts)} parts: "
                + ", ".join(p["name"] for p in assembly_parts)
            )
        if spec is not None and report is not None:
            notes.append(
                f"Verified — {spec.describe()}" if report.passed else report.summary()
            )
        return notes


    import time as _t

    started = _t.perf_counter()
    budget = max(int(settings.generation_budget_seconds), 60)

    def _elapsed() -> float:
        return _t.perf_counter() - started


    downgraded = False
    escalated = False

    def _escalate() -> bool:
        """Move the rest of this generation onto the escalation model."""
        nonlocal escalated
        if escalated:
            return False
        try:
            from textocad.core.config import setting_or_default
            from textocad.services import llm_models

            spec = setting_or_default("escalation_model").strip()
            if spec.lower() in ("", "off", "none", "disabled", "false"):
                return False
            provider, _, model_id = spec.partition(":")
            provider, model_id = provider.strip().lower(), model_id.strip()
            if not provider or not model_id:
                return False

            active = llm_models.get_active()
            stronger = llm_models.resolve_utility(provider, model_id)
            if stronger is None:
                _log(f"escalation target {spec} is not configured - staying put")
                return False
            if active is not None and stronger.model == active.model:
                return False

            llm_models.set_active(stronger)
            escalated = True
            _log(
                f"ESCALATING to {stronger.model} after attempt 1 failed "
                f"({_elapsed():.0f}s)"
            )
            return True
        except Exception:
            return False

    def _downgrade() -> bool:
        """Move the rest of this generation onto the fast tier. False if already there or nothing to move to."""
        nonlocal downgraded
        if downgraded:
            return False
        try:
            from textocad.services import llm_models

            active = llm_models.get_active()
            if active is None:
                return False
            fast = llm_models.resolve(active.provider, "fast")
            if fast is None or fast.model == active.model:
                return False
            llm_models.set_active(fast)
            downgraded = True
            _log(f"dropping to {fast.model} for the remaining attempts")
            return True
        except Exception:
            return False


    last_was_format = False
    format_retried = False

    repair_from: str | None = None
    assembly_retried = False
    check_assembly = resolved_mode in ("printing", "precision") and not (
        assembly_layout == "separate" or _wants_separate_layout(prompt)
    )

    for attempt_no in range(max_attempts):


        format_retry = last_was_format and not format_retried
        if attempt_no > 0 and _elapsed() > budget and not _downgrade() and not format_retry:
            _log(f"budget {budget}s exhausted after {_elapsed():.0f}s - stopping retries")
            break
        if format_retry:
            format_retried = True


        if attempt_no > 0:
            _escalate()


        if attempt_no > 0 and not escalated and _elapsed() > budget * 0.5:
            _downgrade()
        _log(f"attempt {attempt_no + 1}/{max_attempts} generating ({_elapsed():.0f}s elapsed)")
        try:
            pkg = generate_codegen_package(
                prompt,
                resolved_mode,
                prior_error=prior,
                prior_script=repair_from or prior_script,
                prior_prompt=prior_prompt,
                assembly_parts=assembly_parts,
                assembly_layout=assembly_layout,


                image=image if prior is None else None,
            )
        except ProviderError:


            raise
        except Exception as exc:
            prior = str(exc)
            last_was_format = isinstance(exc, CodegenFormatError)
            last = ExecResult(ok=False, error=prior)


            _log(
                f"attempt {attempt_no + 1} GENERATION FAILED "
                f"({_elapsed():.0f}s): {prior[:200]}"
            )
            continue

        _log(f"attempt {attempt_no + 1} script received ({_elapsed():.0f}s) - building")
        last = run_codegen(pkg, build_dir=build_dir, timeout_s=timeout_s)
        if not last.ok:


            _log(f"attempt {attempt_no + 1} BUILD FAILED ({_elapsed():.0f}s): {last.error[:160]}")
            prior = last.error
            last_was_format = False
            continue


        report = _verify_geometry(spec, last) if spec is not None else None


        floating = _assembly_issues(last) if check_assembly else []
        if floating and not assembly_retried and attempt_no + 1 < max_attempts and _elapsed() <= budget:
            assembly_retried = True
            best_pkg, best_last = pkg, last
            repair_from = pkg.script if pkg is not None else None
            _log(f"attempt {attempt_no + 1} built but parts float ({_elapsed():.0f}s) - repairing")
            prior = (
                "The model BUILT, but its parts do not join up:\n- "
                + "\n- ".join(floating)
                + "\nKeep every part and dimension. Only fix their positions so each part "
                "touches or slightly overlaps the part it attaches to."
            )
            continue


        if (
            extra_details
            and spec is not None
            and report is not None
            and not report.passed
            and attempt_no + 1 < max_attempts
            and _elapsed() <= budget
        ):
            best_pkg, best_last = pkg, last
            _log(
                f"attempt {attempt_no + 1} built but MISMATCH ({_elapsed():.0f}s) - "
                f"repairing (ED): {report.summary()[:160]}"
            )
            prior = (
                "The model BUILT correctly but does NOT match the request. "
                + report.summary()
                + "\nFix ONLY these discrepancies and keep everything that was already correct."
            )
            continue


        notes = _notes_for(report)
        if floating:
            notes.append("Some parts may not touch: " + "; ".join(floating[:3]))
        if pkg is not None and notes:
            pkg.notes = notes
        if pkg is not None and escalated:
            try:
                from textocad.services import llm_models

                current = llm_models.get_active()
                pkg.escalated_to = current.model if current else "escalated"
            except Exception:
                pkg.escalated_to = "escalated"
        _log(f"BUILD OK on attempt {attempt_no + 1} ({_elapsed():.0f}s) - shipping")
        return pkg, last


    if best_pkg is not None and best_last is not None:
        report = _verify_geometry(spec, best_last) if spec is not None else None
        notes = _notes_for(report)
        if notes:
            best_pkg.notes = notes
        _log(f"ED attempts exhausted - shipping best buildable ({_elapsed():.0f}s)")
        return best_pkg, best_last


    if resolved_mode in ("precision", "printing"):


        if last_was_format and resolved_mode == "printing":
            safe_prior = (
                "Every attempt was rejected for its FORMAT. Reply with ONE fenced "
                "```openscad block containing the whole script and nothing else. Keep "
                "the object simple and certain to compile."
            )
        elif resolved_mode == "printing":
            safe_prior = (
                (f"Every attempt failed to compile. Last error:\n{last.error}\n" if last.error
                 else "Every attempt failed to compile.\n")
                + "Produce the SIMPLEST robust version that is GUARANTEED to compile: plain "
                "OpenSCAD primitives (cube/cylinder/difference/union), correct overall "
                "dimensions and the main holes/cut-outs, no BOSL2, no minkowski()."
            )
        elif last_was_format:
            safe_prior = (
                (
                    "Every attempt was rejected for its FORMAT. Last error:\n"
                    f"{last.error}\n"
                    if last.error
                    else "Every attempt was rejected for its FORMAT.\n"
                )
                + "Reply with a single raw JSON object and NOTHING else: no prose, no "
                "markdown fence, and not a JSON string containing JSON. Exactly the keys "
                "title, param_schema and script, where script is Python source defining "
                "one top-level `def build(params):` that returns the solid. Keep the part "
                "simple and certain to build."
            )
        else:
            safe_prior = (
                (
                    "Every attempt failed in the CAD kernel. Last error:\n"
                    f"{last.error}\n"
                    if last.error
                    else "Every attempt failed in the CAD kernel.\n"
                )
                + "Produce the SIMPLEST robust version that is GUARANTEED to build: the core "
                "solid with correct overall dimensions and the main cuts/holes/compartments, "
                "but OMIT all fillets, chamfers, and any other fragile/decorative features."
            )
        try:


            from textocad.services.llm_models import fast_tier

            _log(f"safe-mode attempt ({_elapsed():.0f}s) - simplest buildable solid")
            with fast_tier():
                spkg = generate_codegen_package(
                    prompt,
                    resolved_mode,
                    prior_error=safe_prior,
                    prior_script=prior_script,
                    prior_prompt=prior_prompt,
                    assembly_parts=assembly_parts,
                    assembly_layout=assembly_layout,
                )
            sres = run_codegen(spkg, build_dir=build_dir, timeout_s=timeout_s)
            if sres.ok:
                if spkg is not None:
                    spkg.notes = [
                        "Built a simplified version — fragile features (e.g. fillets/"
                        "chamfers the CAD kernel rejected) were omitted so the model "
                        "would build."
                    ]
                return spkg, sres
        except Exception:
            pass

    return pkg, last

"""Phase 1 — Requirement spec."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class DimensionReq(BaseModel):
    name: str
    axis: Literal["x", "y", "z"] | None = None
    value: float
    tol: float = 0.5
    measure: Literal["bbox", "feature"] = "bbox"


class BBoxReq(BaseModel):

    x: tuple[float, float] | None = None
    y: tuple[float, float] | None = None
    z: tuple[float, float] | None = None


class FeatureReq(BaseModel):
    kind: Literal["hole", "gear_teeth", "fillet", "chamfer", "spiral", "boss", "rib"]
    of: str | None = None
    count: int | None = None
    present: bool | None = None
    r: float | None = None
    angle_deg: float | None = None
    tol: float | None = None


class TopologyReq(BaseModel):
    watertight: bool = True
    solids: int | None = 1


class JointReq(BaseModel):
    type: Literal["revolute", "prismatic", "continuous", "fixed"] = "revolute"
    axis: Literal["x", "y", "z"] | None = None


class RobotReq(BaseModel):
    dof: int | None = None
    joints: list[JointReq] = Field(default_factory=list)


class RequirementSpec(BaseModel):
    title: str = ""
    dimensions: list[DimensionReq] = Field(default_factory=list)
    bbox: BBoxReq | None = None
    features: list[FeatureReq] = Field(default_factory=list)
    topology: TopologyReq = Field(default_factory=TopologyReq)
    robot: RobotReq | None = None

    extraction_notes: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        """Human-readable summary of what the gate will enforce (for surfacing to the user)."""
        parts: list[str] = []
        if self.title:
            parts.append(f"'{self.title}'")
        if self.bbox:
            for axis in ("x", "y", "z"):
                rng = getattr(self.bbox, axis)
                if rng:
                    parts.append(f"{axis}∈[{rng[0]},{rng[1]}]mm")
        for d in self.dimensions:
            parts.append(f"{d.name}={d.value}±{d.tol}mm")
        for f in self.features:
            if f.kind == "gear_teeth" and f.count is not None:
                parts.append(f"{f.count} teeth")
            elif f.kind in {"fillet", "chamfer"}:
                parts.append(f"{f.kind} r={f.r}mm on {f.of or 'edges'}")
            elif f.kind == "spiral" and f.angle_deg is not None:
                parts.append(f"spiral {f.angle_deg}°")
            elif f.kind == "thread":
                parts.append("thread")
            elif f.kind == "hole" and f.count is not None:
                parts.append(f"{f.count} holes")
        if self.topology.watertight:
            parts.append("watertight")
        if self.robot and self.robot.dof:
            parts.append(f"{self.robot.dof}-DOF")
        return "built to: " + ", ".join(parts) if parts else "no checkable requirements extracted"


_FEATURE_KEYWORDS: dict[str, list[str]] = {
    "gear_teeth": ["teeth", "tooth", "gear", "involute", "spur gear", "helical gear"],
    "thread": ["thread", "threaded", "screw thread", "metric thread", "m6", "m8", "m10"],
    "fillet": ["fillet", "filleted", "rounded edge", "round-over", "round over"],
    "chamfer": ["chamfer", "chamfered", "beveled edge"],
    "hole": ["hole", "bore", "through-hole", "counterbore", "cbore", "countersink", "csk", "drill"],
    "spiral": ["spiral", "helix angle", "spiral angle"],
    "robot": ["dof", "degrees of freedom", "revolute joint", "prismatic joint", "robot arm", "links and joints"],
}


def find_spec_gaps(prompt: str, spec: RequirementSpec) -> list[str]:
    """Return human warnings for prompt feature-keywords that produced NO matching spec feature."""
    low = prompt.lower()
    have_kinds = {f.kind for f in spec.features}
    has_robot = spec.robot is not None and bool(spec.robot.dof or spec.robot.joints)
    gaps: list[str] = []
    for kind, words in _FEATURE_KEYWORDS.items():
        matched = next((w for w in words if w in low), None)
        if not matched:
            continue
        satisfied = has_robot if kind == "robot" else (kind in have_kinds)
        if not satisfied:
            gaps.append(
                f"prompt mentions '{matched}' but no '{kind}' requirement was extracted"
            )
    return gaps


_EXTRACTION_INSTRUCTIONS = """You convert a CAD prompt into a STRUCTURED, CHECKABLE requirement spec.
A downstream gate measures the built geometry and asserts it against your spec, so you
MUST capture every verifiable feature the prompt states — not just size and watertightness.

Return ONLY JSON of this shape (omit fields you truly cannot determine; never invent numbers):
{
  "title": str,
  "bbox": {"x":[min,max]|null, "y":[min,max]|null, "z":[min,max]|null},
  "dimensions": [{"name":str,"axis":"x|y|z"|null,"value":number,"tol":number,"measure":"bbox|feature"}],
  "features": [{"kind":"hole|gear_teeth|fillet|chamfer|spiral|thread|boss|rib","of":str|null,
                "count":int|null,"present":bool|null,"r":number|null,"angle_deg":number|null,"tol":number|null}],
  "topology": {"watertight": true, "solids": 1},
  "robot": {"dof":int,"joints":[{"type":"revolute|prismatic|continuous|fixed","axis":"x|y|z"}]} | null
}

FEATURE EXTRACTION IS MANDATORY. Map prompt language to features:
- "N teeth" / "gear" / "spur/helical gear"        -> feature {"kind":"gear_teeth","count":N}
- "spiral angle X" / "helix angle X"              -> feature {"kind":"spiral","angle_deg":X}
- "M6/threaded/screw thread"                      -> feature {"kind":"thread"}
- "fillet r" / "rounded edges"                    -> feature {"kind":"fillet","present":true,"r":R}
- "chamfer"                                        -> feature {"kind":"chamfer","present":true,"r":R}
- "N holes" / "bore" / "counterbore/countersink"  -> feature {"kind":"hole","count":N}
- "N-DOF" / joints                                -> robot {"dof":N,"joints":[...]}
If the prompt names a quantity (teeth, holes, DOF), you MUST emit that feature with the count.
Set bbox from an explicit overall size with a small tolerance. Default topology.watertight=true.

EXAMPLES:
prompt: "a spur gear with 36 teeth, module 5, 10 mm wide"
json: {"title":"Spur gear","features":[{"kind":"gear_teeth","of":"gear","count":36}],
       "dimensions":[{"name":"width","axis":"z","value":10,"tol":0.5}],"topology":{"watertight":true,"solids":1}}

prompt: "an L-bracket 80x60x40, 4 mounting holes, 3mm fillet on the inner corner"
json: {"title":"L-bracket","bbox":{"x":[79,81],"y":[59,61],"z":[39,41]},
       "features":[{"kind":"hole","count":4},{"kind":"fillet","of":"inner_corner","present":true,"r":3}],
       "topology":{"watertight":true,"solids":1}}

prompt: "a 40 mm cube"
json: {"title":"Cube","bbox":{"x":[39.5,40.5],"y":[39.5,40.5],"z":[39.5,40.5]},"topology":{"watertight":true,"solids":1}}"""


def _extract_once(
    prompt: str, extra: str = "", image: dict[str, Any] | None = None
) -> RequirementSpec:
    from textocad.services.llm_service import multimodal_generate

    user = f"{extra}\n\nPROMPT:\n{prompt}".strip()
    if image:
        user = (
            f"{user}\n\nA reference image is attached — also extract any features/"
            "dimensions it clearly shows (e.g. visible holes, overall proportions)."
        )
    raw = multimodal_generate(
        _EXTRACTION_INSTRUCTIONS, user, image=image, temperature=0.0
    ).strip()
    data: dict[str, Any] = json.loads(raw) if raw else {}
    return RequirementSpec.model_validate(data)


def extract_requirement_spec(
    prompt: str, max_retries: int = 1, *, image: dict[str, Any] | None = None
) -> RequirementSpec:
    """Extract a RequirementSpec (JSON mode), with a completeness-driven retry."""
    spec = _extract_once(prompt, image=image)
    gaps = find_spec_gaps(prompt, spec)

    attempts = 0
    while gaps and attempts < max_retries:
        attempts += 1
        nudge = (
            "Your previous extraction was INCOMPLETE. The prompt clearly implies features "
            "you omitted:\n" + "\n".join(f"- {g}" for g in gaps)
            + "\nRe-extract and INCLUDE those features with their counts/values."
        )
        spec = _extract_once(prompt, extra=nudge, image=image)
        gaps = find_spec_gaps(prompt, spec)

    if gaps:
        spec.extraction_notes = [
            "Spec may be under-specified — the gate cannot enforce what wasn't extracted:",
            *gaps,
        ]
    return spec

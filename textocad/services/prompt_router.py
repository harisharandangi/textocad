"""Pick the model tier and the verification path from the prompt itself."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace

from textocad.core.config import setting_or_default, settings

logger = logging.getLogger(__name__)


_HARD_GEOMETRY = (
    "gear", "involute", "sprocket", "worm", "helical", "helix", "thread",
    "threaded", "screw thread", "cam ", "camshaft", "spline", "impeller",
    "turbine", "propeller", "auger", "knurl", "lofted", "loft ", "sweep",
    "revolve", "airfoil", "gearbox",
)


_ASSEMBLY = (
    "assembly", "assemble", "sub-assembly", "subassembly", "mating",
    "mates with", "two parts", "three parts", "multi-part", "multipart",
    "exploded", "each part", "separate parts", "set of parts", "kit",
)


_TOLERANCE = (
    "tolerance", "clearance", "press fit", "press-fit", "interference fit",
    "slip fit", "sliding fit", "±", "+/-", "h7", "h8", "g6", "k6",
    "snug", "backlash",
)


_ACCESSORY = (
    "guard", "mount", "holder", "cover", "case", "bracket", "clip", "stand",
    "enclosure", "housing", "shroud", "adapter", "spacer",
)


_SIMPLE_SHAPES = (
    "cube", "box", "cuboid", "cylinder", "sphere", "cone", "disc", "disk",
    "washer", "spacer", "ring", "rod", "bar", "plate", "sheet", "block",
    "tube", "pipe", "square", "rectangle", "triangle", "pyramid",
)


_DIMENSION = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mm|cm|millimet|centimet|meters?\b|m\b|in\b|inch|inches|\"|deg\b|degrees?|°)",
    re.I,
)


_TOOTH_COUNT = re.compile(r"\b(\d+)\s*(?:teeth|tooth)\b", re.I)
_HOLE_COUNT = re.compile(r"\b(\d+)\s*(?:holes?|bores?|slots?|cutouts?)\b", re.I)


_RADIUS_WORD = r"(?:fillet|chamfer|radius|rounded corners?|corner radius)"
_RADIUS_SPEC = re.compile(
    r"\d[^.]{0,20}?\b" + _RADIUS_WORD + r"\b|\b" + _RADIUS_WORD + r"\b[^.]{0,30}?\d",
    re.I,
)
_GEAR_MODULE = re.compile(r"\bmodule\s*\d|\bpitch\s+diameter\b|\bpressure\s+angle\b", re.I)


@dataclass(frozen=True)
class Route:
    """What to run, and why."""

    speed: str
    extra_details: bool
    reason: str
    complexity: int


    provider: str | None = None


    source: str = "heuristic"


def _count(text: str, needles: tuple[str, ...]) -> int:
    return sum(1 for n in needles if n in text)


def _heuristic_route(
    prompt: str,
    *,
    is_edit: bool = False,
    mode: str | None = None,
) -> Route:
    """Choose a speed tier and whether to run the verification path."""
    text = (prompt or "").strip()
    low = text.lower()
    words = len(text.split())

    dimensions = len(_DIMENSION.findall(text))
    hard = _count(low, _HARD_GEOMETRY)
    assembly = _count(low, _ASSEMBLY)
    tolerance = _count(low, _TOLERANCE)
    simple = _count(low, _SIMPLE_SHAPES)


    if hard and words < 15 and _count(low, _ACCESSORY):
        hard = 0

    tooth = bool(_TOOTH_COUNT.search(text) or _GEAR_MODULE.search(text))
    holes = bool(_HOLE_COUNT.search(text))
    radii = bool(_RADIUS_SPEC.search(text))

    reasons: list[str] = []
    score = 0


    if hard:
        score += 4
        reasons.append("hard geometry")
    if assembly:
        score += 4
        reasons.append("multi-part")
    if tolerance:
        score += 2
        reasons.append("fits and tolerances")


    if dimensions >= 14:
        score += 1
        reasons.append(f"{dimensions} dimensions")
    if words >= 180:
        score += 1
        reasons.append("very long spec")


    if (mode or "") == "organic":
        score -= 2
        reasons.append("organic")


    if is_edit and not hard and not assembly:
        score -= 2
        reasons.append("edit of an existing design")


    if simple and words < 30 and dimensions <= 3 and not hard and not assembly:
        score -= 3
        reasons.append("simple shape")


    if score >= 4:
        speed = "deepthink"
    elif score <= -2:
        speed = "fast"
    else:
        speed = "standard"


    verifiable = tooth or holes or radii or tolerance > 0 or dimensions >= 4
    extra_details = bool(verifiable and (mode or "") != "organic")
    if extra_details:
        checks = [
            name
            for name, present in (
                ("tooth count", tooth),
                ("hole count", holes),
                ("fillet radii", radii),
                ("tolerances", tolerance > 0),
                ("dimensions", dimensions >= 4),
            )
            if present
        ]
        reasons.append("verifiable: " + ", ".join(checks))

    return Route(
        speed=speed,
        extra_details=extra_details,
        reason="; ".join(reasons) or "no complexity signals",
        complexity=max(score, 0),
        source="heuristic",
    )


def _split_target(speed: str) -> tuple[str, str]:
    """Parse the configured "provider:speed" for a tier."""
    raw = {
        "fast": setting_or_default("route_target_fast"),
        "standard": setting_or_default("route_target_standard"),
        "deepthink": setting_or_default("route_target_deepthink"),
    }.get(speed) or ""
    parts = raw.split(":", 1)
    provider = parts[0].strip().lower()
    at_target = parts[1].strip().lower() if len(parts) == 2 and parts[1].strip() else speed
    return provider, at_target


def _target(speed: str) -> tuple[str | None, str]:
    """The provider that should serve a tier, and the tier name to ask it for."""
    provider, at_target = _split_target(speed)
    if not provider:
        return None, speed

    from textocad.services.llm_models import resolve

    if resolve(provider, at_target) is None:
        return None, speed
    return provider, at_target


_CLASSIFIER_SYSTEM = """You route CAD generation requests to the right model. \
You do not write CAD code and you do not answer the request.

Decide two things about the user's request:

tier - how hard is it to WRITE correct CAD geometry for this?
  "simple"   a primitive or near-primitive: a box, cylinder, washer, plate,
             rod, simple bracket. Also an ordinary edit to an existing part.
  "standard" an everyday mechanical part: brackets with hole patterns,
             enclosures, flanges, mounting plates, dimensioned fixtures.
  "complex"  geometry that is genuinely hard to author correctly: gear teeth,
             threads, helices, cams, airfoils, impellers, lofts along curves,
             OR several distinct parts that must fit together.

Judge the GEOMETRY, not the wording. A long, heavily dimensioned request for a
plate is still "standard": length is more work, not harder shape. A short
request for a helical gear is "complex". A part described only by what it
attaches to ("a cap that threads onto the bottle above") is complex when the
FITTING is the hard part.

verify - does the request state a countable or measurable fact that could be
checked against the finished model? Tooth counts, hole counts, fillet radii,
tolerances, or several explicit dimensions. true or false.

Reply with ONLY a JSON object, no prose and no code fence:
{"tier": "simple|standard|complex", "verify": true|false, "why": "<8 words max>"}"""

_TIER_TO_SPEED = {"simple": "fast", "standard": "standard", "complex": "deepthink"}


_CLASSIFIER_PROMPT_CHARS = 4000

_OFF = {"", "off", "none", "disabled", "false"}


def _classify(prompt: str, *, is_edit: bool, mode: str | None) -> dict | None:
    """Ask ROUTER_MODEL ("provider:model", OpenAI-protocol) to classify the prompt."""
    spec = setting_or_default("router_model").strip()
    if spec.lower() in _OFF:
        return None
    provider, _, model_id = spec.partition(":")

    from textocad.services.llm_models import resolve_utility

    llm = resolve_utility(provider, model_id)
    if llm is None or llm.protocol != "openai":
        return None

    body = (prompt or "").strip()[:_CLASSIFIER_PROMPT_CHARS]
    if not body:
        return None

    context = []
    if is_edit:
        context.append("This is an EDIT to an existing model, not a new part.")
    if mode:
        context.append("Requested style: " + str(mode) + ".")
    user = ("\n".join(context) + "\n\n" if context else "") + "Request:\n" + body

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=llm.api_key,
            base_url=llm.base_url,
            timeout=float(settings.router_timeout_seconds),
            max_retries=0,
        )
        completion = client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=120,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.info("prompt router: classifier unavailable (%s)", exc)
        return None

    return _parse_classification(text)


def _parse_classification(text: str) -> dict | None:
    """Pull the decision out of whatever the model actually sent."""
    raw = (text or "").strip()
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    tier = str(data.get("tier") or "").strip().lower()
    if tier not in _TIER_TO_SPEED:
        return None
    return {
        "tier": tier,
        "verify": bool(data.get("verify")),
        "why": str(data.get("why") or "").strip()[:80],
    }


_ORDER = {"fast": 0, "standard": 1, "deepthink": 2}


def route_prompt(
    prompt: str,
    *,
    is_edit: bool = False,
    mode: str | None = None,
    use_classifier: bool = True,
) -> Route:
    """Choose provider, speed tier and verification path for a prompt."""
    base = _heuristic_route(prompt, is_edit=is_edit, mode=mode)

    decided = base
    if use_classifier:
        verdict = _classify(prompt, is_edit=is_edit, mode=mode)
        if verdict is None:
            configured = setting_or_default("router_model").lower() not in _OFF
            decided = replace(
                base,
                source="heuristic (classifier unavailable)" if configured else "heuristic",
            )
        else:
            speed = _TIER_TO_SPEED[verdict["tier"]]
            source = "classifier"
            if base.complexity >= 4 and _ORDER[speed] < _ORDER[base.speed]:
                speed = base.speed
                source = "classifier+floor"


            reasons = [r for r in (base.reason, verdict["why"]) if r]
            decided = replace(
                base,
                speed=speed,
                extra_details=base.extra_details or verdict["verify"],
                reason="; ".join(reasons),
                source=source,
            )

    provider, at_target = _target(decided.speed)
    if provider is None:
        return decided
    return replace(decided, provider=provider, speed=at_target)

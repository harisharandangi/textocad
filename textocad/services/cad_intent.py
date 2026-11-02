from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable


Number = int | float


@dataclass(frozen=True)
class DimensionHint:
    key: str
    value: float
    unit: str = "mm"


UNIT_TO_MM = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "millimetre": 1.0,
    "millimetres": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "centimetre": 10.0,
    "centimetres": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metre": 1000.0,
    "metres": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
    "'": 304.8,
}

COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "sixteen": 16,
    "twenty": 20,
    "twenty two": 22,
    "twenty-two": 22,
    "twenty four": 24,
    "twenty-four": 24,
}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _number(value: Any, fallback: float) -> float:
    try:
        if isinstance(value, bool):
            return fallback

        parsed = float(str(value).strip())
        if not math.isfinite(parsed):
            return fallback
        if parsed.is_integer():
            return int(parsed)
        return parsed
    except Exception:
        return fallback


def _clamp(value: Number, min_value: Number, max_value: Number) -> Number:
    return max(min_value, min(value, max_value))


def _contains_any(text: str, words: list[str] | tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _safe_identifier(value: str, fallback: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(value or "").strip())
    clean = re.sub(r"_+", "_", clean).strip("_")

    if not clean:
        clean = fallback

    if clean[0].isdigit():
        clean = f"id_{clean}"

    return clean


def _to_camel_key(value: str, fallback: str) -> str:
    clean = _safe_identifier(value, fallback)
    parts = [part for part in clean.split("_") if part]
    if not parts:
        return fallback
    return parts[0].lower() + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _unit_factor(unit: str | None) -> float:
    if not unit:
        return 1.0
    return UNIT_TO_MM.get(unit.lower().strip(), 1.0)


def _parse_number_with_optional_unit(value: str, fallback: float) -> float:
    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*(mm|millimeters?|millimetres?|cm|centimeters?|centimetres?|m|meters?|metres?|in|inch|inches|ft|feet|foot|\"|')?\s*$",
        str(value or ""),
        flags=re.IGNORECASE,
    )

    if not match:
        return _number(value, fallback)

    number = _number(match.group(1), fallback)
    unit = match.group(2)
    return number * _unit_factor(unit)


def _find_number_after_patterns(text: str, patterns: list[str], fallback: float) -> float:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            number = _number(match.group(1), fallback)
            unit = match.group(2) if len(match.groups()) >= 2 else None
            return number * _unit_factor(unit)
    return fallback


def _find_count_after_patterns(text: str, patterns: list[str], fallback: int) -> int:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        raw = str(match.group(1)).strip().lower()
        if raw in COUNT_WORDS:
            return COUNT_WORDS[raw]
        return int(round(_number(raw, fallback)))

    return fallback


def _find_angle_after_patterns(text: str, patterns: list[str], fallback: float) -> float:
    return _find_number_after_patterns(text, patterns, fallback)


def _extract_triplet(text: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    """Extract dimensions such as 100 x 50 x 20 mm."""
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return fallback

    factor = _unit_factor(match.group(4))
    return (
        _number(match.group(1), fallback[0]) * factor,
        _number(match.group(2), fallback[1]) * factor,
        _number(match.group(3), fallback[2]) * factor,
    )


def _extract_pair(text: str, fallback: tuple[float, float]) -> tuple[float, float]:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:x|×|by)\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return fallback

    factor = _unit_factor(match.group(3))
    return (
        _number(match.group(1), fallback[0]) * factor,
        _number(match.group(2), fallback[1]) * factor,
    )


def extract_dimension_hints(prompt: str) -> dict[str, float]:
    text = _clean_text(prompt)

    default_width, default_depth, default_height = _extract_triplet(text, (90, 60, 30))
    pair_width, pair_depth = _extract_pair(text, (default_width, default_depth))

    outer_diameter = _find_number_after_patterns(
        text,
        [
            r"(?:outer|outside|external|overall|total)\s*(?:dia|diameter|od)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(?:od)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:outer|outside|external|overall)\s*(?:dia|diameter|od)",
        ],
        60,
    )

    inner_diameter = _find_number_after_patterns(
        text,
        [
            r"(?:inner|inside|internal|bore|hole|center\s*hole|central\s*bore)\s*(?:dia|diameter|id)?\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(?:id)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:inner|inside|internal|bore|hole)\s*(?:dia|diameter|id)?",
        ],
        20,
    )

    diameter = _find_number_after_patterns(
        text,
        [
            r"(?:dia|diameter)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:diameter|dia)",
        ],
        outer_diameter,
    )

    radius = _find_number_after_patterns(
        text,
        [
            r"(?:radius)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:radius)",
        ],
        diameter / 2,
    )

    length = _find_number_after_patterns(
        text,
        [
            r"(?:length|long)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:long|length)",
        ],
        default_height if default_height != 30 else 60,
    )

    thickness = _find_number_after_patterns(
        text,
        [
            r"(?:thickness|thick|height|face\s*width|facewidth|width\s*of\s*gear)\s*(?:is|=|:|of|to)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:thick|thickness|height|face\s*width)",
        ],
        default_height if default_height != 30 else 12,
    )

    width = _find_number_after_patterns(
        text,
        [
            r"(?:width|wide)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:wide|width)",
        ],
        pair_width,
    )

    depth = _find_number_after_patterns(
        text,
        [
            r"(?:depth|deep)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:deep|depth)",
        ],
        pair_depth,
    )

    height = _find_number_after_patterns(
        text,
        [
            r"(?:height|tall)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:tall|height)",
        ],
        thickness,
    )

    hole_count = _find_count_after_patterns(
        text,
        [
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s*(?:bolt\s*)?holes?",
            r"(?:holes?|bolt\s*holes?)\s*(?:count|number)?\s*(?:is|=|:|of)?\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)",
        ],
        4,
    )

    tooth_count = _find_count_after_patterns(
        text,
        [
            r"(\d+|twenty\s*two|twenty-two|twenty\s*four|twenty-four|twelve|sixteen)\s*(?:teeth|tooth)",
            r"(?:teeth|tooth\s*count|toothcount)\s*(?:is|=|:|of)?\s*(\d+|twenty\s*two|twenty-two|twenty\s*four|twenty-four|twelve|sixteen)",
        ],
        24,
    )

    fin_count = _find_count_after_patterns(
        text,
        [
            r"(\d+|six|eight|ten|twelve|sixteen|twenty)\s*(?:fins?|ribs?)",
            r"(?:fins?|ribs?)\s*(?:count|number)?\s*(?:is|=|:|of)?\s*(\d+|six|eight|ten|twelve|sixteen|twenty)",
        ],
        10,
    )

    segment_count = _find_count_after_patterns(
        text,
        [
            r"(\d+)\s*(?:segments?|sides?)",
            r"(?:segments?|sides?)\s*(?:count|number)?\s*(?:is|=|:|of)?\s*(\d+)",
        ],
        48,
    )

    pressure_angle = _find_angle_after_patterns(
        text,
        [
            r"(?:pressure\s*angle)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)?",
            r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)?\s*(?:pressure\s*angle)",
        ],
        20,
    )

    helix_angle = _find_angle_after_patterns(
        text,
        [
            r"(?:helix\s*angle)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)?",
            r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)?\s*(?:helix\s*angle)",
        ],
        15,
    )

    chamfer = _find_number_after_patterns(
        text,
        [
            r"(?:chamfer)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:chamfer)",
        ],
        0.5 if "slight" in text or "slightly" in text else 1,
    )

    fillet = _find_number_after_patterns(
        text,
        [
            r"(?:fillet|rounding|corner\s*radius)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:fillet|rounding|corner\s*radius)",
        ],
        2,
    )

    wall_thickness = _find_number_after_patterns(
        text,
        [
            r"(?:wall\s*thickness|wall)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:wall\s*thickness|wall)",
        ],
        max(min(thickness, 4), 2),
    )

    pitch = _find_number_after_patterns(
        text,
        [
            r"(?:pitch)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
            r"(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?\s*(?:pitch)",
        ],
        2,
    )

    thread_pitch = _find_number_after_patterns(
        text,
        [
            r"(?:thread\s*pitch)\s*(?:is|=|:|of)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch|inches)?",
        ],
        pitch,
    )

    if outer_diameter <= inner_diameter:
        outer_diameter = inner_diameter + 10

    if diameter <= 0:
        diameter = outer_diameter

    radius = max(radius, diameter / 2 if diameter else 1)

    return {
        "outerDiameter": outer_diameter,
        "innerDiameter": inner_diameter,
        "boreDiameter": inner_diameter,
        "diameter": diameter,
        "radius": radius,
        "length": length,
        "thickness": thickness,
        "width": width,
        "depth": depth,
        "height": height,
        "wallThickness": wall_thickness,
        "holeCount": max(int(hole_count), 0),
        "boltHoleCount": max(int(hole_count), 0),
        "toothCount": max(int(tooth_count), 6),
        "finCount": max(int(fin_count), 2),
        "segmentCount": max(int(segment_count), 8),
        "pressureAngle": pressure_angle,
        "helixAngle": helix_angle,
        "chamfer": chamfer,
        "fillet": fillet,
        "pitch": pitch,
        "threadPitch": thread_pitch,
    }


INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("robot", ("robot", "robotic", "urdf", "xacro", "ros", "servo", "actuator", "joint", "link", "robot arm", "mobile robot")),
    ("drone_frame", ("drone frame", "quadcopter", "quad copter", "uav frame")),
    ("heat_sink", ("heat sink", "heatsink", "heat-sink", "cooling fin", "cooler")),
    ("helical_gear", ("helical gear", "helix gear", "right-handed gear", "left-handed gear", "right hand gear", "left hand gear")),
    ("gear", ("gear", "spur gear", "teeth", "tooth")),
    ("rack_gear", ("gear rack", "rack gear", "linear gear rack")),
    ("flange", ("flange", "flanged")),
    ("computer_mouse", ("computer mouse", "wireless mouse", "mouse shell", "ergonomic mouse")),
    ("cup", ("cup", "mug", "tumbler", "coffee cup", "handle cup")),
    ("bushing", ("bushing", "bush", "sleeve", "bearing sleeve")),
    ("hollow_cylinder", ("pipe", "tube", "hollow cylinder", "hollow tube")),
    ("bracket", ("bracket", "angle bracket", "l bracket", "support bracket")),
    ("phone_stand", ("phone stand", "tablet stand", "stand for phone")),
    ("plate", ("plate", "base plate", "mounting plate", "flat plate")),
    ("pulley", ("pulley", "belt wheel")),
    ("box", ("box", "enclosure", "case", "container", "housing")),
    ("lid", ("lid", "cover", "cap")),
    ("shaft", ("shaft", "rod", "pin", "dowel")),
    ("sphere", ("sphere", "ball")),
    ("cone", ("cone", "funnel")),
    ("washer", ("washer", "ring")),
    ("bearing", ("bearing", "ball bearing")),
    ("bolt", ("bolt", "screw", "threaded bolt", "machine screw")),
    ("nut", ("nut", "hex nut", "threaded nut")),
    ("spring", ("spring", "coil", "helical spring")),
    ("knob", ("knob", "dial", "thumb wheel")),
    ("wheel", ("wheel", "tire", "tyre")),
    ("fan", ("fan blade", "fan", "impeller", "propeller")),
    ("clamp", ("clamp", "clip", "pipe clamp")),
    ("hinge", ("hinge", "door hinge")),
    ("nozzle", ("nozzle", "adapter nozzle", "hose nozzle")),
]


def infer_cad_intent(prompt: str) -> str:
    text = _clean_text(prompt)

    if (
        "cylinder" in text
        and _contains_any(text, ["inner", "inside", "bore", "hole", "id"])
        and _contains_any(text, ["outer", "outside", "od"])
    ):
        return "hollow_cylinder"

    for intent, keywords in INTENT_KEYWORDS:
        if _contains_any(text, keywords):
            return intent

    if "cylinder" in text:
        return "cylinder"

    return "generic"


def _param_group_from_key(key: str) -> str:
    clean = key.lower()
    if any(word in clean for word in ["count", "teeth", "fin", "holecount", "segments"]):
        return "Pattern"
    if any(word in clean for word in ["hole", "bore", "inner"]):
        return "Holes"
    if any(word in clean for word in ["angle", "rotation", "shoulder", "elbow", "wrist", "joint"]):
        return "Angles"
    if any(word in clean for word in ["chamfer", "fillet", "round"]):
        return "Finishing"
    return "Dimensions"


def make_param(
    key: str,
    label: str,
    value: Number | str | bool,
    *,
    min_value: Number | None = None,
    max_value: Number | None = None,
    step: Number | None = 1,
    unit: str | None = "mm",
    param_type: str = "number",
    description: str | None = None,
    group: str | None = None,
    control: str | None = None,
    options: list[Any] | None = None,
) -> dict[str, Any]:
    if param_type == "number" and isinstance(value, (int, float)):
        if min_value is None:
            min_value = 0
        if max_value is None:
            max_value = max(abs(float(value)) * 3, 100)
        if min_value >= max_value:
            max_value = min_value + max(step or 1, 1)

    item: dict[str, Any] = {
        "key": key,
        "label": label,
        "value": value,
        "defaultValue": value,
        "min": min_value,
        "max": max_value,
        "step": step,
        "unit": unit,
        "type": param_type,
        "description": description or label,
        "locked": False,
        "group": group or _param_group_from_key(key),
        "control": control or ("slider" if param_type == "number" else param_type),
    }

    if options is not None:
        item["options"] = options

    return item


def make_feature(
    feature_id: str,
    name: str,
    feature_type: str,
    *,
    original_type: str | None = None,
    operation: str | None = None,
    children: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": feature_id,
        "name": name,
        "type": feature_type,
        "originalType": original_type or feature_type,
        "operation": operation or feature_type,
        "visible": True,
        "locked": False,
        "children": children or [],
    }

    if metadata:
        item["metadata"] = metadata

    return item


def _base_planes() -> list[dict[str, Any]]:
    return [
        make_feature("origin", "Origin", "origin", original_type="origin", operation="reference"),
        make_feature("xy", "XY Plane", "plane", original_type="plane", operation="reference"),
        make_feature("xz", "XZ Plane", "plane", original_type="plane", operation="reference"),
        make_feature("yz", "YZ Plane", "plane", original_type="plane", operation="reference"),
    ]


def _package(
    *,
    title: str,
    description: str,
    code: str,
    params: list[dict[str, Any]],
    features: list[dict[str, Any]],
    message: str,
    intent: str,
    robot_description: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package: dict[str, Any] = {
        "title": title,
        "description": description,
        "code": code,
        "params": params,
        "features": _base_planes() + features,
        "message": message,
        "intent": intent,
    }

    if robot_description is not None:
        package["robotDescription"] = robot_description
        package["robot_description"] = robot_description

    return package


def _finish_features(chamfer: float = 0, fillet: float = 0) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    if chamfer > 0:
        features.append(make_feature("edge-chamfer", "Edge Chamfer", "chamfer", original_type="chamfer", operation="finish"))
    if fillet > 0:
        features.append(make_feature("edge-fillet", "Edge Fillet", "fillet", original_type="fillet", operation="finish"))
    return features


def _build_hollow_cylinder(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], dims["innerDiameter"] + 2)
    inner = min(dims["innerDiameter"], outer - 1)
    length = dims["length"]
    chamfer = dims["chamfer"]

    return _package(
        title="Hollow Cylinder",
        description="Parametric hollow cylinder generated from inner and outer diameter values.",
        code=f"""// TextoCAD parametric hollow cylinder
const outerDiameter = {outer};
const innerDiameter = {inner};
const length = {length};
const chamfer = {chamfer};

const outerBody = Cylinder({{ radius: outerDiameter / 2, height: length }});
const innerCut = Cylinder({{ radius: innerDiameter / 2, height: length + 2 }});
let model = outerBody.subtract(innerCut);
if (chamfer > 0) model = model.chamfer(chamfer);
return model;
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=inner + 1, max_value=max(outer * 3, 100), description="Outside diameter of the hollow cylinder."),
            make_param("innerDiameter", "Inner Diameter", inner, min_value=1, max_value=outer - 1, description="Inside bore diameter of the hollow cylinder."),
            make_param("length", "Length", length, min_value=5, max_value=max(length * 3, 200), description="Cylinder length."),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=10, step=0.25, description="Edge chamfer size."),
        ],
        features=[
            make_feature("outer-cylinder", "Outer Cylinder", "cylinder", original_type="cylinder", operation="add", metadata={"outerDiameter": outer, "length": length}),
            make_feature("inner-bore", "Inner Bore Cut", "hole", original_type="cylinder", operation="subtract", metadata={"innerDiameter": inner}),
            make_feature("boolean-difference", "Boolean Difference", "boolean", original_type="subtract", operation="subtract"),
            *_finish_features(chamfer=chamfer),
        ],
        message="Hollow cylinder CAD package generated successfully.",
        intent="hollow_cylinder",
    )


def _build_flange(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], 80)
    bore = min(dims["boreDiameter"], outer - 20)
    thickness = max(dims["thickness"], 6)
    holes = max(int(dims["boltHoleCount"]), 4)
    bolt_hole = max(outer * 0.08, 6)
    pcd = max(min(outer * 0.72, outer - bolt_hole - 4), bore + bolt_hole + 8)
    chamfer = dims["chamfer"]

    return _package(
        title="Bolt Flange",
        description=f"Round flange with central bore and {holes} bolt holes.",
        code=f"""// TextoCAD parametric bolt flange
const flangeDiameter = {outer};
const boreDiameter = {bore};
const flangeThickness = {thickness};
const boltHoleCount = {holes};
const boltCircleDiameter = {pcd};
const boltHoleDiameter = {bolt_hole};
const chamfer = {chamfer};

let model = Cylinder({{ radius: flangeDiameter / 2, height: flangeThickness }});
model = model.subtract(Cylinder({{ radius: boreDiameter / 2, height: flangeThickness + 2 }}));
for (let i = 0; i < boltHoleCount; i++) {{
  const angle = (Math.PI * 2 * i) / boltHoleCount;
  const x = Math.cos(angle) * boltCircleDiameter / 2;
  const y = Math.sin(angle) * boltCircleDiameter / 2;
  model = model.subtract(Cylinder({{ radius: boltHoleDiameter / 2, height: flangeThickness + 2 }}).translate([x, y, 0]));
}}
if (chamfer > 0) model = model.chamfer(chamfer);
return model;
""",
        params=[
            make_param("flangeDiameter", "Flange Diameter", outer, min_value=bore + 20, max_value=max(outer * 3, 180)),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=5, max_value=outer - 20),
            make_param("flangeThickness", "Flange Thickness", thickness, min_value=3, max_value=60),
            make_param("boltHoleCount", "Bolt Hole Count", holes, min_value=0, max_value=24, step=1, unit="", param_type="number"),
            make_param("boltCircleDiameter", "Bolt Circle Diameter", pcd, min_value=bore + 10, max_value=outer - 8),
            make_param("boltHoleDiameter", "Bolt Hole Diameter", bolt_hole, min_value=2, max_value=outer * 0.2),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=8, step=0.25),
        ],
        features=[
            make_feature("flange-body", "Flange Body", "cylinder", original_type="cylinder", operation="add"),
            make_feature("center-bore", "Center Bore", "hole", original_type="cylinder", operation="subtract"),
            make_feature("bolt-hole-pattern", "Bolt Hole Pattern", "pattern", original_type="circular-pattern", operation="pattern"),
            make_feature("bolt-hole-cuts", "Bolt Hole Cuts", "hole", original_type="cylinder", operation="subtract"),
            *_finish_features(chamfer=chamfer),
        ],
        message="Flange CAD package generated successfully.",
        intent="flange",
    )


def _build_gear(clean_prompt: str, dims: dict[str, float], *, helical: bool = False) -> dict[str, Any]:
    teeth = max(int(dims["toothCount"]), 6)
    outer = max(dims["outerDiameter"], 50)
    bore = min(dims["boreDiameter"], outer - 4)
    face_width = max(dims["thickness"], 6)
    pressure_angle = dims["pressureAngle"]
    helix_angle = dims["helixAngle"] if helical else 0
    module = outer / max(teeth + 2, 1)
    chamfer = dims["chamfer"]

    title = "Right-Handed Helical Gear" if helical else "Spur Gear"
    gear_type = "helical_gear" if helical else "gear"
    twist_feature = make_feature("helical-twist", "Helical Twist", "feature", original_type="helix", operation="twist") if helical else make_feature("straight-teeth", "Straight Teeth", "feature", original_type="spur", operation="tooth-profile")

    return _package(
        title=title,
        description=f"Parametric {'helical' if helical else 'spur'} gear with {teeth} teeth and central bore.",
        code=f"""// TextoCAD parametric {'helical gear' if helical else 'spur gear'} approximation
const outerDiameter = {outer};
const rootDiameter = outerDiameter * 0.82;
const boreDiameter = {bore};
const faceWidth = {face_width};
const toothCount = {teeth};
const module = {module};
const pressureAngle = {pressure_angle};
const helixAngle = {helix_angle};
const chamfer = {chamfer};

const gearBlank = Gear({{
  outerRadius: outerDiameter / 2,
  rootRadius: rootDiameter / 2,
  toothCount,
  module,
  thickness: faceWidth,
  pressureAngle,
  helixAngle,
  handedness: "right"
}});
let gear = gearBlank.subtract(Cylinder({{ radius: boreDiameter / 2, height: faceWidth + 2 }}));
if (chamfer > 0) gear = gear.chamfer(chamfer);
return gear;
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=bore + 2, max_value=max(outer * 3, 200)),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=1, max_value=outer - 1),
            make_param("faceWidth", "Face Width", face_width, min_value=1, max_value=max(face_width * 4, 80)),
            make_param("toothCount", "Tooth Count", teeth, min_value=6, max_value=180, step=1, unit="", param_type="number"),
            make_param("module", "Module", round(module, 3), min_value=0.25, max_value=max(module * 4, 10), step=0.25),
            make_param("pressureAngle", "Pressure Angle", pressure_angle, min_value=10, max_value=35, step=0.5, unit="deg"),
            make_param("helixAngle", "Helix Angle", helix_angle, min_value=0, max_value=45, step=0.5, unit="deg"),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=8, step=0.25),
        ],
        features=[
            make_feature("gear-body", title, gear_type, original_type=gear_type, operation="add", metadata={"outerDiameter": outer, "boreDiameter": bore, "toothCount": teeth, "faceWidth": face_width, "helixAngle": helix_angle}),
            twist_feature,
            make_feature("center-bore", "Center Bore", "hole", original_type="cylinder", operation="subtract"),
            *_finish_features(chamfer=chamfer),
        ],
        message=f"{title} CAD package generated successfully.",
        intent="helical_gear" if helical else "gear",
    )


def _build_rack_gear(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    length = max(dims["length"], 100)
    width = max(dims["width"], 20)
    height = max(dims["thickness"], 10)
    teeth = max(int(dims["toothCount"]), 12)
    tooth_height = max(height * 0.35, 3)

    return _package(
        title="Linear Gear Rack",
        description=f"Parametric straight gear rack with {teeth} teeth.",
        code=f"""// TextoCAD parametric gear rack
const length = {length};
const width = {width};
const baseHeight = {height};
const toothCount = {teeth};
const toothHeight = {tooth_height};

let rack = Box({{ width: length, depth: width, height: baseHeight }});
for (let i = 0; i < toothCount; i++) {{
  const x = -length / 2 + (i + 0.5) * length / toothCount;
  rack = rack.union(TriangularPrism({{ width: length / toothCount, depth: width, height: toothHeight }}).translate([x, 0, baseHeight / 2 + toothHeight / 2]));
}}
return rack;
""",
        params=[
            make_param("length", "Length", length, min_value=30, max_value=max(length * 3, 500)),
            make_param("width", "Width", width, min_value=5, max_value=100),
            make_param("baseHeight", "Base Height", height, min_value=4, max_value=80),
            make_param("toothCount", "Tooth Count", teeth, min_value=4, max_value=100, step=1, unit=""),
            make_param("toothHeight", "Tooth Height", tooth_height, min_value=1, max_value=30),
        ],
        features=[
            make_feature("rack-base", "Rack Base", "box", original_type="box", operation="add"),
            make_feature("rack-teeth", "Rack Teeth", "pattern", original_type="linear-pattern", operation="add"),
        ],
        message="Gear rack CAD package generated successfully.",
        intent="rack_gear",
    )


def _build_heat_sink(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 60)
    depth = max(dims["depth"], 40)
    base_thickness = max(min(dims["thickness"], 12), 4)
    fin_count = max(int(dims["finCount"]), 4)
    fin_height = max(dims["height"] * 2.5, 25)
    fin_thickness = max(width / (fin_count * 3), 1.5)
    spacing = max((width - fin_count * fin_thickness) / max(fin_count + 1, 1), 1)

    return _package(
        title="Heat Sink",
        description=f"Parametric heat sink with base plate and {fin_count} cooling fins.",
        code=f"""// TextoCAD parametric heat sink
const baseWidth = {width};
const baseDepth = {depth};
const baseThickness = {base_thickness};
const finCount = {fin_count};
const finHeight = {fin_height};
const finThickness = {fin_thickness};
const finSpacing = {spacing};

let model = Box({{ width: baseWidth, depth: baseDepth, height: baseThickness }});
for (let i = 0; i < finCount; i++) {{
  const x = -baseWidth / 2 + finSpacing + finThickness / 2 + i * (finThickness + finSpacing);
  model = model.union(Box({{ width: finThickness, depth: baseDepth, height: finHeight }}).translate([x, 0, baseThickness / 2 + finHeight / 2]));
}}
return model;
""",
        params=[
            make_param("baseWidth", "Base Width", width, min_value=20, max_value=max(width * 3, 300)),
            make_param("baseDepth", "Base Depth", depth, min_value=20, max_value=max(depth * 3, 250)),
            make_param("baseThickness", "Base Thickness", base_thickness, min_value=2, max_value=30),
            make_param("finCount", "Fin Count", fin_count, min_value=2, max_value=60, step=1, unit=""),
            make_param("finHeight", "Fin Height", fin_height, min_value=5, max_value=150),
            make_param("finThickness", "Fin Thickness", fin_thickness, min_value=0.8, max_value=15, step=0.1),
        ],
        features=[
            make_feature("base-plate", "Base Plate", "box", original_type="box", operation="add"),
            make_feature("cooling-fin-pattern", "Cooling Fin Pattern", "pattern", original_type="linear-pattern", operation="pattern"),
            make_feature("cooling-fins", "Cooling Fins", "box", original_type="box", operation="add"),
        ],
        message="Heat sink CAD package generated successfully.",
        intent="heat_sink",
    )


def _build_plate(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 30)
    depth = max(dims["depth"], 30)
    thickness = max(dims["thickness"], 2)
    holes = max(int(dims["holeCount"]), 0)
    hole_diameter = max(min(width, depth) * 0.09, 5)
    chamfer = dims["chamfer"]

    return _package(
        title="Mounting Plate",
        description=f"Rectangular mounting plate with {holes} holes.",
        code=f"""// TextoCAD parametric mounting plate
const width = {width};
const depth = {depth};
const thickness = {thickness};
const holeCount = {holes};
const holeDiameter = {hole_diameter};
const chamfer = {chamfer};

let model = Box({{ width, depth, height: thickness }});
const positions = [
  [-width * 0.35, -depth * 0.35, 0],
  [ width * 0.35, -depth * 0.35, 0],
  [ width * 0.35,  depth * 0.35, 0],
  [-width * 0.35,  depth * 0.35, 0]
];
for (const position of positions.slice(0, holeCount)) {{
  model = model.subtract(Cylinder({{ radius: holeDiameter / 2, height: thickness + 2 }}).translate(position));
}}
if (chamfer > 0) model = model.chamfer(chamfer);
return model;
""",
        params=[
            make_param("width", "Width", width, min_value=10, max_value=max(width * 3, 300)),
            make_param("depth", "Depth", depth, min_value=10, max_value=max(depth * 3, 300)),
            make_param("thickness", "Thickness", thickness, min_value=1, max_value=60),
            make_param("holeCount", "Hole Count", holes, min_value=0, max_value=12, step=1, unit=""),
            make_param("holeDiameter", "Hole Diameter", hole_diameter, min_value=1, max_value=min(width, depth) / 2),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=8, step=0.25),
        ],
        features=[
            make_feature("plate-body", "Plate Body", "box", original_type="box", operation="add"),
            make_feature("mounting-holes", "Mounting Holes", "hole", original_type="cylinder", operation="subtract"),
            *_finish_features(chamfer=chamfer),
        ],
        message="Mounting plate CAD package generated successfully.",
        intent="plate",
    )


def _build_bracket(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 50)
    depth = max(dims["depth"], 40)
    height = max(dims["height"], dims["length"], 50)
    thickness = max(dims["thickness"], 4)
    hole_diameter = max(min(width, depth) * 0.12, 6)
    fillet = dims["fillet"]

    return _package(
        title="L Bracket",
        description="Parametric L bracket with base plate, vertical web, and mounting holes.",
        code=f"""// TextoCAD parametric L bracket
const width = {width};
const depth = {depth};
const height = {height};
const thickness = {thickness};
const holeDiameter = {hole_diameter};
const fillet = {fillet};

const basePlate = Box({{ width, depth, height: thickness }});
const verticalWeb = Box({{ width, depth: thickness, height }}).translate([0, -depth / 2 + thickness / 2, height / 2]);
let bracket = basePlate.union(verticalWeb);
if (fillet > 0) bracket = bracket.fillet(fillet);
return bracket;
""",
        params=[
            make_param("width", "Base Width", width, min_value=20, max_value=max(width * 3, 300)),
            make_param("depth", "Base Depth", depth, min_value=20, max_value=max(depth * 3, 300)),
            make_param("height", "Bracket Height", height, min_value=20, max_value=max(height * 3, 300)),
            make_param("thickness", "Wall Thickness", thickness, min_value=2, max_value=40),
            make_param("holeDiameter", "Hole Diameter", hole_diameter, min_value=2, max_value=35),
            make_param("fillet", "Inside Fillet", fillet, min_value=0, max_value=20, step=0.25),
        ],
        features=[
            make_feature("base-plate", "Base Plate", "box", original_type="box", operation="add"),
            make_feature("vertical-web", "Vertical Web", "box", original_type="box", operation="add"),
            make_feature("bracket-union", "Bracket Union", "boolean", original_type="union", operation="union"),
            make_feature("mounting-holes", "Mounting Holes", "hole", original_type="cylinder", operation="subtract"),
            *_finish_features(fillet=fillet),
        ],
        message="Bracket CAD package generated successfully.",
        intent="bracket",
    )


def _build_box(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 20)
    depth = max(dims["depth"], 20)
    height = max(dims["height"], dims["length"], 20)
    wall = max(min(dims["wallThickness"], min(width, depth) / 4), 1)
    open_top = not _contains_any(clean_prompt, ["closed", "solid", "filled"])

    code = f"""// TextoCAD parametric {'open box enclosure' if open_top else 'box'}
const width = {width};
const depth = {depth};
const height = {height};
const wallThickness = {wall};
const openTop = {str(open_top).lower()};
const model = openTop
  ? OpenBox({{ width, depth, height, wallThickness, bottomThickness: wallThickness }})
  : Box({{ width, depth, height }});
return model;
"""

    return _package(
        title="Open Box Enclosure" if open_top else "Rectangular Box",
        description="Parametric rectangular enclosure with editable wall thickness." if open_top else "Parametric rectangular box.",
        code=code,
        params=[
            make_param("width", "Width", width, min_value=10, max_value=max(width * 3, 300)),
            make_param("depth", "Depth", depth, min_value=10, max_value=max(depth * 3, 300)),
            make_param("height", "Height", height, min_value=5, max_value=max(height * 3, 300)),
            make_param("wallThickness", "Wall Thickness", wall, min_value=1, max_value=max(min(width, depth) / 3, 2)),
            make_param("openTop", "Open Top", open_top, unit="", param_type="boolean", group="Options", control="checkbox"),
        ],
        features=[
            make_feature("box-body", "Box Body", "box", original_type="box", operation="add"),
            make_feature("shell-open-top", "Shell/Open Top", "feature", original_type="shell", operation="shell") if open_top else make_feature("solid-box", "Solid Box", "box", original_type="box", operation="add"),
        ],
        message="Box CAD package generated successfully.",
        intent="box",
    )


def _build_lid(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 90)
    depth = max(dims["depth"], 60)
    thickness = max(dims["thickness"], 3)
    lip = max(dims["wallThickness"], 2)

    return _package(
        title="Snap-On Lid",
        description="Parametric rectangular lid with inner locating lip.",
        code=f"""// TextoCAD parametric snap-on lid
const width = {width};
const depth = {depth};
const thickness = {thickness};
const lipHeight = {lip};
const lipInset = {lip};

const top = Box({{ width, depth, height: thickness }});
const lip = OpenBox({{ width: width - lipInset * 2, depth: depth - lipInset * 2, height: lipHeight, wallThickness: lipInset }}).translate([0, 0, -thickness / 2]);
return top.union(lip);
""",
        params=[
            make_param("width", "Width", width, min_value=20, max_value=300),
            make_param("depth", "Depth", depth, min_value=20, max_value=300),
            make_param("thickness", "Top Thickness", thickness, min_value=1, max_value=20),
            make_param("lipHeight", "Lip Height", lip, min_value=1, max_value=30),
            make_param("lipInset", "Lip Inset", lip, min_value=1, max_value=20),
        ],
        features=[
            make_feature("lid-top", "Lid Top", "box", original_type="box", operation="add"),
            make_feature("inner-lip", "Inner Lip", "feature", original_type="open-box", operation="add"),
        ],
        message="Lid CAD package generated successfully.",
        intent="lid",
    )


def _build_pulley(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], dims["diameter"], 40)
    bore = min(dims["boreDiameter"], outer - 4)
    width = max(dims["thickness"], 10)
    groove_depth = max(min(outer * 0.07, 8), 2)

    return _package(
        title="Pulley",
        description="Parametric pulley with central bore and belt groove.",
        code=f"""// TextoCAD parametric pulley approximation
const pulleyDiameter = {outer};
const boreDiameter = {bore};
const width = {width};
const grooveDepth = {groove_depth};

const pulleyBody = Pulley({{ radius: pulleyDiameter / 2, width, grooveDepth }});
const bore = Cylinder({{ radius: boreDiameter / 2, height: width + 2 }});
return pulleyBody.subtract(bore);
""",
        params=[
            make_param("pulleyDiameter", "Pulley Diameter", outer, min_value=bore + 4, max_value=max(outer * 3, 200)),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=1, max_value=outer - 2),
            make_param("width", "Pulley Width", width, min_value=3, max_value=80),
            make_param("grooveDepth", "Groove Depth", groove_depth, min_value=0, max_value=20),
        ],
        features=[
            make_feature("pulley-profile", "Pulley Profile", "revolve", original_type="pulley", operation="add"),
            make_feature("center-bore", "Center Bore", "hole", original_type="cylinder", operation="subtract"),
            make_feature("belt-groove", "Belt Groove", "feature", original_type="groove", operation="cut"),
        ],
        message="Pulley CAD package generated successfully.",
        intent="pulley",
    )


def _build_computer_mouse(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    length = max(dims["length"], 110)
    width = max(dims["width"], 60)
    height = max(dims["height"], 28)
    wheel_radius = 5

    return _package(
        title="Wireless Computer Mouse",
        description="Organic-style computer mouse approximation with shell, buttons, and scroll wheel.",
        code=f"""// TextoCAD parametric wireless computer mouse approximation
const length = {length};
const width = {width};
const height = {height};
const wheelRadius = {wheel_radius};

const shell = SmoothEllipsoid({{ length, width, height }});
const leftButton = MouseButton({{ side: "left", length: length * 0.45, width: width * 0.45, height: 2 }});
const rightButton = MouseButton({{ side: "right", length: length * 0.45, width: width * 0.45, height: 2 }});
const wheel = Cylinder({{ radius: wheelRadius, height: width * 0.12 }}).rotate([Math.PI / 2, 0, 0]).translate([0, 0, height * 0.42]);
return shell.union(leftButton).union(rightButton).union(wheel);
""",
        params=[
            make_param("length", "Length", length, min_value=70, max_value=180),
            make_param("width", "Width", width, min_value=35, max_value=100),
            make_param("height", "Height", height, min_value=15, max_value=60),
            make_param("wheelRadius", "Wheel Radius", wheel_radius, min_value=2, max_value=12, step=0.25),
        ],
        features=[
            make_feature("mouse-shell", "Smooth Mouse Shell", "feature", original_type="ellipsoid", operation="add"),
            make_feature("left-button", "Left Button", "feature", original_type="button", operation="add"),
            make_feature("right-button", "Right Button", "feature", original_type="button", operation="add"),
            make_feature("scroll-wheel", "Scroll Wheel", "cylinder", original_type="cylinder", operation="add"),
        ],
        message="Wireless mouse CAD package generated successfully.",
        intent="computer_mouse",
    )


def _build_cup(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["outerDiameter"], dims["diameter"], 60)
    height = max(dims["height"], dims["length"], 80)
    wall = max(min(dims["wallThickness"], 6), 1.2)
    handle_radius = diameter * 0.35

    return _package(
        title="Cup With Handle",
        description="Parametric cup with hollow body and side handle.",
        code=f"""// TextoCAD parametric cup approximation
const outerDiameter = {diameter};
const height = {height};
const wallThickness = {wall};
const handleRadius = {handle_radius};

const cupBody = TaperedCup({{ topRadius: outerDiameter / 2, bottomRadius: outerDiameter * 0.38, height, wallThickness }});
const handle = TorusSegment({{ majorRadius: handleRadius, tubeRadius: wallThickness * 1.4, angleDegrees: 220 }}).translate([outerDiameter / 2, 0, height * 0.52]);
return cupBody.union(handle);
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", diameter, min_value=30, max_value=180),
            make_param("height", "Height", height, min_value=30, max_value=220),
            make_param("wallThickness", "Wall Thickness", wall, min_value=1, max_value=10, step=0.1),
            make_param("handleRadius", "Handle Radius", handle_radius, min_value=8, max_value=100),
        ],
        features=[
            make_feature("cup-profile", "Cup Profile", "revolve", original_type="tapered-cup", operation="add"),
            make_feature("hollow-interior", "Hollow Interior", "boolean", original_type="shell", operation="shell"),
            make_feature("side-handle", "Side Handle", "torus", original_type="torus-segment", operation="add"),
        ],
        message="Cup CAD package generated successfully.",
        intent="cup",
    )


def _build_washer(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], 20)
    inner = min(max(dims["innerDiameter"], 3), outer - 2)
    thickness = max(dims["thickness"], 1)

    return _package(
        title="Washer Ring",
        description="Flat washer/ring with central hole.",
        code=f"""// TextoCAD parametric washer
const outerDiameter = {outer};
const innerDiameter = {inner};
const thickness = {thickness};
return Cylinder({{ radius: outerDiameter / 2, height: thickness }}).subtract(Cylinder({{ radius: innerDiameter / 2, height: thickness + 2 }}));
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=inner + 1, max_value=max(outer * 3, 120)),
            make_param("innerDiameter", "Inner Diameter", inner, min_value=1, max_value=outer - 1),
            make_param("thickness", "Thickness", thickness, min_value=0.5, max_value=25),
        ],
        features=[
            make_feature("washer-body", "Washer Body", "cylinder", original_type="cylinder", operation="add"),
            make_feature("center-hole", "Center Hole", "hole", original_type="cylinder", operation="subtract"),
        ],
        message="Washer CAD package generated successfully.",
        intent="washer",
    )


def _build_solid_cylinder(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 2)
    length = max(dims["length"], 2)
    chamfer = dims["chamfer"]

    return _package(
        title="Solid Cylinder",
        description="Simple parametric solid cylinder.",
        code=f"""// TextoCAD parametric solid cylinder
const diameter = {diameter};
const length = {length};
const chamfer = {chamfer};
let cylinder = Cylinder({{ radius: diameter / 2, height: length }});
if (chamfer > 0) cylinder = cylinder.chamfer(chamfer);
return cylinder;
""",
        params=[
            make_param("diameter", "Diameter", diameter, min_value=1, max_value=max(diameter * 3, 150)),
            make_param("length", "Length", length, min_value=1, max_value=max(length * 3, 250)),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=8, step=0.25),
        ],
        features=[
            make_feature("main-cylinder", "Main Cylinder", "cylinder", original_type="cylinder", operation="add"),
            *_finish_features(chamfer=chamfer),
        ],
        message="Cylinder CAD package generated successfully.",
        intent="cylinder",
    )


def _build_sphere(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 10)
    return _package(
        title="Sphere",
        description="Parametric sphere/ball.",
        code=f"""// TextoCAD parametric sphere
const diameter = {diameter};
return Sphere({{ radius: diameter / 2 }});
""",
        params=[make_param("diameter", "Diameter", diameter, min_value=1, max_value=max(diameter * 3, 200))],
        features=[make_feature("sphere-body", "Sphere Body", "sphere", original_type="sphere", operation="add")],
        message="Sphere CAD package generated successfully.",
        intent="sphere",
    )


def _build_cone(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 20)
    height = max(dims["height"], dims["length"], 30)
    top_diameter = 0 if "cone" in clean_prompt else diameter * 0.25
    return _package(
        title="Cone",
        description="Parametric cone/frustum.",
        code=f"""// TextoCAD parametric cone/frustum
const bottomDiameter = {diameter};
const topDiameter = {top_diameter};
const height = {height};
return Cone({{ bottomRadius: bottomDiameter / 2, topRadius: topDiameter / 2, height }});
""",
        params=[
            make_param("bottomDiameter", "Bottom Diameter", diameter, min_value=1, max_value=max(diameter * 3, 200)),
            make_param("topDiameter", "Top Diameter", top_diameter, min_value=0, max_value=diameter),
            make_param("height", "Height", height, min_value=1, max_value=max(height * 3, 250)),
        ],
        features=[make_feature("cone-body", "Cone Body", "cone", original_type="cone", operation="add")],
        message="Cone CAD package generated successfully.",
        intent="cone",
    )


def _build_bolt(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    shaft_diameter = max(dims["diameter"], 6)
    length = max(dims["length"], 30)
    head_diameter = max(shaft_diameter * 1.7, 10)
    head_height = max(shaft_diameter * 0.6, 4)
    pitch = max(dims["threadPitch"], 0.5)

    return _package(
        title="Threaded Bolt",
        description="Parametric bolt with hex head and simplified thread representation.",
        code=f"""// TextoCAD parametric bolt approximation
const shaftDiameter = {shaft_diameter};
const length = {length};
const headDiameter = {head_diameter};
const headHeight = {head_height};
const threadPitch = {pitch};
const shaft = Cylinder({{ radius: shaftDiameter / 2, height: length }});
const head = HexPrism({{ radius: headDiameter / 2, height: headHeight }}).translate([0, 0, length / 2 + headHeight / 2]);
const thread = Thread({{ diameter: shaftDiameter, length, pitch: threadPitch }});
return shaft.union(head).union(thread);
""",
        params=[
            make_param("shaftDiameter", "Shaft Diameter", shaft_diameter, min_value=2, max_value=40),
            make_param("length", "Length", length, min_value=5, max_value=200),
            make_param("headDiameter", "Head Diameter", head_diameter, min_value=shaft_diameter, max_value=80),
            make_param("headHeight", "Head Height", head_height, min_value=2, max_value=40),
            make_param("threadPitch", "Thread Pitch", pitch, min_value=0.25, max_value=5, step=0.25),
        ],
        features=[
            make_feature("bolt-shaft", "Bolt Shaft", "cylinder", original_type="cylinder", operation="add"),
            make_feature("hex-head", "Hex Head", "feature", original_type="hex-prism", operation="add"),
            make_feature("thread", "Thread", "feature", original_type="thread", operation="add"),
        ],
        message="Bolt CAD package generated successfully.",
        intent="bolt",
    )


def _build_nut(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    bore = max(dims["boreDiameter"], 6)
    outer = max(dims["outerDiameter"], bore * 1.9)
    thickness = max(dims["thickness"], bore * 0.8)

    return _package(
        title="Hex Nut",
        description="Parametric hex nut with central threaded bore.",
        code=f"""// TextoCAD parametric hex nut approximation
const outerDiameter = {outer};
const boreDiameter = {bore};
const thickness = {thickness};
const body = HexPrism({{ radius: outerDiameter / 2, height: thickness }});
const bore = Cylinder({{ radius: boreDiameter / 2, height: thickness + 2 }});
return body.subtract(bore);
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=bore + 2, max_value=100),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=1, max_value=outer - 1),
            make_param("thickness", "Thickness", thickness, min_value=2, max_value=60),
        ],
        features=[
            make_feature("hex-body", "Hex Body", "feature", original_type="hex-prism", operation="add"),
            make_feature("threaded-bore", "Threaded Bore", "hole", original_type="cylinder", operation="subtract"),
        ],
        message="Nut CAD package generated successfully.",
        intent="nut",
    )


def _build_spring(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 30)
    wire_diameter = max(dims["thickness"], 2)
    length = max(dims["length"], 80)
    turns = max(int(dims["segmentCount"] / 4), 4)

    return _package(
        title="Helical Spring",
        description="Parametric helical compression spring approximation.",
        code=f"""// TextoCAD parametric helical spring
const coilDiameter = {diameter};
const wireDiameter = {wire_diameter};
const length = {length};
const turns = {turns};
return HelicalSpring({{ coilRadius: coilDiameter / 2, wireRadius: wireDiameter / 2, length, turns }});
""",
        params=[
            make_param("coilDiameter", "Coil Diameter", diameter, min_value=5, max_value=150),
            make_param("wireDiameter", "Wire Diameter", wire_diameter, min_value=0.5, max_value=20),
            make_param("length", "Length", length, min_value=10, max_value=300),
            make_param("turns", "Turns", turns, min_value=1, max_value=40, step=1, unit=""),
        ],
        features=[make_feature("spring-coil", "Spring Coil", "feature", original_type="helix", operation="sweep")],
        message="Spring CAD package generated successfully.",
        intent="spring",
    )


def _build_knob(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 35)
    height = max(dims["height"], dims["thickness"], 18)
    bore = min(max(dims["boreDiameter"], 6), diameter - 4)
    ridges = max(int(dims["segmentCount"] / 3), 12)

    return _package(
        title="Knurled Knob",
        description="Parametric knob with central bore and grip ridges.",
        code=f"""// TextoCAD parametric knob
const diameter = {diameter};
const height = {height};
const boreDiameter = {bore};
const ridgeCount = {ridges};
let knob = Cylinder({{ radius: diameter / 2, height }});
knob = knob.addRadialRidges({{ count: ridgeCount, depth: diameter * 0.04 }});
knob = knob.subtract(Cylinder({{ radius: boreDiameter / 2, height: height + 2 }}));
return knob;
""",
        params=[
            make_param("diameter", "Diameter", diameter, min_value=bore + 4, max_value=120),
            make_param("height", "Height", height, min_value=5, max_value=80),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=1, max_value=diameter - 2),
            make_param("ridgeCount", "Ridge Count", ridges, min_value=6, max_value=80, step=1, unit=""),
        ],
        features=[
            make_feature("knob-body", "Knob Body", "cylinder", original_type="cylinder", operation="add"),
            make_feature("grip-ridges", "Grip Ridges", "pattern", original_type="radial-pattern", operation="add"),
            make_feature("center-bore", "Center Bore", "hole", original_type="cylinder", operation="subtract"),
        ],
        message="Knob CAD package generated successfully.",
        intent="knob",
    )


def _build_wheel(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 70)
    width = max(dims["thickness"], 20)
    bore = min(max(dims["boreDiameter"], 8), diameter - 10)

    return _package(
        title="Wheel",
        description="Parametric wheel with hub and central bore.",
        code=f"""// TextoCAD parametric wheel
const diameter = {diameter};
const width = {width};
const boreDiameter = {bore};
let wheel = Cylinder({{ radius: diameter / 2, height: width }});
wheel = wheel.subtract(Cylinder({{ radius: boreDiameter / 2, height: width + 2 }}));
return wheel;
""",
        params=[
            make_param("diameter", "Diameter", diameter, min_value=bore + 10, max_value=300),
            make_param("width", "Width", width, min_value=5, max_value=120),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=1, max_value=diameter - 4),
        ],
        features=[
            make_feature("wheel-body", "Wheel Body", "cylinder", original_type="cylinder", operation="add"),
            make_feature("center-bore", "Center Bore", "hole", original_type="cylinder", operation="subtract"),
        ],
        message="Wheel CAD package generated successfully.",
        intent="wheel",
    )


def _build_fan(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    diameter = max(dims["diameter"], dims["outerDiameter"], 80)
    blades = max(int(dims["finCount"]), 3)
    hub = diameter * 0.18
    blade_pitch = max(dims["helixAngle"], 15)

    return _package(
        title="Fan / Propeller",
        description=f"Parametric fan/propeller with {blades} blades.",
        code=f"""// TextoCAD parametric fan/propeller approximation
const diameter = {diameter};
const bladeCount = {blades};
const hubDiameter = {hub};
const bladePitch = {blade_pitch};
let fan = Cylinder({{ radius: hubDiameter / 2, height: diameter * 0.08 }});
for (let i = 0; i < bladeCount; i++) {{
  fan = fan.union(PropellerBlade({{ length: diameter * 0.42, width: diameter * 0.14, pitchAngle: bladePitch }}).rotate([0, 0, i * 360 / bladeCount]));
}}
return fan;
""",
        params=[
            make_param("diameter", "Diameter", diameter, min_value=20, max_value=400),
            make_param("bladeCount", "Blade Count", blades, min_value=2, max_value=12, step=1, unit=""),
            make_param("hubDiameter", "Hub Diameter", hub, min_value=5, max_value=diameter / 2),
            make_param("bladePitch", "Blade Pitch", blade_pitch, min_value=0, max_value=60, step=1, unit="deg"),
        ],
        features=[
            make_feature("hub", "Hub", "cylinder", original_type="cylinder", operation="add"),
            make_feature("blade-pattern", "Blade Pattern", "pattern", original_type="radial-pattern", operation="add"),
        ],
        message="Fan/propeller CAD package generated successfully.",
        intent="fan",
    )


def _build_clamp(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], 50)
    bore = min(max(dims["boreDiameter"], 25), outer - 8)
    width = max(dims["thickness"], 20)
    gap = max(outer * 0.12, 5)

    return _package(
        title="Split Pipe Clamp",
        description="Parametric split clamp with bore and tightening gap.",
        code=f"""// TextoCAD parametric split pipe clamp
const outerDiameter = {outer};
const boreDiameter = {bore};
const width = {width};
const gap = {gap};
let clamp = Cylinder({{ radius: outerDiameter / 2, height: width }});
clamp = clamp.subtract(Cylinder({{ radius: boreDiameter / 2, height: width + 2 }}));
clamp = clamp.subtract(Box({{ width: gap, depth: outerDiameter, height: width + 2 }}).translate([outerDiameter / 2, 0, 0]));
return clamp;
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=bore + 4, max_value=200),
            make_param("boreDiameter", "Bore Diameter", bore, min_value=5, max_value=outer - 4),
            make_param("width", "Clamp Width", width, min_value=5, max_value=100),
            make_param("gap", "Split Gap", gap, min_value=1, max_value=30),
        ],
        features=[
            make_feature("clamp-ring", "Clamp Ring", "cylinder", original_type="cylinder", operation="add"),
            make_feature("pipe-bore", "Pipe Bore", "hole", original_type="cylinder", operation="subtract"),
            make_feature("split-gap", "Split Gap", "boolean", original_type="box-cut", operation="subtract"),
        ],
        message="Clamp CAD package generated successfully.",
        intent="clamp",
    )


def _build_hinge(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 70)
    leaf_height = max(dims["height"], dims["length"], 50)
    thickness = max(dims["thickness"], 3)
    barrel_diameter = max(thickness * 3, 10)
    holes = max(int(dims["holeCount"]), 4)

    return _package(
        title="Door Hinge",
        description="Parametric hinge with two leaves, barrel, pin, and screw holes.",
        code=f"""// TextoCAD parametric hinge approximation
const width = {width};
const leafHeight = {leaf_height};
const thickness = {thickness};
const barrelDiameter = {barrel_diameter};
const holeCount = {holes};
const leftLeaf = Box({{ width: width / 2, depth: leafHeight, height: thickness }}).translate([-width / 4, 0, 0]);
const rightLeaf = Box({{ width: width / 2, depth: leafHeight, height: thickness }}).translate([width / 4, 0, 0]);
const barrel = Cylinder({{ radius: barrelDiameter / 2, height: leafHeight }}).rotate([Math.PI / 2, 0, 0]);
return leftLeaf.union(rightLeaf).union(barrel);
""",
        params=[
            make_param("width", "Total Width", width, min_value=30, max_value=200),
            make_param("leafHeight", "Leaf Height", leaf_height, min_value=20, max_value=200),
            make_param("thickness", "Leaf Thickness", thickness, min_value=1, max_value=15),
            make_param("barrelDiameter", "Barrel Diameter", barrel_diameter, min_value=3, max_value=40),
            make_param("holeCount", "Hole Count", holes, min_value=0, max_value=12, step=1, unit=""),
        ],
        features=[
            make_feature("left-leaf", "Left Leaf", "box", original_type="box", operation="add"),
            make_feature("right-leaf", "Right Leaf", "box", original_type="box", operation="add"),
            make_feature("barrel", "Hinge Barrel", "cylinder", original_type="cylinder", operation="add"),
            make_feature("screw-holes", "Screw Holes", "hole", original_type="cylinder", operation="subtract"),
        ],
        message="Hinge CAD package generated successfully.",
        intent="hinge",
    )


def _build_nozzle(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    inlet = max(dims["outerDiameter"], 30)
    outlet = max(min(dims["innerDiameter"], inlet - 2), 5)
    length = max(dims["length"], 60)
    wall = max(dims["wallThickness"], 2)

    return _package(
        title="Tapered Nozzle",
        description="Parametric tapered nozzle/funnel adapter.",
        code=f"""// TextoCAD parametric tapered nozzle
const inletDiameter = {inlet};
const outletDiameter = {outlet};
const length = {length};
const wallThickness = {wall};
const outer = Cone({{ bottomRadius: inletDiameter / 2, topRadius: outletDiameter / 2 + wallThickness, height: length }});
const inner = Cone({{ bottomRadius: inletDiameter / 2 - wallThickness, topRadius: outletDiameter / 2, height: length + 2 }});
return outer.subtract(inner);
""",
        params=[
            make_param("inletDiameter", "Inlet Diameter", inlet, min_value=outlet + 2, max_value=200),
            make_param("outletDiameter", "Outlet Diameter", outlet, min_value=1, max_value=inlet - 1),
            make_param("length", "Length", length, min_value=10, max_value=300),
            make_param("wallThickness", "Wall Thickness", wall, min_value=1, max_value=15),
        ],
        features=[
            make_feature("outer-taper", "Outer Taper", "cone", original_type="cone", operation="add"),
            make_feature("inner-bore", "Inner Bore", "hole", original_type="cone", operation="subtract"),
        ],
        message="Nozzle CAD package generated successfully.",
        intent="nozzle",
    )


def _build_phone_stand(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 80)
    depth = max(dims["depth"], 70)
    height = max(dims["height"], 90)
    thickness = max(dims["thickness"], 6)
    angle = 70

    return _package(
        title="Phone Stand",
        description="Parametric angled phone/tablet stand.",
        code=f"""// TextoCAD parametric phone stand
const width = {width};
const depth = {depth};
const height = {height};
const thickness = {thickness};
const backAngle = {angle};
const base = Box({{ width, depth, height: thickness }});
const back = Box({{ width, depth: thickness, height }}).rotate([backAngle, 0, 0]).translate([0, depth * 0.2, height / 2]);
const lip = Box({{ width, depth: thickness * 2, height: thickness * 1.5 }}).translate([0, -depth / 2 + thickness, thickness]);
return base.union(back).union(lip);
""",
        params=[
            make_param("width", "Width", width, min_value=40, max_value=200),
            make_param("depth", "Depth", depth, min_value=30, max_value=180),
            make_param("height", "Back Height", height, min_value=40, max_value=220),
            make_param("thickness", "Thickness", thickness, min_value=2, max_value=20),
            make_param("backAngle", "Back Angle", angle, min_value=45, max_value=85, step=1, unit="deg"),
        ],
        features=[
            make_feature("base", "Base", "box", original_type="box", operation="add"),
            make_feature("back-support", "Back Support", "box", original_type="box", operation="add"),
            make_feature("front-lip", "Front Lip", "box", original_type="box", operation="add"),
        ],
        message="Phone stand CAD package generated successfully.",
        intent="phone_stand",
    )


def _build_bearing(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    outer = max(dims["outerDiameter"], 40)
    inner = min(max(dims["innerDiameter"], 12), outer - 10)
    width = max(dims["thickness"], 10)
    balls = max(int(dims["segmentCount"] / 4), 8)

    return _package(
        title="Ball Bearing",
        description="Simplified ball bearing with inner/outer races and ball pattern.",
        code=f"""// TextoCAD simplified ball bearing
const outerDiameter = {outer};
const innerDiameter = {inner};
const width = {width};
const ballCount = {balls};
let outerRace = Cylinder({{ radius: outerDiameter / 2, height: width }}).subtract(Cylinder({{ radius: (outerDiameter * 0.72) / 2, height: width + 2 }}));
let innerRace = Cylinder({{ radius: (innerDiameter * 1.35) / 2, height: width }}).subtract(Cylinder({{ radius: innerDiameter / 2, height: width + 2 }}));
let bearing = outerRace.union(innerRace);
for (let i = 0; i < ballCount; i++) {{
  const angle = i * 360 / ballCount;
  bearing = bearing.union(Sphere({{ radius: (outerDiameter - innerDiameter) * 0.06 }}).rotate([0, 0, angle]).translate([(outerDiameter + innerDiameter) / 4, 0, 0]));
}}
return bearing;
""",
        params=[
            make_param("outerDiameter", "Outer Diameter", outer, min_value=inner + 10, max_value=200),
            make_param("innerDiameter", "Inner Diameter", inner, min_value=2, max_value=outer - 10),
            make_param("width", "Width", width, min_value=3, max_value=60),
            make_param("ballCount", "Ball Count", balls, min_value=6, max_value=40, step=1, unit=""),
        ],
        features=[
            make_feature("outer-race", "Outer Race", "cylinder", original_type="ring", operation="add"),
            make_feature("inner-race", "Inner Race", "cylinder", original_type="ring", operation="add"),
            make_feature("ball-pattern", "Ball Pattern", "pattern", original_type="radial-pattern", operation="add"),
        ],
        message="Bearing CAD package generated successfully.",
        intent="bearing",
    )


def _build_robot(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    base_width = max(dims["width"], 120)
    base_depth = max(dims["depth"], 80)
    base_height = max(dims["thickness"], 35)
    arm_length = max(dims["length"], 220)
    forearm_length = max(arm_length * 0.75, 120)

    robot_description = {
        "format": "textocad_robot_description",
        "version": "0.1.0",
        "links": [
            {"id": "base_link", "name": "Base Link", "visual": {"kind": "box", "size": [base_width / 1000, base_depth / 1000, base_height / 1000]}},
            {"id": "upper_arm_link", "name": "Upper Arm Link", "visual": {"kind": "box", "size": [arm_length / 1000, 0.035, 0.035]}},
            {"id": "forearm_link", "name": "Forearm Link", "visual": {"kind": "box", "size": [forearm_length / 1000, 0.03, 0.03]}},
        ],
        "joints": [
            {"id": "shoulder_joint", "name": "Shoulder Joint", "jointType": "revolute", "parent": "base_link", "child": "upper_arm_link", "axis": [0, 0, 1], "limit": {"lower": -1.57, "upper": 1.57}},
            {"id": "elbow_joint", "name": "Elbow Joint", "jointType": "revolute", "parent": "upper_arm_link", "child": "forearm_link", "axis": [0, 1, 0], "limit": {"lower": -1.57, "upper": 1.57}},
        ],
    }

    return _package(
        title="Robot Arm",
        description="Simple robot arm with base, upper arm, forearm, and movable joints.",
        code=f"""// TextoCAD robot description package
const baseWidth = {base_width};
const baseDepth = {base_depth};
const baseHeight = {base_height};
const upperArmLength = {arm_length};
const forearmLength = {forearm_length};
const shoulderAngle = 0;
const elbowAngle = 0;
return Robot({{
  links: ["base_link", "upper_arm_link", "forearm_link"],
  joints: ["shoulder_joint", "elbow_joint"]
}});
""",
        params=[
            make_param("baseWidth", "Base Width", base_width, min_value=40, max_value=400),
            make_param("baseDepth", "Base Depth", base_depth, min_value=30, max_value=300),
            make_param("baseHeight", "Base Height", base_height, min_value=10, max_value=120),
            make_param("upperArmLength", "Upper Arm Length", arm_length, min_value=50, max_value=700),
            make_param("forearmLength", "Forearm Length", forearm_length, min_value=50, max_value=700),
            make_param("shoulderAngle", "Shoulder Angle", 0, min_value=-90, max_value=90, step=1, unit="deg", group="Robot joints"),
            make_param("elbowAngle", "Elbow Angle", 0, min_value=-90, max_value=90, step=1, unit="deg", group="Robot joints"),
        ],
        features=[
            make_feature("base_link", "Base Link", "robot_link", original_type="robot_link", operation="link", metadata={"visual": robot_description["links"][0]["visual"]}),
            make_feature("upper_arm_link", "Upper Arm Link", "robot_link", original_type="robot_link", operation="link", metadata={"visual": robot_description["links"][1]["visual"]}),
            make_feature("forearm_link", "Forearm Link", "robot_link", original_type="robot_link", operation="link", metadata={"visual": robot_description["links"][2]["visual"]}),
            {"id": "shoulder_joint", "name": "Shoulder Joint", "type": "robot_joint", "originalType": "revolute", "operation": "joint", "visible": True, "locked": False, "children": [], "jointType": "revolute", "parent": "base_link", "child": "upper_arm_link", "axis": [0, 0, 1], "limit": {"lower": -1.57, "upper": 1.57}},
            {"id": "elbow_joint", "name": "Elbow Joint", "type": "robot_joint", "originalType": "revolute", "operation": "joint", "visible": True, "locked": False, "children": [], "jointType": "revolute", "parent": "upper_arm_link", "child": "forearm_link", "axis": [0, 1, 0], "limit": {"lower": -1.57, "upper": 1.57}},
        ],
        message="Robot arm package generated successfully.",
        intent="robot",
        robot_description=robot_description,
    )


def _build_drone_frame(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    span = max(dims["width"], dims["length"], 220)
    arm_width = max(dims["thickness"], 12)
    plate_diameter = max(dims["diameter"], 60)
    motor_mount = max(arm_width * 1.8, 24)

    return _package(
        title="Quadcopter Drone Frame",
        description="Parametric quadcopter frame with crossed arms and motor mounts.",
        code=f"""// TextoCAD parametric drone frame
const span = {span};
const armWidth = {arm_width};
const centerPlateDiameter = {plate_diameter};
const motorMountDiameter = {motor_mount};
let frame = Cylinder({{ radius: centerPlateDiameter / 2, height: armWidth * 0.45 }});
frame = frame.union(Box({{ width: span, depth: armWidth, height: armWidth * 0.4 }}));
frame = frame.union(Box({{ width: armWidth, depth: span, height: armWidth * 0.4 }}));
for (const p of [[span/2,0,0],[-span/2,0,0],[0,span/2,0],[0,-span/2,0]]) {{
  frame = frame.union(Cylinder({{ radius: motorMountDiameter / 2, height: armWidth * 0.45 }}).translate(p));
}}
return frame;
""",
        params=[
            make_param("span", "Motor-to-Motor Span", span, min_value=100, max_value=700),
            make_param("armWidth", "Arm Width", arm_width, min_value=5, max_value=50),
            make_param("centerPlateDiameter", "Center Plate Diameter", plate_diameter, min_value=30, max_value=200),
            make_param("motorMountDiameter", "Motor Mount Diameter", motor_mount, min_value=10, max_value=100),
        ],
        features=[
            make_feature("center-plate", "Center Plate", "cylinder", original_type="cylinder", operation="add"),
            make_feature("cross-arms", "Cross Arms", "box", original_type="box", operation="add"),
            make_feature("motor-mounts", "Motor Mounts", "pattern", original_type="radial-pattern", operation="add"),
        ],
        message="Drone frame CAD package generated successfully.",
        intent="drone_frame",
    )


def _build_generic(clean_prompt: str, dims: dict[str, float]) -> dict[str, Any]:
    width = max(dims["width"], 20)
    depth = max(dims["depth"], 20)
    height = max(dims["height"], dims["thickness"], 10)
    chamfer = dims["chamfer"]

    return _package(
        title="Generic CAD Model",
        description="Generic parametric block generated from available dimensions.",
        code=f"""// TextoCAD generic parametric model
const width = {width};
const depth = {depth};
const height = {height};
const chamfer = {chamfer};
let model = Box({{ width, depth, height }});
if (chamfer > 0) model = model.chamfer(chamfer);
return model;
""",
        params=[
            make_param("width", "Width", width, min_value=1, max_value=max(width * 3, 200)),
            make_param("depth", "Depth", depth, min_value=1, max_value=max(depth * 3, 200)),
            make_param("height", "Height", height, min_value=1, max_value=max(height * 3, 120)),
            make_param("chamfer", "Chamfer", chamfer, min_value=0, max_value=8, step=0.25),
        ],
        features=[
            make_feature("main-body", "Main Body", "box", original_type="box", operation="add"),
            *_finish_features(chamfer=chamfer),
        ],
        message="Generic CAD package generated successfully.",
        intent="generic",
    )


BUILDER_BY_INTENT: dict[str, Callable[[str, dict[str, float]], dict[str, Any]]] = {
    "robot": _build_robot,
    "drone_frame": _build_drone_frame,
    "heat_sink": _build_heat_sink,
    "helical_gear": lambda prompt, dims: _build_gear(prompt, dims, helical=True),
    "gear": lambda prompt, dims: _build_gear(prompt, dims, helical=False),
    "rack_gear": _build_rack_gear,
    "flange": _build_flange,
    "computer_mouse": _build_computer_mouse,
    "cup": _build_cup,
    "bushing": _build_hollow_cylinder,
    "hollow_cylinder": _build_hollow_cylinder,
    "washer": _build_washer,
    "plate": _build_plate,
    "bracket": _build_bracket,
    "phone_stand": _build_phone_stand,
    "pulley": _build_pulley,
    "box": _build_box,
    "lid": _build_lid,
    "shaft": _build_solid_cylinder,
    "cylinder": _build_solid_cylinder,
    "sphere": _build_sphere,
    "cone": _build_cone,
    "bearing": _build_bearing,
    "bolt": _build_bolt,
    "nut": _build_nut,
    "spring": _build_spring,
    "knob": _build_knob,
    "wheel": _build_wheel,
    "fan": _build_fan,
    "clamp": _build_clamp,
    "hinge": _build_hinge,
    "nozzle": _build_nozzle,
}


def build_deterministic_package(prompt: str) -> dict[str, Any]:
    clean_prompt = prompt.strip() or "Create a simple CAD model"
    intent = infer_cad_intent(clean_prompt)
    dims = extract_dimension_hints(clean_prompt)

    builder = BUILDER_BY_INTENT.get(intent, _build_generic)
    package = builder(clean_prompt, dims)

    package["intent"] = intent
    package["dimensionHints"] = dims
    package["source"] = "deterministic_intent_builder"

    return package

from __future__ import annotations

import json
import re
from typing import Any


def strip_markdown_fences(text: str) -> str:
    clean = text.strip()

    if clean.startswith("```"):
        clean = re.sub(
            r"^```(?:json|JSON|ts|typescript|js|javascript)?",
            "",
            clean,
        ).strip()
        clean = re.sub(r"```$", "", clean).strip()

    return clean


def extract_first_json_object(text: str) -> str:
    clean = strip_markdown_fences(text)

    if clean.startswith("{") and clean.endswith("}"):
        return clean

    start = clean.find("{")

    if start < 0:
        raise ValueError("No JSON object start found in Gemini response.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(clean)):
        char = clean[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1

            if depth == 0:
                return clean[start : index + 1]

    raise ValueError("No complete JSON object found in Gemini response.")


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def parse_llm_json(text: str) -> dict[str, Any]:
    extracted = extract_first_json_object(text)
    repaired = normalize_quotes(remove_trailing_commas(extracted))

    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as error:
        raise ValueError(f"Gemini returned invalid JSON: {error}") from error

    if not isinstance(parsed, dict):
        raise ValueError("Gemini JSON response must be an object.")

    return parsed


def safe_string(value: Any, fallback: str) -> str:
    if value is None:
        return fallback

    text = str(value).strip()

    return text or fallback


def safe_number(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback

    if isinstance(value, (int, float)) and value == value:
        return float(value)

    try:
        parsed = float(value)
        if parsed == parsed:
            return parsed
    except Exception:
        pass

    return fallback


def safe_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        clean = value.strip().lower()

        if clean in {"true", "yes", "1", "on"}:
            return True

        if clean in {"false", "no", "0", "off"}:
            return False

    return fallback


ALLOWED_FEATURE_TYPES = {
    "origin",
    "plane",
    "sketch",
    "extrude",
    "revolve",
    "fillet",
    "chamfer",
    "hole",
    "pattern",
    "boolean",
    "feature",
}

FEATURE_TYPE_MAP = {
    "solid": "extrude",
    "body": "extrude",
    "part": "extrude",
    "base": "extrude",
    "boss": "extrude",
    "cylinder": "extrude",
    "cube": "extrude",
    "box": "extrude",
    "cut": "boolean",
    "subtract": "boolean",
    "subtraction": "boolean",
    "difference": "boolean",
    "union": "boolean",
    "combine": "boolean",
    "intersect": "boolean",
    "shell": "boolean",
    "bolt-hole": "hole",
    "bolt_hole": "hole",
    "through-hole": "hole",
    "through_hole": "hole",
    "mounting-hole": "hole",
    "mounting_hole": "hole",
    "bore": "hole",
}


def normalize_feature_type(value: Any) -> tuple[str, str | None]:
    original = safe_string(value, "feature").lower()
    clean = original.strip().replace(" ", "-")

    if clean in ALLOWED_FEATURE_TYPES:
        return clean, None

    mapped = FEATURE_TYPE_MAP.get(clean)

    if mapped:
        return mapped, original

    return "feature", original


def normalize_feature(feature: Any, index: int) -> dict[str, Any]:
    if not isinstance(feature, dict):
        feature = {}

    normalized_type, original_type = normalize_feature_type(feature.get("type"))

    operation = feature.get("operation")
    if operation is None:
        operation = feature.get("op")
    if operation is None:
        operation = feature.get("booleanOperation")

    item: dict[str, Any] = {
        "id": safe_string(feature.get("id"), f"feature-{index + 1}"),
        "name": safe_string(feature.get("name"), f"Feature {index + 1}"),
        "type": normalized_type,
        "visible": safe_bool(feature.get("visible"), True),
        "locked": safe_bool(feature.get("locked"), False),
    }

    if original_type:
        item["originalType"] = original_type

    if operation:
        item["operation"] = safe_string(operation, "")

    children = feature.get("children")

    if isinstance(children, list) and len(children) > 0:
        item["children"] = [
            normalize_feature(child, child_index)
            for child_index, child in enumerate(children)
        ]

    return item


def normalize_param(param: Any, index: int) -> dict[str, Any]:
    if not isinstance(param, dict):
        param = {}

    key = safe_string(param.get("key"), f"param_{index + 1}")
    key = re.sub(r"[^a-zA-Z0-9_]", "_", key)

    if not re.match(r"^[a-zA-Z_]", key):
        key = f"param_{key}"

    label = safe_string(param.get("label"), key.replace("_", " ").title())

    raw_type = safe_string(param.get("type"), "number").lower()
    param_type = (
        raw_type
        if raw_type in {"number", "integer", "boolean", "select"}
        else "number"
    )

    value = param.get("value", param.get("defaultValue", 0))

    if param_type in {"number", "integer"}:
        value = safe_number(value, 10)
        default_value = safe_number(param.get("defaultValue"), value)

        min_value = param.get("min")
        max_value = param.get("max")
        step_value = param.get("step")

        if min_value is None:
            min_value = max(value * 0.1, 0)
        else:
            min_value = safe_number(min_value, 0)

        if max_value is None:
            max_value = max(value * 2, value + 10, 100)
        else:
            max_value = safe_number(max_value, max(value * 2, 100))

        if max_value <= min_value:
            max_value = min_value + max(abs(value), 10)

        if value < min_value:
            value = min_value

        if value > max_value:
            value = max_value

        if param_type == "integer":
            value = int(round(value))
            default_value = int(round(default_value))
            min_value = int(round(min_value))
            max_value = int(round(max_value))

        return {
            "key": key,
            "label": label,
            "value": value,
            "defaultValue": default_value,
            "min": min_value,
            "max": max_value,
            "step": safe_number(step_value, 1),
            "unit": safe_string(param.get("unit"), "mm"),
            "type": param_type,
            "description": safe_string(param.get("description"), ""),
        }

    if param_type == "boolean":
        bool_value = safe_bool(value, False)

        return {
            "key": key,
            "label": label,
            "value": bool_value,
            "defaultValue": safe_bool(param.get("defaultValue"), bool_value),
            "type": "boolean",
            "description": safe_string(param.get("description"), ""),
        }

    options = param.get("options")

    if not isinstance(options, list) or len(options) == 0:
        options = [
            {"label": "Option A", "value": "A"},
            {"label": "Option B", "value": "B"},
        ]

    normalized_options = []

    for option_index, option in enumerate(options):
        if isinstance(option, dict):
            normalized_options.append(
                {
                    "label": safe_string(
                        option.get("label"),
                        f"Option {option_index + 1}",
                    ),
                    "value": option.get("value", f"option_{option_index + 1}"),
                }
            )
        else:
            normalized_options.append(
                {
                    "label": str(option),
                    "value": str(option),
                }
            )

    selected_value = value
    option_values = [option["value"] for option in normalized_options]

    if selected_value not in option_values:
        selected_value = option_values[0]

    return {
        "key": key,
        "label": label,
        "value": selected_value,
        "defaultValue": param.get("defaultValue", selected_value),
        "type": "select",
        "options": normalized_options,
        "description": safe_string(param.get("description"), ""),
    }


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _find_prompt_dimension(prompt: str, labels: list[str]) -> float | None:
    lower = prompt.lower()

    label_patterns = []

    for label in labels:
        clean = re.escape(label.lower())
        label_patterns.append(clean)

    joined = "|".join(label_patterns)

    patterns = [
        rf"(?:{joined})\s*(?:=|:|is|of|with|dia|diameter)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:mm|millimeter|millimeters|cm|m|inch|in)?",
        rf"([0-9]+(?:\.[0-9]+)?)\s*(?:mm|millimeter|millimeters|cm|m|inch|in)?\s*(?:{joined})",
    ]

    for pattern in patterns:
        match = re.search(pattern, lower)

        if match:
            return safe_number(match.group(1), 0)

    return None


def _infer_prompt_kind(prompt: str) -> str:
    text = prompt.lower()

    if "gear" in text or "teeth" in text or "tooth" in text:
        return "gear"

    if "pulley" in text:
        return "pulley"

    if "enclosure" in text or "case" in text or "housing" in text:
        return "enclosure"

    if "flange" in text:
        return "flange"

    if "washer" in text:
        return "washer"

    if "pipe" in text or "tube" in text:
        return "pipe"

    if "hollow cylinder" in text:
        return "hollow-cylinder"

    if "cylinder" in text and (
        "inner" in text or "bore" in text or "hole" in text
    ):
        return "hollow-cylinder"

    if "cylinder" in text:
        return "cylinder"

    if "bracket" in text:
        return "bracket"

    if "plate" in text:
        return "plate"

    if "shaft" in text or "rod" in text:
        return "shaft"

    return "generic"


def _find_param(params: list[dict[str, Any]], keys: list[str]) -> dict[str, Any] | None:
    normalized = {_normalize_key(key) for key in keys}

    for param in params:
        param_key = _normalize_key(str(param.get("key", "")))
        param_label = _normalize_key(str(param.get("label", "")))

        if param_key in normalized or param_label in normalized:
            return param

    return None


def _upsert_numeric_param(
    params: list[dict[str, Any]],
    *,
    keys: list[str],
    key: str,
    label: str,
    value: float,
    min_value: float,
    max_value: float,
    step: float = 1,
    unit: str = "mm",
    description: str = "",
) -> None:
    existing = _find_param(params, keys + [key, label])

    value = safe_number(value, min_value)

    if max_value <= min_value:
        max_value = min_value + 10

    value = max(min_value, min(value, max_value))

    payload = {
        "key": key,
        "label": label,
        "value": value,
        "defaultValue": value,
        "min": min_value,
        "max": max_value,
        "step": step,
        "unit": unit,
        "type": "number",
        "description": description,
    }

    if existing is not None:
        existing.update(payload)
    else:
        params.append(payload)


def _ensure_outer_inner_are_valid(params: list[dict[str, Any]]) -> None:
    outer = _find_param(
        params,
        [
            "outerDiameter",
            "outsideDiameter",
            "od",
            "flangeDiameter",
            "diameter",
        ],
    )
    inner = _find_param(
        params,
        [
            "innerDiameter",
            "insideDiameter",
            "boreDiameter",
            "id",
            "holeDiameter",
        ],
    )

    if not outer or not inner:
        return

    outer_value = safe_number(outer.get("value"), 40)
    inner_value = safe_number(inner.get("value"), 0)

    if inner_value >= outer_value:
        inner_value = max(outer_value * 0.5, outer_value - 10)

    inner["value"] = inner_value
    inner["defaultValue"] = inner_value
    inner["max"] = min(
        safe_number(inner.get("max"), outer_value * 0.9),
        outer_value * 0.9,
    )


def _apply_prompt_dimension_overrides(
    *,
    params: list[dict[str, Any]],
    features: list[dict[str, Any]],
    prompt: str,
    title: str,
) -> tuple[str, list[dict[str, Any]]]:
    kind = _infer_prompt_kind(prompt)

    outer = _find_prompt_dimension(
        prompt,
        [
            "outer diameter",
            "outside diameter",
            "outer dia",
            "outside dia",
            "od",
        ],
    )
    inner = _find_prompt_dimension(
        prompt,
        [
            "inner diameter",
            "inside diameter",
            "inner dia",
            "inside dia",
            "bore diameter",
            "bore dia",
            "id",
        ],
    )
    length = _find_prompt_dimension(
        prompt,
        [
            "length",
            "long",
        ],
    )
    height = _find_prompt_dimension(
        prompt,
        [
            "height",
            "tall",
        ],
    )
    thickness = _find_prompt_dimension(
        prompt,
        [
            "thickness",
            "thick",
            "width",
        ],
    )
    width = _find_prompt_dimension(
        prompt,
        [
            "width",
            "wide",
        ],
    )
    depth = _find_prompt_dimension(
        prompt,
        [
            "depth",
            "deep",
        ],
    )

    if kind in {"pipe", "hollow-cylinder"}:
        if outer is None:
            outer = 40
        if inner is None:
            inner = max(outer * 0.6, outer - 10)
        if inner >= outer:
            inner = outer * 0.6

        model_length = length or height or thickness or outer * 2

        _upsert_numeric_param(
            params,
            keys=["outerDiameter", "outsideDiameter", "od", "diameter"],
            key="outerDiameter",
            label="Outer Diameter",
            value=outer,
            min_value=max(outer * 0.25, 1),
            max_value=max(outer * 3, outer + 50),
            description="Outer diameter of the hollow cylinder or pipe.",
        )

        _upsert_numeric_param(
            params,
            keys=["innerDiameter", "insideDiameter", "boreDiameter", "id"],
            key="innerDiameter",
            label="Inner Diameter",
            value=inner,
            min_value=1,
            max_value=outer * 0.9,
            description="Inner bore diameter of the hollow cylinder or pipe.",
        )

        _upsert_numeric_param(
            params,
            keys=["length", "height", "pipeLength", "cylinderHeight"],
            key="length",
            label="Length",
            value=model_length,
            min_value=1,
            max_value=max(model_length * 3, 200),
            description="Length of the hollow cylinder or pipe.",
        )

        title = "Hollow Cylinder" if kind == "hollow-cylinder" else "Pipe"

        features = [
            {
                "id": "outer-body",
                "name": "Outer Cylindrical Body",
                "type": "extrude",
                "visible": True,
                "locked": False,
                "originalType": "solid",
                "operation": "add",
            },
            {
                "id": "inner-bore",
                "name": "Inner Bore",
                "type": "hole",
                "visible": True,
                "locked": False,
                "originalType": "cut",
                "operation": "subtract",
            },
        ]

    elif kind == "washer":
        if outer is None:
            outer = 60
        if inner is None:
            inner = outer * 0.4
        if inner >= outer:
            inner = outer * 0.45

        model_thickness = thickness or height or length or max(outer * 0.1, 5)

        _upsert_numeric_param(
            params,
            keys=["outerDiameter", "outsideDiameter", "od"],
            key="outerDiameter",
            label="Outer Diameter",
            value=outer,
            min_value=max(outer * 0.25, 1),
            max_value=max(outer * 3, outer + 50),
            description="Outer diameter of the washer.",
        )

        _upsert_numeric_param(
            params,
            keys=["innerDiameter", "boreDiameter", "holeDiameter", "id"],
            key="innerDiameter",
            label="Inner Diameter",
            value=inner,
            min_value=1,
            max_value=outer * 0.9,
            description="Inner hole diameter of the washer.",
        )

        _upsert_numeric_param(
            params,
            keys=["thickness", "height"],
            key="thickness",
            label="Thickness",
            value=model_thickness,
            min_value=0.5,
            max_value=max(model_thickness * 3, 50),
            step=0.5,
            description="Washer thickness.",
        )

        title = "Washer"

        features = [
            {
                "id": "washer-body",
                "name": "Washer Body",
                "type": "extrude",
                "visible": True,
                "locked": False,
                "originalType": "solid",
                "operation": "add",
            },
            {
                "id": "center-hole",
                "name": "Center Hole",
                "type": "hole",
                "visible": True,
                "locked": False,
                "originalType": "cut",
                "operation": "subtract",
            },
        ]

    elif kind == "cylinder":
        diameter = outer or _find_prompt_dimension(prompt, ["diameter", "dia"]) or 40
        model_height = height or length or thickness or diameter

        _upsert_numeric_param(
            params,
            keys=["diameter", "outerDiameter"],
            key="diameter",
            label="Diameter",
            value=diameter,
            min_value=1,
            max_value=max(diameter * 3, 100),
            description="Cylinder diameter.",
        )

        _upsert_numeric_param(
            params,
            keys=["height", "length"],
            key="height",
            label="Height",
            value=model_height,
            min_value=1,
            max_value=max(model_height * 3, 200),
            description="Cylinder height.",
        )

        title = "Cylinder"

        features = [
            {
                "id": "cylinder-body",
                "name": "Cylinder Body",
                "type": "extrude",
                "visible": True,
                "locked": False,
                "originalType": "solid",
                "operation": "add",
            },
        ]

    elif kind == "enclosure":
        model_width = width or 100
        model_depth = depth or 70
        model_height = height or 45
        wall = thickness or 4

        _upsert_numeric_param(
            params,
            keys=["width", "enclosureWidth"],
            key="width",
            label="Width",
            value=model_width,
            min_value=10,
            max_value=max(model_width * 3, 300),
            description="Overall enclosure width.",
        )

        _upsert_numeric_param(
            params,
            keys=["depth", "enclosureDepth"],
            key="depth",
            label="Depth",
            value=model_depth,
            min_value=10,
            max_value=max(model_depth * 3, 300),
            description="Overall enclosure depth.",
        )

        _upsert_numeric_param(
            params,
            keys=["height", "enclosureHeight"],
            key="height",
            label="Height",
            value=model_height,
            min_value=10,
            max_value=max(model_height * 3, 200),
            description="Overall enclosure height.",
        )

        _upsert_numeric_param(
            params,
            keys=["wallThickness", "thickness"],
            key="wallThickness",
            label="Wall Thickness",
            value=wall,
            min_value=1,
            max_value=max(wall * 3, 20),
            step=0.5,
            description="Wall thickness.",
        )

        title = "Open Electronics Enclosure"

        features = [
            {
                "id": "enclosure-shell",
                "name": "Enclosure Shell",
                "type": "extrude",
                "visible": True,
                "locked": False,
                "originalType": "shell",
                "operation": "add",
            },
        ]

    _ensure_outer_inner_are_valid(params)

    return title, features


def normalize_cad_json(data: dict[str, Any], *, fallback_prompt: str) -> dict[str, Any]:
    title = safe_string(data.get("title"), "Generated CAD Model")
    description = safe_string(data.get("description"), "")

    code = safe_string(
        data.get("code"),
        f"// Generated CAD model\n// Prompt: {fallback_prompt}\n",
    )

    params_raw = data.get("params")
    if not isinstance(params_raw, list):
        params_raw = []

    features_raw = data.get("features")
    if not isinstance(features_raw, list):
        features_raw = []

    params = [normalize_param(param, index) for index, param in enumerate(params_raw)]
    features = [
        normalize_feature(feature, index)
        for index, feature in enumerate(features_raw)
    ]

    title, features = _apply_prompt_dimension_overrides(
        params=params,
        features=features,
        prompt=fallback_prompt,
        title=title,
    )

    if not features:
        features = [
            {
                "id": "origin",
                "name": "Origin",
                "type": "origin",
                "visible": True,
                "locked": False,
            },
            {
                "id": "generated-feature",
                "name": title,
                "type": "feature",
                "visible": True,
                "locked": False,
            },
        ]

    return {
        "title": title,
        "description": description,
        "code": code,
        "params": params,
        "features": features,
        "units": safe_string(data.get("units"), "mm"),
        "version": safe_string(data.get("version"), "1.0.0"),
    }

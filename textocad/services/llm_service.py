from __future__ import annotations

import json
import re
from typing import Any, Literal

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """What the adapters actually read off a config object."""

    system_instruction: str | None = None
    temperature: float | None = None
    response_mime_type: str = "application/json"
from pydantic import BaseModel, Field, ValidationError, field_validator

from textocad.core.config import settings
from textocad.services.cad_intent import build_deterministic_package
from textocad.services.cad_repair import repair_cad_document, repair_cad_document_dict
from textocad.services.cad_schema import cad_document_to_dict
from textocad.services.cad_operations import supported_operation_names
from textocad.services.cadquery_compiler import compile_cadquery
from textocad.services.openscad_compiler import compile_openscad


GenerationMode = Literal["fast", "standard", "extended", "robot"]
GenerationPipeline = Literal["fast_mesh", "openscad", "cadquery", "robot_urdf"]

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
    "robot_link",
    "robot_joint",
}

PIPELINE_BY_MODE: dict[GenerationMode, GenerationPipeline] = {
    "fast": "fast_mesh",
    "standard": "openscad",
    "extended": "cadquery",
    "robot": "robot_urdf",
}

SCRIPT_TARGET_BY_MODE: dict[GenerationMode, str] = {
    "fast": "textocad",
    "standard": "openscad",
    "extended": "cadquery",
    "robot": "robot_urdf",
}


def normalize_feature_type(value: str | None) -> str:
    clean = str(value or "").strip().lower()

    if clean in ALLOWED_FEATURE_TYPES:
        return clean

    if clean in {"solid", "body", "part", "cylinder", "boss", "base", "block"}:
        return "extrude"

    if clean in {"cut", "subtract", "union", "combine", "difference", "shell"}:
        return "boolean"

    if clean in {"bolt-hole", "bolt_hole", "through-hole", "hole-cut", "bore"}:
        return "hole"

    if clean in {"link", "robotlink", "robot-link"}:
        return "robot_link"

    if clean in {"joint", "robotjoint", "robot-joint", "revolute", "prismatic"}:
        return "robot_joint"

    return "feature"


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback

    text = str(value).strip()
    return text or fallback


def _normalize_generation_mode(value: Any) -> GenerationMode:
    clean = _safe_text(value).lower()

    if clean == "standard":
        return "standard"
    if clean == "extended":
        return "extended"
    if clean in {"robot", "robotics", "urdf", "xacro"}:
        return "robot"

    return "fast"


def _normalize_generation_pipeline(
    value: Any,
    mode: GenerationMode,
) -> GenerationPipeline:
    clean = _safe_text(value).lower()

    if clean == "fast_mesh":
        return "fast_mesh"
    if clean == "openscad":
        return "openscad"
    if clean == "cadquery":
        return "cadquery"
    if clean == "robot_urdf":
        return "robot_urdf"

    return PIPELINE_BY_MODE[mode]


class CadParamLLM(BaseModel):
    key: str
    label: str
    value: int | float | str | bool
    defaultValue: int | float | str | bool | None = None
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    unit: str | None = "mm"
    type: str | None = "number"
    description: str | None = None
    locked: bool | None = False
    group: str | None = None
    control: str | None = None
    options: list[Any] | None = None

    model_config = {"extra": "allow"}

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_]", "", value.strip())

        if not clean:
            return "parameter"

        if clean[0].isdigit():
            clean = f"param_{clean}"

        return clean

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str | None) -> str:
        clean = str(value or "number").strip().lower()

        if clean in {"number", "integer", "boolean", "select", "text", "enum", "string"}:
            if clean == "integer":
                return "number"
            if clean == "select":
                return "enum"
            if clean == "text":
                return "string"
            return clean

        return "number"


class FeatureItemLLM(BaseModel):
    id: str
    name: str
    type: str
    originalType: str | None = None
    operation: str | None = None
    visible: bool | None = True
    locked: bool | None = False
    children: list["FeatureItemLLM"] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def validate_feature_type(cls, value: str | None) -> str:
        return normalize_feature_type(value)

    @field_validator("children", mode="before")
    @classmethod
    def validate_children(cls, value: Any) -> list[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return []


class GeminiCadPackage(BaseModel):
    title: str = Field(..., min_length=1)
    description: str | None = None
    code: str = ""
    schemaVersion: str | None = None
    modelType: str | None = None
    operations: list[dict[str, Any]] = Field(default_factory=list)
    robot: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    params: list[CadParamLLM] = Field(default_factory=list)
    features: list[FeatureItemLLM] = Field(default_factory=list)
    message: str | None = None

    generationMode: str | None = None
    generationPipeline: str | None = None
    scriptTarget: str | None = None
    allowedExportFormats: list[str] | None = None
    robotDescription: dict[str, Any] | None = None

    model_config = {"extra": "allow"}


class CadGenerationResult(BaseModel):
    ok: bool
    package: GeminiCadPackage | None = None
    raw_text: str | None = None
    error: str | None = None


BASE_SYSTEM_INSTRUCTION = """
You are TextoCAD, an AI CAD generation engine.

Return ONLY valid JSON.
Do not return Markdown.
Do not wrap JSON in ``` fences.
Do not return OpenSCAD, CadQuery, Python, JavaScript, imports, shell commands, or executable code.
The LLM output is data only. TextoCAD compilers generate all code later.

Your output must match this JSON shape:
{
  "schemaVersion": "textocad.v1",
  "title": "Short CAD model title",
  "units": "mm",
  "modelType": "part",
  "params": [
    {"key": "width", "label": "Width", "type": "number", "value": 80, "defaultValue": 80, "min": 1, "max": 300, "step": 1, "unit": "mm", "group": "Dimensions", "description": "Width", "locked": false}
  ],
  "operations": [
    {"id": "main_body", "type": "box", "visible": true, "enabled": true, "params": {"size": ["width", 40, 10]}, "properties": {}, "inputs": [], "target": null, "transform": {"translate": [0, 0, 5], "rotate": [0, 0, 0], "scale": [1, 1, 1]}, "tags": ["body"], "metadata": {}}
  ],
  "robot": null,
  "metadata": {"intent": "part"}
}

Common operation.type values include:
- box
- cylinder
- sphere
- cone
- wedge
- triangular_prism
- polygon_extrude
- linear_extrude
- revolve
- union
- subtract
- intersect
- translate
- rotate
- scale
- mirror
- pattern_linear
- pattern_circular
- hole
- hole_pattern
- fillet
- chamfer
- gear_spur_approx
- gear_helical_approx
- text_emboss
- import_mesh_placeholder

Full registry rule:
- Use operation registry names only. Do not invent unsupported operation names.
- Prefer high-level TextoCAD operations over raw CadQuery/OpenSCAD method names.
- Use operation ids and references through target and inputs.
- Do not output raw CadQuery, OpenSCAD, Python, JavaScript, imports, or executable code as source of truth.
- If an advanced feature is requested, add metadata.supportIntent tags when useful:
  true_fillet, true_chamfer, true_loft, true_sweep, true_shell, hole_pattern, robot_joint.
- If a requested operation is not in the registry, represent it with the closest supported operation and add warning metadata.

Operation object rules:
- Always include id, type, visible, enabled, params, properties, inputs, target, transform, tags, metadata.
- Use transform.translate/rotate/scale for placement.
- For holes use hole or hole_pattern with diameter, depth, axis, position/positions.
- For L-brackets create separate base plate, rear vertical plate, base holes, back holes, triangular gussets, and fillet operation.
- For gears use gear_spur_approx or gear_helical_approx with outer_diameter, root_diameter, bore_diameter, face_width, tooth_count.

Rules:
- Always include useful numeric params with min, max, step, unit, and type.
- Use millimeters unless user clearly asks otherwise.
- For hollow cylinder / tube / pipe, include outerDiameter, innerDiameter, and length or thickness.
- For flange, include flangeDiameter, boreDiameter, flangeThickness, boltHoleCount, boltCircleDiameter, boltHoleDiameter.
- For gear, include outerDiameter, boreDiameter, thickness, toothCount.
- Expressions may use only numbers, param names, + - * / parentheses, min/max/clamp.
- Do not invent code fields.

Examples:
1. L-bracket: operations are base_plate box, rear_vertical_plate box, base_holes hole_pattern on Z axis, back_plate_holes hole_pattern on X axis, two triangular_prism gussets, fillet.
2. Spur gear: one gear_spur_approx operation with tooth_count, outer_diameter, root_diameter, bore_diameter, face_width.
3. Simple robot arm: modelType robot, robot links each with operations, joints with origin/rpy/axis/limits.
4. Lofted duct: use loft with metadata.supportIntent ["true_loft"], shell with supportIntent ["true_shell"], and chamfer with supportIntent ["true_chamfer"]. Keep fallback dimensions and warnings.
5. Swept pipe/bracket: use sweep with explicit profile/path ids and supportIntent ["true_sweep"]; include a safe cylinder/box approximation if needed.
6. Four-bar linkage: modelType robot, four link objects, revolute joints with parent/child/origin/axis/limits, metadata.supportIntent ["robot_joint"].
"""


MODE_INSTRUCTIONS: dict[GenerationMode, str] = {
    "fast": """
FAST MODE:
- Target: instant browser preview and lightweight mesh export.
- Return canonical CAD JSON only.
- Prefer robust simple primitives: cylinder, box, hole, fillet/chamfer descriptions, patterns.
- Exports available: STL, GLB, JSON.
- Do not promise true STEP or B-Rep CAD in Fast mode.
""",
    "standard": """
STANDARD MODE:
- Target: backend OpenSCAD-style rebuild for reliable printable mesh output.
- Return canonical CAD JSON only. TextoCAD will compile this JSON to OpenSCAD.
- Prefer CSG operations: union, difference, cylinder, cube, rotate_extrude, linear_extrude, translate, rotate.
- Avoid freeform NURBS or complex B-Rep-only operations.
- Exports available after backend rebuild: STL, 3MF, GLB, JSON.
- Do not promise true STEP in Standard mode.
""",
    "extended": """
EXTENDED MODE:
- Target: backend CadQuery/OpenCascade-style true CAD model.
- Return canonical CAD JSON only. TextoCAD will compile this JSON to deterministic CadQuery Python.
- Prefer precise feature history: Workplane, circle, rect, extrude, cut, hole, fillet, chamfer, revolve, loft where appropriate.
- Exports available after backend rebuild: STEP, STL, GLB, JSON.
- This mode is for true CAD / B-Rep intent.
""",
    "robot": """
ROBOT MODE:
- Target: robot description with links, joints, URDF/Xacro/ROS package output.
- Return canonical CAD JSON only with modelType "robot" and a robot object.
- Every movable joint must include:
  - jointType: revolute, prismatic, continuous, fixed
  - parent
  - child
  - axis
  - limit with lower and upper when applicable
- Include robotDescription with links and joints if useful.
- Exports available after backend rebuild: URDF, Xacro, ROS ZIP, GLB, JSON.
- The viewer should later rotate movable links using joint values.
""",
}


def _mode_exports(mode: GenerationMode) -> list[str]:
    if mode == "standard":
        return ["stl", "3mf", "glb", "json"]
    if mode == "extended":
        return ["step", "stl", "glb", "json"]
    if mode == "robot":
        return ["urdf", "xacro", "ros_zip", "glb", "json"]

    return ["stl", "glb", "json"]


def _extract_json_object(text: str) -> dict[str, Any]:
    clean = text.strip()

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?", "", clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r"```$", "", clean).strip()

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    first = clean.find("{")
    last = clean.rfind("}")

    if first == -1 or last == -1 or last <= first:
        raise ValueError("No JSON object found in Gemini response.")

    possible_json = clean[first : last + 1]
    parsed = json.loads(possible_json)

    if not isinstance(parsed, dict):
        raise ValueError("Gemini response JSON is not an object.")

    return parsed


def _json_preview(value: Any, limit: int = 4000) -> str:
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def _build_conversation_context(params: dict[str, Any]) -> str:
    previous_messages = params.get("previousMessages") or params.get("previous_messages") or []
    conversation_summary = params.get("conversationSummary") or params.get("conversation_summary") or ""

    lines: list[str] = []

    if conversation_summary:
        lines.append("Conversation summary:")
        lines.append(str(conversation_summary)[:2000])

    if isinstance(previous_messages, list) and previous_messages:
        lines.append("Recent messages:")
        for message in previous_messages[-8:]:
            if not isinstance(message, dict):
                continue

            role = _safe_text(message.get("role"), "user")
            content = _safe_text(message.get("content"))
            if not content:
                continue

            lines.append(f"- {role}: {content[:700]}")

    return "\n".join(lines).strip()


def _build_user_prompt(params: dict[str, Any]) -> str:
    prompt = str(params.get("prompt") or "").strip()

    generation_mode = _normalize_generation_mode(
        params.get("generationMode") or params.get("generation_mode")
    )
    generation_pipeline = _normalize_generation_pipeline(
        params.get("generationPipeline") or params.get("generation_pipeline"),
        generation_mode,
    )

    edit_mode = str(params.get("mode") or "new").strip()
    current_code = str(params.get("currentCode") or params.get("current_code") or "").strip()
    current_params = params.get("currentParams") or params.get("current_params") or []
    current_features = params.get("currentFeatures") or params.get("current_features") or []

    deterministic_hint = build_deterministic_package(prompt)
    conversation_context = _build_conversation_context(params)

    return f"""
User request:
{prompt}

Frontend generation mode:
{generation_mode}

Backend pipeline:
{generation_pipeline}

Edit/generate mode:
{edit_mode}

Mode-specific instructions:
{MODE_INSTRUCTIONS[generation_mode]}

Conversation context:
{conversation_context or "No previous conversation context provided."}

Existing CAD code, if any:
{current_code[:5000]}

Existing parameters, if any:
{_json_preview(current_params, 5000)}

Existing features, if any:
{_json_preview(current_features, 5000)}

Reference deterministic canonical CAD JSON:
{_json_preview(repair_cad_document_dict(deterministic_hint, prompt=prompt, mode=generation_mode)[0], 7000)}

Required package metadata:
{{
  "generationMode": "{generation_mode}",
  "generationPipeline": "{generation_pipeline}",
  "scriptTarget": "{SCRIPT_TARGET_BY_MODE[generation_mode]}",
  "allowedExportFormats": {_json_preview(_mode_exports(generation_mode))}
}}

Return a complete canonical TextoCAD CAD JSON document as strict JSON.

Important behavior:
- Prefer the user's latest request over older existing code.
- If existing code/features are provided and the user asks to modify, keep the previous design intent and apply the requested change.
- If the user asks for a hollow cylinder, pipe, tube, or inner/outer diameter cylinder, do not return a block or generic model.
- If the user asks for a robot, include robot_link and robot_joint features.
- If the requested shape is complex, still return a valid simplified parametric package instead of failing.
"""


def _get_system_instruction(mode: GenerationMode) -> str:
    registry_names = ", ".join(supported_operation_names())
    return (
        f"{BASE_SYSTEM_INSTRUCTION}\n\n"
        f"Current operation registry names:\n{registry_names}\n\n"
        f"{MODE_INSTRUCTIONS[mode]}"
    )


def _get_client() -> Any:


    from textocad.services.llm_models import get_active, resolve_default


    active = get_active() or resolve_default()
    if active is not None:
        if active.protocol == "openai":
            from textocad.services.openai_adapter import OpenAICompatClient

            return OpenAICompatClient(
                api_key=active.api_key,
                base_url=active.base_url,
                model=active.model,
                vision_model=active.vision_model,
                supports_images=active.supports_images,
            )

        from textocad.services.anthropic_adapter import AnthropicClient

        return AnthropicClient(
            api_key=active.api_key,
            base_url=active.base_url,
            model=active.model,
            vision_model=active.vision_model,
            supports_images=active.supports_images,
        )

    from textocad.services.llm_models import byok_active

    if byok_active():


        raise RuntimeError("Your saved API key does not cover this engine.")

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not configured.")

        from textocad.services.anthropic_adapter import AnthropicClient

        return AnthropicClient(api_key=settings.anthropic_api_key)


    raise RuntimeError(
        "No LLM provider is configured. Set DEEPSEEK_API_KEY for the default "
        "engine (or a key for whichever provider LLM_DEFAULT_PROVIDER names)."
    )


def multimodal_generate(
    system_instruction: str,
    user_text: str,
    *,
    image: dict[str, Any] | None = None,
    temperature: float = 0.2,
) -> str:
    """Provider-agnostic single-shot JSON generation, optionally with ONE reference image."""
    from textocad.services.cancellation import check as cancel_check

    cancel_check()
    client = _get_client()
    config = GenerationConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )


    text = (
        client.models.generate_content(
            contents=user_text,
            config=config,
            image=image,
        ).text
        or ""
    )
    cancel_check()
    return text


def _repair_param_payload(param: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(param)

    key = re.sub(r"[^a-zA-Z0-9_]", "", str(item.get("key") or f"param_{index + 1}"))

    if not key:
        key = f"param_{index + 1}"

    if key[0].isdigit():
        key = f"param_{key}"

    item["key"] = key
    item["label"] = str(item.get("label") or key)
    item["value"] = item.get("value", item.get("defaultValue", 0))
    item["defaultValue"] = item.get("defaultValue", item["value"])

    param_type = str(item.get("type") or "number").lower()
    if param_type == "integer":
        param_type = "number"
    if param_type == "select":
        param_type = "enum"
    if param_type == "text":
        param_type = "string"

    if param_type not in {"number", "boolean", "enum", "string"}:
        param_type = "number"

    item["type"] = param_type
    item["unit"] = item.get("unit", "mm")
    item["description"] = str(item.get("description") or item["label"])
    item["locked"] = bool(item.get("locked", False))

    if param_type == "number":
        value = item["value"]

        if isinstance(value, str):
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except Exception:
                value = 0

        if not isinstance(value, (int, float)):
            value = 0

        item["value"] = value
        item["defaultValue"] = item.get("defaultValue", value)

        if item.get("min") is None:
            item["min"] = 0

        if item.get("max") is None:
            item["max"] = max(abs(float(value)) * 3 if value else 100, 100)

        if item.get("step") is None:
            item["step"] = 1

        try:
            if float(item["min"]) >= float(item["max"]):
                item["max"] = float(item["min"]) + 1
        except Exception:
            item["min"] = 0
            item["max"] = 100

    if param_type == "enum" and not isinstance(item.get("options"), list):
        item["options"] = []

    return item


def _repair_feature_payload(feature: dict[str, Any], index: int) -> dict[str, Any]:
    item = dict(feature)

    item["id"] = str(item.get("id") or f"feature-{index + 1}")
    item["name"] = str(item.get("name") or f"Feature {index + 1}")
    item["type"] = normalize_feature_type(str(item.get("type") or "feature"))
    item["originalType"] = str(
        item.get("originalType")
        or item.get("original_type")
        or item["type"]
    )
    item["operation"] = str(item.get("operation") or item["type"])
    item["visible"] = bool(item.get("visible", True))
    item["locked"] = bool(item.get("locked", False))

    children = item.get("children")

    if not isinstance(children, list):
        children = []

    item["children"] = [
        _repair_feature_payload(child, child_index)
        for child_index, child in enumerate(children)
        if isinstance(child, dict)
    ]

    return item


def _default_robot_features() -> list[dict[str, Any]]:
    return [
        {
            "id": "base_link",
            "name": "Base Link",
            "type": "robot_link",
            "originalType": "robot_link",
            "operation": "link",
            "visible": True,
            "locked": False,
            "children": [],
            "visual": {"kind": "box", "size": [0.12, 0.08, 0.04]},
        },
        {
            "id": "arm_link",
            "name": "Arm Link",
            "type": "robot_link",
            "originalType": "robot_link",
            "operation": "link",
            "visible": True,
            "locked": False,
            "children": [],
            "visual": {"kind": "box", "size": [0.22, 0.035, 0.035]},
        },
        {
            "id": "shoulder_joint",
            "name": "Shoulder Joint",
            "type": "robot_joint",
            "originalType": "revolute",
            "operation": "joint",
            "visible": True,
            "locked": False,
            "children": [],
            "jointType": "revolute",
            "parent": "base_link",
            "child": "arm_link",
            "axis": [0, 0, 1],
            "limit": {"lower": -1.57, "upper": 1.57},
        },
    ]


def _default_robot_params() -> list[dict[str, Any]]:
    return [
        {
            "key": "baseWidth",
            "label": "Base Width",
            "value": 120,
            "defaultValue": 120,
            "min": 40,
            "max": 300,
            "step": 1,
            "unit": "mm",
            "type": "number",
            "description": "Robot base width.",
        },
        {
            "key": "armLength",
            "label": "Arm Length",
            "value": 220,
            "defaultValue": 220,
            "min": 50,
            "max": 600,
            "step": 1,
            "unit": "mm",
            "type": "number",
            "description": "Main arm link length.",
        },
        {
            "key": "shoulderAngle",
            "label": "Shoulder Angle",
            "value": 0,
            "defaultValue": 0,
            "min": -90,
            "max": 90,
            "step": 1,
            "unit": "deg",
            "type": "number",
            "description": "Preview joint angle.",
        },
    ]


def _build_robot_description(features: list[dict[str, Any]]) -> dict[str, Any]:
    links: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []

    for feature in features:
        feature_type = str(feature.get("type") or "")

        if feature_type == "robot_link":
            links.append(
                {
                    "id": feature.get("id"),
                    "name": feature.get("name"),
                    "visual": feature.get("visual"),
                    "collision": feature.get("collision"),
                }
            )

        if feature_type == "robot_joint":
            joints.append(
                {
                    "id": feature.get("id"),
                    "name": feature.get("name"),
                    "jointType": feature.get("jointType", feature.get("joint_type", "revolute")),
                    "parent": feature.get("parent"),
                    "child": feature.get("child"),
                    "axis": feature.get("axis", [0, 0, 1]),
                    "limit": feature.get("limit", {"lower": -1.57, "upper": 1.57}),
                }
            )

    if not links or not joints:
        default_features = _default_robot_features()
        return _build_robot_description(default_features)

    return {
        "format": "textocad_robot_description",
        "version": "0.1.0",
        "links": links,
        "joints": joints,
    }


def _features_from_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = [
        {"id": "origin", "name": "Origin", "type": "origin", "originalType": "origin", "operation": "reference", "visible": True, "locked": False, "children": []},
        {"id": "xy", "name": "XY Plane", "type": "plane", "originalType": "plane", "operation": "reference", "visible": True, "locked": False, "children": []},
        {"id": "xz", "name": "XZ Plane", "type": "plane", "originalType": "plane", "operation": "reference", "visible": True, "locked": False, "children": []},
        {"id": "yz", "name": "YZ Plane", "type": "plane", "originalType": "plane", "operation": "reference", "visible": True, "locked": False, "children": []},
    ]
    for op in operations:
        op_type = str(op.get("type") or "feature")
        feature_type = "feature"
        if op_type in {"hole", "hole_pattern"}:
            feature_type = "hole"
        elif op_type in {"fillet", "chamfer"}:
            feature_type = op_type
        elif op_type in {"union", "subtract", "intersect"}:
            feature_type = "boolean"
        elif op_type in {"pattern_linear", "pattern_circular"}:
            feature_type = "pattern"
        elif op_type in {"revolve"}:
            feature_type = "revolve"
        elif op_type != "unsupported":
            feature_type = "extrude"
        features.append(
            {
                "id": str(op.get("id") or f"operation_{len(features)}"),
                "name": str(op.get("id") or op_type).replace("_", " ").title(),
                "type": feature_type,
                "originalType": op_type,
                "operation": "subtract" if op_type in {"hole", "hole_pattern", "subtract"} else "add",
                "visible": bool(op.get("visible", True)),
                "locked": False,
                "children": [],
                "metadata": {"operation": op},
            }
        )
    return features


def _compiled_display_code(document_data: dict[str, Any], generation_pipeline: GenerationPipeline) -> str:
    document = repair_cad_document(document_data, prompt=str(document_data.get("metadata", {}).get("prompt") or ""))[0]
    if generation_pipeline == "openscad":
        return compile_openscad(document).source
    if generation_pipeline == "cadquery":
        return compile_cadquery(document).source
    return json.dumps(document_data, indent=2)


def _repair_package_payload(
    data: dict[str, Any],
    prompt: str,
    generation_mode: GenerationMode,
    generation_pipeline: GenerationPipeline,
) -> dict[str, Any]:
    fallback = build_deterministic_package(prompt)

    repaired = dict(data)

    if repaired.get("schemaVersion") != "textocad.v1" and not isinstance(repaired.get("operations"), list):
        repaired = {**fallback, **repaired}

    document, repair_warnings = repair_cad_document(
        repaired,
        prompt=prompt,
        mode="robot" if generation_mode == "robot" else "part",
    )
    document_data = cad_document_to_dict(document)

    repaired.update(document_data)
    repaired["title"] = str(document_data.get("title") or repaired.get("title") or fallback["title"])
    repaired["description"] = str(repaired.get("description") or fallback.get("description") or "")
    repaired["code"] = _compiled_display_code(document_data, generation_pipeline)
    repaired["message"] = str(repaired.get("message") or "Canonical CAD JSON generated successfully.")
    repaired["cadDocument"] = document_data
    repaired["cad_document"] = document_data
    repaired["features"] = _features_from_operations(document_data.get("operations", []))

    repaired["generationMode"] = generation_mode
    repaired["generationPipeline"] = generation_pipeline
    repaired["scriptTarget"] = SCRIPT_TARGET_BY_MODE[generation_mode]
    repaired["allowedExportFormats"] = _mode_exports(generation_mode)
    repaired["previewSource"] = "fast"
    repaired["compileStatus"] = "ready" if generation_mode == "fast" else "dirty"

    metadata = dict(repaired.get("metadata") or {})
    metadata["directLlmCodeExecution"] = False
    if repair_warnings:
        metadata["repairWarnings"] = repair_warnings
    repaired["metadata"] = metadata

    if generation_mode == "robot":
        robot_description = document_data.get("robot")
        repaired["robot"] = robot_description
        repaired["robotDescription"] = robot_description
        repaired["robot_description"] = robot_description

    return repaired


def generate_cad_package(params: dict[str, Any]) -> CadGenerationResult:
    prompt = str(params.get("prompt") or "").strip()

    generation_mode = _normalize_generation_mode(
        params.get("generationMode") or params.get("generation_mode")
    )
    generation_pipeline = _normalize_generation_pipeline(
        params.get("generationPipeline") or params.get("generation_pipeline"),
        generation_mode,
    )

    try:
        client = _get_client()

        response = client.models.generate_content(
            contents=_build_user_prompt(params),
            config=GenerationConfig(
                system_instruction=_get_system_instruction(generation_mode),
                temperature=0.16,
            ),
        )

        raw_text = response.text or ""
        parsed_json = _extract_json_object(raw_text)
        repaired_json = _repair_package_payload(
            parsed_json,
            prompt,
            generation_mode,
            generation_pipeline,
        )
        cad_package = GeminiCadPackage.model_validate(repaired_json)

        return CadGenerationResult(
            ok=True,
            package=cad_package,
            raw_text=raw_text,
        )

    except ValidationError as exc:
        return CadGenerationResult(
            ok=False,
            raw_text=None,
            error=f"Gemini returned invalid CAD JSON: {exc}",
        )

    except Exception as exc:
        return CadGenerationResult(
            ok=False,
            raw_text=None,
            error=f"Gemini generation failed: {exc}",
        )


def build_safe_fallback_package(
    prompt: str,
    generation_mode: GenerationMode | None = None,
    generation_pipeline: GenerationPipeline | None = None,
) -> GeminiCadPackage:
    deterministic = build_deterministic_package(prompt)

    generation_mode = generation_mode or (
        "robot" if re.search(r"\b(robot|urdf|xacro|joint|link)\b", prompt, re.I) else "fast"
    )
    generation_pipeline = generation_pipeline or PIPELINE_BY_MODE[generation_mode]

    repaired = _repair_package_payload(
        deterministic,
        prompt,
        generation_mode,
        generation_pipeline,
    )

    return GeminiCadPackage.model_validate(repaired)

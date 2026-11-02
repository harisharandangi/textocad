from __future__ import annotations

import ast
import math
import re
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from textocad.services.cad_operations import (
    canonicalize_operation_name,
    supported_operation_names,
)


CadUnit = Literal["mm", "cm", "m", "inch"]
ModelType = Literal["part", "assembly", "robot"]
ParamType = Literal["number", "boolean", "enum", "string"]

SUPPORTED_OPERATION_TYPES = set(supported_operation_names())

OPERATION_TYPES = SUPPORTED_OPERATION_TYPES | {"unsupported"}

KEY_ALIASES = {
    "outerDiameter": "outer_diameter",
    "outer_diameter": "outer_diameter",
    "boreDiameter": "bore_diameter",
    "bore_diameter": "bore_diameter",
    "faceWidth": "face_width",
    "face_width": "face_width",
    "toothCount": "tooth_count",
    "tooth_count": "tooth_count",
    "helixAngle": "helix_angle",
    "helix_angle": "helix_angle",
    "baseLength": "base_length",
    "base_length": "base_length",
    "baseWidth": "base_width",
    "base_width": "base_width",
    "plateThickness": "plate_thickness",
    "plate_thickness": "plate_thickness",
    "verticalHeight": "vertical_height",
    "vertical_height": "vertical_height",
}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def snake_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in KEY_ALIASES:
        return KEY_ALIASES[raw]
    clean = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    clean = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", clean)
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", clean).strip("_").lower()
    return KEY_ALIASES.get(clean, clean)


def normalize_mapping_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {snake_key(key) or str(key): normalize_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_mapping_keys(item) for item in value]
    return value


class SafeExpressionError(ValueError):
    pass


class SafeExpressionEvaluator(ast.NodeVisitor):
    allowed_functions = {"min", "max", "clamp", "abs", "sin", "cos", "tan", "sqrt", "pow"}
    constants = {"pi": math.pi, "PI": math.pi}

    def __init__(self, params: Mapping[str, float]) -> None:
        self.params = dict(params)

    def evaluate(self, expression: str) -> float:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise SafeExpressionError(f"Invalid expression syntax: {expression}") from exc
        return float(self.visit(tree.body))

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise SafeExpressionError("Only numeric constants are allowed in expressions.")
        if not math.isfinite(float(node.value)):
            raise SafeExpressionError("Expression number must be finite.")
        return float(node.value)

    def visit_Name(self, node: ast.Name) -> float:
        key = node.id
        snake = snake_key(key)
        if key in self.constants:
            return float(self.constants[key])
        if key in self.params:
            return float(self.params[key])
        if snake in self.params:
            return float(self.params[snake])
        raise SafeExpressionError(f"Unknown parameter in expression: {key}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise SafeExpressionError("Unsupported unary expression operator.")

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if abs(right) < 1e-12:
                raise SafeExpressionError("Division by zero in expression.")
            return left / right
        raise SafeExpressionError("Unsupported expression operator.")

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name):
            raise SafeExpressionError("Only simple function calls are allowed.")
        name = node.func.id
        if name not in self.allowed_functions:
            raise SafeExpressionError(f"Function is not allowed in expressions: {name}")
        if node.keywords:
            raise SafeExpressionError("Keyword arguments are not allowed in expressions.")
        args = [self.visit(arg) for arg in node.args]
        if name == "min" and len(args) >= 1:
            return min(args)
        if name == "max" and len(args) >= 1:
            return max(args)
        if name == "clamp" and len(args) == 3:
            return min(max(args[0], args[1]), args[2])
        if name == "abs" and len(args) == 1:
            return abs(args[0])
        if name == "sin" and len(args) == 1:
            return math.sin(args[0])
        if name == "cos" and len(args) == 1:
            return math.cos(args[0])
        if name == "tan" and len(args) == 1:
            return math.tan(args[0])
        if name == "sqrt" and len(args) == 1:
            if args[0] < 0:
                raise SafeExpressionError("sqrt argument must be non-negative.")
            return math.sqrt(args[0])
        if name == "pow" and len(args) == 2:
            return math.pow(args[0], args[1])
        raise SafeExpressionError(f"Invalid arguments for {name}.")

    def generic_visit(self, node: ast.AST) -> float:
        raise SafeExpressionError(f"Unsafe expression node: {node.__class__.__name__}")


def param_context(params: list["ParamSpec"] | list[Mapping[str, Any]]) -> dict[str, float]:
    context: dict[str, float] = {}
    for item in params:
        if isinstance(item, ParamSpec):
            key = item.key
            value = item.value
        else:
            key = str(item.get("key") or "")
            value = item.get("value")
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            context[key] = float(value)
            context[snake_key(key)] = float(value)
    return context


def safe_eval_expression(expression: str, params: Mapping[str, float]) -> float:
    if len(expression) > 160:
        raise SafeExpressionError("Expression is too long.")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s]+", expression):
        raise SafeExpressionError("Expression contains unsupported characters.")
    return SafeExpressionEvaluator(params).evaluate(expression)


def resolve_number(value: Any, params: Mapping[str, float], fallback: float = 0.0) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = float(text)
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            pass
        return safe_eval_expression(text, params)
    return fallback


class ParamSpec(BaseModel):
    key: str
    label: str = ""
    type: ParamType = "number"
    value: str | float | int | bool = 0
    defaultValue: str | float | int | bool | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = "mm"
    group: str | None = None
    description: str | None = None
    locked: bool = False
    options: list[Any] | None = None

    model_config = {"extra": "allow"}

    @field_validator("key", mode="before")
    @classmethod
    def normalize_key(cls, value: Any) -> str:
        key = snake_key(value)
        key = re.sub(r"[^A-Za-z0-9_]+", "_", key).strip("_")
        if not key:
            key = "param"
        if not _IDENTIFIER_RE.match(key):
            key = f"param_{key}"
        return key

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> ParamType:
        clean = str(value or "number").strip().lower()
        if clean in {"int", "integer", "float", "double"}:
            return "number"
        if clean in {"bool"}:
            return "boolean"
        if clean in {"select"}:
            return "enum"
        if clean in {"text"}:
            return "string"
        if clean in {"number", "boolean", "enum", "string"}:
            return clean
        return "number"

    @model_validator(mode="after")
    def repair_values(self) -> "ParamSpec":
        if not self.label:
            self.label = self.key.replace("_", " ").title()
        if self.defaultValue is None:
            self.defaultValue = self.value
        if self.type == "number":
            numeric = resolve_number(self.value, {}, 0)
            self.value = numeric
            default_numeric = resolve_number(self.defaultValue, {}, numeric)
            self.defaultValue = default_numeric
            if self.min is None:
                self.min = 0
            if self.max is None:
                self.max = max(abs(numeric) * 3, numeric + 10, 100)
            if self.max <= self.min:
                self.max = self.min + max(abs(numeric), 1)
            self.value = min(max(float(self.value), self.min), self.max)
            self.defaultValue = min(max(float(self.defaultValue), self.min), self.max)
            if self.step is None or self.step <= 0:
                self.step = 1
        if self.type == "boolean":
            self.value = bool(self.value)
            self.defaultValue = bool(self.defaultValue)
            self.min = None
            self.max = None
            self.step = None
        if self.type == "string":
            self.value = str(self.value)
            self.defaultValue = str(self.defaultValue)
        if self.type == "enum" and not isinstance(self.options, list):
            self.options = []
        return self


class CadTransform(BaseModel):
    translate: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    rotate: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    scale: list[Any] = Field(default_factory=lambda: [1, 1, 1])
    mirror: list[Any] | None = None

    model_config = {"extra": "forbid"}


class CadOperation(BaseModel):
    id: str
    type: str
    visible: bool = True
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    properties: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    target: str | None = None
    transform: CadTransform = Field(default_factory=CadTransform)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, value: Any) -> str:
        raw = str(value or "").strip() or "op"
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._") or "op"

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: Any) -> str:
        clean = canonicalize_operation_name(value)
        return clean if clean in OPERATION_TYPES else "unsupported"

    @field_validator("params", "properties", mode="before")
    @classmethod
    def normalize_param_keys(cls, value: Any) -> dict[str, Any]:
        return normalize_mapping_keys(value) if isinstance(value, Mapping) else {}

    @model_validator(mode="before")
    @classmethod
    def mark_unsupported(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw_type = snake_key(value.get("type"))
        clean_type = canonicalize_operation_name(value.get("type"))
        if raw_type and clean_type not in OPERATION_TYPES:
            next_value = dict(value)
            metadata = dict(next_value.get("metadata") or {})
            metadata["unsupportedType"] = value.get("type")
            metadata["unsupportedReason"] = "Unknown operation type from model JSON."
            next_value["metadata"] = metadata
            next_value["type"] = "unsupported"
            return next_value
        return value


class RobotLink(BaseModel):
    name: str
    operations: list[CadOperation] = Field(default_factory=list)
    document: dict[str, Any] | None = None
    mass: float | None = None
    inertia: dict[str, Any] | None = None
    origin: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    rpy: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    visualMesh: str | None = None
    collisionMesh: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class RobotJoint(BaseModel):
    name: str
    type: Literal["fixed", "revolute", "prismatic", "continuous"] = "fixed"
    parent: str
    child: str
    origin: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    rpy: list[Any] = Field(default_factory=lambda: [0, 0, 0])
    axis: list[Any] = Field(default_factory=lambda: [0, 0, 1])
    limit: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("type", mode="before")
    @classmethod
    def normalize_joint_type(cls, value: Any) -> str:
        clean = str(value or "fixed").strip().lower()
        if clean in {"fixed", "revolute", "prismatic", "continuous"}:
            return clean
        return "fixed"


class RobotDescription(BaseModel):
    robotName: str = "textocad_robot"
    baseLink: str = "base_link"
    units: Literal["m", "mm"] = "m"
    links: list[RobotLink] = Field(default_factory=list)
    joints: list[RobotJoint] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class CadDocument(BaseModel):
    schemaVersion: Literal["textocad.v1"] = "textocad.v1"
    title: str = "TextoCAD Model"
    units: CadUnit = "mm"
    modelType: ModelType = "part"
    params: list[ParamSpec] = Field(default_factory=list)
    operations: list[CadOperation] = Field(default_factory=list)
    robot: RobotDescription | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("units", mode="before")
    @classmethod
    def normalize_units(cls, value: Any) -> CadUnit:
        clean = str(value or "mm").strip().lower()
        if clean in {"in", "inch", "inches"}:
            return "inch"
        if clean in {"mm", "cm", "m"}:
            return clean
        return "mm"

    @field_validator("modelType", mode="before")
    @classmethod
    def normalize_model_type(cls, value: Any) -> ModelType:
        clean = str(value or "part").strip().lower()
        if clean in {"assembly", "robot"}:
            return clean
        return "part"

    @model_validator(mode="after")
    def repair_metadata(self) -> "CadDocument":
        unsupported = [
            op.metadata.get("unsupportedType", op.type)
            for op in self.operations
            if op.type == "unsupported"
        ]
        metadata = dict(self.metadata or {})
        metadata.setdefault("unsupportedOperations", unsupported)
        metadata.setdefault("directLlmCodeExecution", False)
        metadata.setdefault("canonicalSource", "textocad_json")
        self.metadata = metadata
        return self


def validate_cad_document(data: Any) -> CadDocument:
    return CadDocument.model_validate(data)


def cad_document_to_dict(document: CadDocument) -> dict[str, Any]:
    return document.model_dump(mode="json", by_alias=False)


def validate_cad_document_strict(data: Any) -> tuple[CadDocument | None, list[str]]:
    try:
        return validate_cad_document(data), []
    except (ValidationError, SafeExpressionError) as exc:
        return None, [str(exc)]

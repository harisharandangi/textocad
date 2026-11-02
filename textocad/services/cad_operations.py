from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

import re

OperationCategory = Literal[
    "primitive",
    "sketch_2d",
    "solid_creation",
    "boolean",
    "transform",
    "feature",
    "pattern",
    "assembly",
    "selection",
    "export_import",
    "openscad_internal",
    "textocad_extension",
]

SupportLevel = Literal[
    "native",
    "approximate",
    "visual_only",
    "fallback_skip",
    "unsupported",
]

PipelineName = Literal["fast", "openscad", "cadquery", "robot"]


def snake_key(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    clean = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", raw)
    clean = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", clean)
    return re.sub(r"[^A-Za-z0-9_]+", "_", clean).strip("_").lower()


@dataclass(frozen=True)
class OperationSupport:
    fast: SupportLevel = "unsupported"
    openscad: SupportLevel = "unsupported"
    cadquery: SupportLevel = "unsupported"
    robot: SupportLevel = "unsupported"

    def for_pipeline(self, pipeline: PipelineName) -> SupportLevel:
        if pipeline == "openscad":
            return self.openscad
        if pipeline == "cadquery":
            return self.cadquery
        if pipeline == "robot":
            return self.robot
        return self.fast


@dataclass(frozen=True)
class OperationDefinition:
    name: str
    category: OperationCategory
    support: OperationSupport
    handler: str
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    validation_rules: tuple[str, ...] = ()
    error_message: str = ""
    warnings: tuple[str, ...] = ()
    test_example: Mapping[str, Any] = field(default_factory=dict)
    compiler_internal: bool = False


def _support(
    *,
    fast: SupportLevel = "unsupported",
    openscad: SupportLevel = "unsupported",
    cadquery: SupportLevel = "unsupported",
    robot: SupportLevel = "unsupported",
) -> OperationSupport:
    return OperationSupport(fast=fast, openscad=openscad, cadquery=cadquery, robot=robot)


def _op(
    name: str,
    category: OperationCategory,
    support: OperationSupport,
    handler: str | None = None,
    *,
    required: Iterable[str] = (),
    optional: Iterable[str] = (),
    rules: Iterable[str] = (),
    error: str = "",
    warnings: Iterable[str] = (),
    example: Mapping[str, Any] | None = None,
    compiler_internal: bool = False,
) -> OperationDefinition:
    return OperationDefinition(
        name=name,
        category=category,
        support=support,
        handler=handler or name,
        required_fields=tuple(required),
        optional_fields=tuple(optional),
        validation_rules=tuple(rules),
        error_message=error,
        warnings=tuple(warnings),
        test_example=dict(example or {"id": f"{name}_1", "type": name, "params": {}}),
        compiler_internal=compiler_internal,
    )


PRIMITIVE_OPERATIONS = [
    "box",
    "cube",
    "cylinder",
    "sphere",
    "cone",
    "wedge",
    "triangular_prism",
    "polyhedron",
    "surface",
    "text",
    "text_emboss",
    "import_mesh_placeholder",
    "import_step_placeholder",
    "import_dxf_placeholder",
]

SKETCH_2D_OPERATIONS = [
    "sketch",
    "sketch_start",
    "workplane",
    "center",
    "move_to",
    "move",
    "line_to",
    "line",
    "h_line",
    "h_line_to",
    "v_line",
    "v_line_to",
    "polar_line",
    "polar_line_to",
    "polyline",
    "spline",
    "parametric_curve",
    "three_point_arc",
    "sagitta_arc",
    "radius_arc",
    "tangent_arc_point",
    "rect",
    "circle",
    "ellipse",
    "ellipse_arc",
    "slot_2d",
    "wire",
    "close",
    "finalize",
    "offset_2d",
    "polygon",
    "regular_polygon",
    "trapezoid",
    "slot",
]

SOLID_CREATION_OPERATIONS = [
    "extrude",
    "cut_blind",
    "cut_thru_all",
    "linear_extrude",
    "revolve",
    "rotate_extrude",
    "sweep",
    "loft",
    "twist_extrude",
    "shell",
    "polygon_extrude",
]

BOOLEAN_OPERATIONS = [
    "union",
    "difference",
    "subtract",
    "cut",
    "intersect",
    "intersection",
    "combine",
    "split",
]

TRANSFORM_OPERATIONS = [
    "translate",
    "rotate",
    "rotate_about_center",
    "scale",
    "resize",
    "mirror",
    "mirror_x",
    "mirror_y",
    "multmatrix",
    "located",
    "moved",
    "copy",
]

FEATURE_OPERATIONS = [
    "hole",
    "cbore_hole",
    "csk_hole",
    "hole_pattern",
    "fillet",
    "chamfer",
    "offset",
    "offset_2d",
    "hull",
    "minkowski",
]

PATTERN_OPERATIONS = [
    "rarray",
    "polar_array",
    "parray",
    "pattern_linear",
    "pattern_circular",
    "distribute",
]

ASSEMBLY_OPERATIONS = [
    "assembly",
    "assembly_add",
    "constrain",
    "solve",
    "constraint",
    "color",
]

SELECTION_OPERATIONS = [
    "faces",
    "edges",
    "vertices",
    "wires",
    "solids",
    "shells",
    "compounds",
    "val",
    "vals",
    "all",
    "first",
    "last",
    "item",
    "end",
    "size",
    "add",
    "nearest_to_point_selector",
    "box_selector",
    "base_dir_selector",
    "parallel_dir_selector",
    "direction_selector",
    "direction_nth_selector",
    "length_nth_selector",
    "area_nth_selector",
    "radius_nth_selector",
    "perpendicular_dir_selector",
    "type_selector",
    "direction_min_max_selector",
    "center_nth_selector",
    "and_selector",
    "sum_selector",
    "subtract_selector",
    "inverse_selector",
    "string_syntax_selector",
]

EXPORT_IMPORT_OPERATIONS = [
    "to_svg",
    "export_svg",
    "import_step",
    "import_dxf",
    "export",
]

OPENSCAD_INTERNAL_OPERATIONS = [
    "for",
    "intersection_for",
    "if",
    "let",
    "assert",
    "echo",
    "render",
    "children",
    "concat",
    "lookup",
    "str",
    "chr",
    "ord",
    "search",
    "version",
    "version_num",
    "abs",
    "sign",
    "sin",
    "cos",
    "tan",
    "acos",
    "asin",
    "atan",
    "atan2",
    "floor",
    "round",
    "ceil",
    "ln",
    "len",
    "log",
    "pow",
    "sqrt",
    "exp",
    "rands",
    "min",
    "max",
    "norm",
    "cross",
]

TEXTOCAD_EXTENSION_OPERATIONS = [
    "gear_spur_approx",
    "gear_helical_approx",
]

OPERATION_ALIASES: dict[str, str] = {
    "cube": "box",
    "triangularprism": "triangular_prism",
    "polygonextrude": "polygon_extrude",
    "difference": "subtract",
    "cut": "subtract",
    "intersection": "intersect",
    "polararray": "pattern_circular",
    "polar_array": "pattern_circular",
    "rarray": "pattern_linear",
    "linear_pattern": "pattern_linear",
    "circular_pattern": "pattern_circular",
    "rotateaboutcenter": "rotate_about_center",
    "mirrorx": "mirror_x",
    "mirrory": "mirror_y",
    "cborehole": "cbore_hole",
    "cskhole": "csk_hole",
    "offset2d": "offset_2d",
    "slot2d": "slot_2d",
    "moveto": "move_to",
    "lineto": "line_to",
    "hline": "h_line",
    "hlineto": "h_line_to",
    "vline": "v_line",
    "vlineto": "v_line_to",
    "polarline": "polar_line",
    "polarlinto": "polar_line_to",
    "polarlineto": "polar_line_to",
    "threepointarc": "three_point_arc",
    "sagittaarc": "sagitta_arc",
    "radiusarc": "radius_arc",
    "tangentarcpoint": "tangent_arc_point",
    "regularpolygon": "regular_polygon",
    "cutblind": "cut_blind",
    "cutthruall": "cut_thru_all",
    "rotateextrude": "rotate_extrude",
    "twistextrude": "twist_extrude",
    "assemblyadd": "assembly_add",
    "tosvg": "to_svg",
    "exportsvg": "export_svg",
    "importstep": "import_step",
    "importdxf": "import_dxf",
    "nearesttopointselector": "nearest_to_point_selector",
    "boxselector": "box_selector",
    "basedirselector": "base_dir_selector",
    "paralleldirselector": "parallel_dir_selector",
    "directionselector": "direction_selector",
    "directionnthselector": "direction_nth_selector",
    "lengthnthselector": "length_nth_selector",
    "areanthselector": "area_nth_selector",
    "radiusnthselector": "radius_nth_selector",
    "perpendiculardirselector": "perpendicular_dir_selector",
    "typeselector": "type_selector",
    "directionminmaxselector": "direction_min_max_selector",
    "centernthselector": "center_nth_selector",
    "andselector": "and_selector",
    "sumselector": "sum_selector",
    "subtractselector": "subtract_selector",
    "inverseselector": "inverse_selector",
    "stringsyntaxselector": "string_syntax_selector",
}


def _build_registry() -> dict[str, OperationDefinition]:
    registry: dict[str, OperationDefinition] = {}

    for name in PRIMITIVE_OPERATIONS:
        registry[name] = _op(name, "primitive", _support(), "primitive")
    for name in SKETCH_2D_OPERATIONS:
        registry[name] = _op(name, "sketch_2d", _support(), "sketch")
    for name in SOLID_CREATION_OPERATIONS:
        registry[name] = _op(name, "solid_creation", _support(), "solid_creation")
    for name in BOOLEAN_OPERATIONS:
        registry[name] = _op(name, "boolean", _support(), "boolean")
    for name in TRANSFORM_OPERATIONS:
        registry[name] = _op(name, "transform", _support(), "transform")
    for name in FEATURE_OPERATIONS:
        registry[name] = _op(name, "feature", _support(), "feature")
    for name in PATTERN_OPERATIONS:
        registry[name] = _op(name, "pattern", _support(), "pattern")
    for name in ASSEMBLY_OPERATIONS:
        registry[name] = _op(name, "assembly", _support(), "assembly")
    for name in SELECTION_OPERATIONS:
        registry[name] = _op(name, "selection", _support(cadquery="native"), "selector")
    for name in EXPORT_IMPORT_OPERATIONS:
        registry[name] = _op(name, "export_import", _support(), "export_import")
    for name in OPENSCAD_INTERNAL_OPERATIONS:
        registry[name] = _op(
            name,
            "openscad_internal",
            _support(openscad="native"),
            "openscad_internal",
            compiler_internal=True,
        )
    for name in TEXTOCAD_EXTENSION_OPERATIONS:
        registry[name] = _op(name, "textocad_extension", _support(), "textocad_extension")

    primitive_native = {
        "box",
        "cube",
        "cylinder",
        "sphere",
        "cone",
        "wedge",
        "triangular_prism",
        "gear_spur_approx",
        "gear_helical_approx",
    }
    for name in primitive_native:
        registry[name] = _op(
            name,
            registry[name].category,
            _support(
                fast="native" if not name.startswith("gear_") else "approximate",
                openscad="native" if not name.startswith("gear_") else "approximate",
                cadquery="native" if not name.startswith("gear_") else "approximate",
                robot="visual_only",
            ),
            handler=OPERATION_ALIASES.get(name, name),
        )

    for name in {"polyhedron", "polygon", "linear_extrude", "rotate_extrude"}:
        registry[name] = _op(
            name,
            registry[name].category,
            _support(fast="approximate", openscad="native", cadquery="approximate", robot="visual_only"),
            handler=name,
        )

    registry["polygon_extrude"] = _op(
        "polygon_extrude",
        "solid_creation",
        _support(fast="native", openscad="native", cadquery="native", robot="visual_only"),
        "polygon_extrude",
    )

    for name in {"text", "text_emboss"}:
        registry[name] = _op(
            name,
            registry[name].category,
            _support(fast="visual_only", openscad="native", cadquery="approximate", robot="visual_only"),
            handler="text",
            warnings=("Browser preview uses a visual text placeholder.",),
        )

    for name in {"union", "combine"}:
        registry[name] = _op(
            name,
            "boolean",
            _support(fast="native", openscad="native", cadquery="native", robot="visual_only"),
            "union",
        )

    for name in {"subtract", "difference", "cut"}:
        registry[name] = _op(
            name,
            "boolean",
            _support(fast="approximate", openscad="native", cadquery="native", robot="visual_only"),
            "subtract",
            warnings=("Fast preview may show cutters or approximate the subtraction.",),
        )

    for name in {"intersect", "intersection"}:
        registry[name] = _op(
            name,
            "boolean",
            _support(fast="approximate", openscad="native", cadquery="native", robot="visual_only"),
            "intersect",
            warnings=("Fast preview may approximate intersections.",),
        )

    for name in {
        "translate",
        "rotate",
        "rotate_about_center",
        "scale",
        "resize",
        "mirror",
        "mirror_x",
        "mirror_y",
        "located",
        "moved",
        "copy",
    }:
        registry[name] = _op(
            name,
            "transform",
            _support(fast="native", openscad="native", cadquery="native", robot="visual_only"),
            OPERATION_ALIASES.get(name, name),
        )

    for name in {"hole", "hole_pattern"}:
        registry[name] = _op(
            name,
            "feature",
            _support(fast="visual_only", openscad="native", cadquery="native", robot="visual_only"),
            name,
            required=("diameter",),
            optional=("depth", "axis", "position", "positions", "target"),
        )

    for name in {"cbore_hole", "csk_hole"}:
        registry[name] = _op(
            name,
            "feature",
            _support(fast="visual_only", openscad="approximate", cadquery="native", robot="visual_only"),
            name,
        )

    for name in {"fillet", "chamfer"}:
        registry[name] = _op(
            name,
            "feature",
            _support(fast="fallback_skip", openscad="fallback_skip", cadquery="native", robot="fallback_skip"),
            name,
            warnings=(f"{name} is skipped in Fast/OpenSCAD fallback unless a specific safe implementation exists.",),
        )

    for name in {"shell", "sweep", "loft", "revolve", "extrude", "cut_blind", "cut_thru_all", "twist_extrude"}:
        registry[name] = _op(
            name,
            registry[name].category,
            _support(fast="fallback_skip", openscad="approximate", cadquery="native", robot="fallback_skip"),
            name,
        )

    for name in {"hull", "offset", "offset_2d"}:
        registry[name] = _op(
            name,
            registry[name].category,
            _support(fast="fallback_skip", openscad="native", cadquery="approximate", robot="fallback_skip"),
            name,
        )

    registry["minkowski"] = _op(
        "minkowski",
        "feature",
        _support(fast="fallback_skip", openscad="approximate", cadquery="unsupported", robot="fallback_skip"),
        "minkowski",
        warnings=("OpenSCAD minkowski is emitted only for small safe inputs.",),
    )

    for name in {"pattern_linear", "rarray", "pattern_circular", "polar_array", "parray", "distribute"}:
        registry[name] = _op(
            name,
            "pattern",
            _support(fast="native", openscad="native", cadquery="native", robot="visual_only"),
            OPERATION_ALIASES.get(name, name),
        )

    registry["color"] = _op(
        "color",
        "assembly",
        _support(fast="visual_only", openscad="native", cadquery="native", robot="visual_only"),
        "color",
    )

    for name in {"assembly", "assembly_add"}:
        registry[name] = _op(
            name,
            "assembly",
            _support(fast="visual_only", openscad="fallback_skip", cadquery="approximate", robot="native"),
            name,
        )

    for name in {"import_mesh_placeholder", "import_step_placeholder", "import_dxf_placeholder"}:
        registry[name] = _op(
            name,
            "primitive",
            _support(fast="visual_only", openscad="fallback_skip", cadquery="fallback_skip", robot="visual_only"),
            name,
        )

    return registry


OPERATION_REGISTRY: dict[str, OperationDefinition] = _build_registry()


_GEAR_SUPPORT = OperationSupport(
    fast="approximate", openscad="approximate", cadquery="native", robot="unsupported"
)
for _gear_name in ("spur_gear", "helical_gear"):
    OPERATION_REGISTRY[_gear_name] = OperationDefinition(
        name=_gear_name,
        category="solid_creation",
        support=_GEAR_SUPPORT,
        handler="gear_generator",
        required_fields=("module", "teeth"),
        optional_fields=("width", "bore_diameter", "helix_angle", "pressure_angle"),
    )

OPERATION_NAMES: frozenset[str] = frozenset(OPERATION_REGISTRY)


def canonicalize_operation_name(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw in OPERATION_REGISTRY:
        return raw
    clean = snake_key(raw)
    if clean in OPERATION_REGISTRY:
        return clean
    compact = clean.replace("_", "")
    return OPERATION_ALIASES.get(clean, OPERATION_ALIASES.get(compact, clean))


def is_known_operation(value: Any) -> bool:
    return canonicalize_operation_name(value) in OPERATION_REGISTRY


def get_operation_definition(value: Any) -> OperationDefinition | None:
    return OPERATION_REGISTRY.get(canonicalize_operation_name(value))


def get_operation_support(value: Any, pipeline: PipelineName) -> SupportLevel:
    definition = get_operation_definition(value)
    if definition is None:
        return "unsupported"
    return definition.support.for_pipeline(pipeline)


def supported_operation_names(*, include_internal: bool = False) -> list[str]:
    names = [
        name
        for name, definition in OPERATION_REGISTRY.items()
        if include_internal or not definition.compiler_internal
    ]
    return sorted(names)


def operation_support_summary(
    operations: Iterable[Any],
    pipeline: PipelineName,
) -> dict[str, list[str]]:
    summary: dict[str, set[str]] = {
        "native": set(),
        "approximate": set(),
        "skipped": set(),
        "unsupported": set(),
    }
    for raw_op in operations:
        if isinstance(raw_op, Mapping):
            op_type = raw_op.get("type")
        else:
            op_type = getattr(raw_op, "type", raw_op)
        name = canonicalize_operation_name(op_type)
        if not name:
            continue
        level = get_operation_support(name, pipeline)
        if level == "native":
            summary["native"].add(name)
        elif level in {"approximate", "visual_only"}:
            summary["approximate"].add(name)
        elif level == "fallback_skip":
            summary["skipped"].add(name)
        else:
            summary["unsupported"].add(name)
    return {key: sorted(value) for key, value in summary.items()}


def operation_summary_warning(summary: Mapping[str, list[str]]) -> str:
    native = ", ".join(summary.get("native", [])) or "none"
    approximate = ", ".join(summary.get("approximate", [])) or "none"
    skipped = ", ".join(summary.get("skipped", [])) or "none"
    unsupported = ", ".join(summary.get("unsupported", [])) or "none"
    return f"Native: {native}. Approximate: {approximate}. Skipped: {skipped}. Unsupported: {unsupported}."

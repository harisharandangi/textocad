from __future__ import annotations

from pathlib import Path
from typing import Any

from textocad.services.cad_schema import CadDocument
from textocad.services.robot_compiler import compile_urdf, compile_xacro, get_robot_description, write_robot_package


def build_urdf_artifacts(document: CadDocument, build_dir: Path, mesh_files: list[Path]) -> dict[str, Any]:
    build_dir.mkdir(parents=True, exist_ok=True)
    urdf_text = compile_urdf(document)
    xacro_text = compile_xacro(document)
    urdf_path = build_dir / "robot.urdf"
    xacro_path = build_dir / "robot.xacro"
    urdf_path.write_text(urdf_text, encoding="utf-8")
    xacro_path.write_text(xacro_text, encoding="utf-8")
    (build_dir / "model.urdf").write_text(urdf_text, encoding="utf-8")
    (build_dir / "model.xacro").write_text(xacro_text, encoding="utf-8")
    zip_path = write_robot_package(
        document,
        build_dir,
        urdf_text=urdf_text,
        xacro_text=xacro_text,
        mesh_files=mesh_files,
    )
    robot = get_robot_description(document)
    return {
        "urdfPath": str(urdf_path),
        "xacroPath": str(xacro_path),
        "zipPath": str(zip_path),
        "linkCount": len(robot.links),
        "jointCount": len(robot.joints),
        "warnings": [],
        "safeWorker": True,
        "safeFallbackUsed": False,
        "directLlmCodeExecution": False,
        "cadKernel": "robot_urdf",
    }

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from textocad.services.cad_schema import CadDocument, RobotDescription


def _slug(value: str, fallback: str = "robot") -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not clean:
        clean = fallback
    if clean[0].isdigit():
        clean = f"robot_{clean}"
    return clean


def _vec(values: list[Any] | None, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) < 3:
        return fallback
    out = []
    for index, default in enumerate(fallback):
        try:
            out.append(float(values[index]))
        except Exception:
            out.append(default)
    return (out[0], out[1], out[2])


def _fmt_vec(values: tuple[float, float, float]) -> str:
    return f"{values[0]:.6g} {values[1]:.6g} {values[2]:.6g}"


def get_robot_description(document: CadDocument) -> RobotDescription:
    if document.robot is not None:
        return document.robot
    return RobotDescription(
        robotName=_slug(document.title, "textocad_robot"),
        baseLink="base_link",
        units="m",
        links=[],
        joints=[],
    )


def compile_urdf(document: CadDocument, mesh_package_prefix: str = "package://textocad_robot/meshes") -> str:
    robot = get_robot_description(document)
    robot_name = _slug(robot.robotName or document.title, "textocad_robot")
    lines = [f'<robot name="{robot_name}">']

    for link in robot.links:
        name = _slug(link.name, "link")
        mass = float(link.mass or 1.0)
        visual_mesh = link.visualMesh or f"meshes/{name}.stl"
        mesh_uri = f"{mesh_package_prefix}/{Path(visual_mesh).name}"
        lines.append(f'  <link name="{name}">')
        lines.append("    <visual>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append("      <geometry>")
        lines.append(f'        <mesh filename="{mesh_uri}" scale="1 1 1"/>')
        lines.append("      </geometry>")
        lines.append("    </visual>")
        lines.append("    <collision>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append("      <geometry>")
        lines.append(f'        <mesh filename="{mesh_uri}" scale="1 1 1"/>')
        lines.append("      </geometry>")
        lines.append("    </collision>")
        lines.append("    <inertial>")
        lines.append(f'      <mass value="{mass:.6g}"/>')
        lines.append('      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>')
        lines.append("    </inertial>")
        lines.append("  </link>")

    for joint in robot.joints:
        name = _slug(joint.name, "joint")
        joint_type = joint.type
        origin = _vec(joint.origin, (0, 0, 0))
        rpy = _vec(joint.rpy, (0, 0, 0))
        axis = _vec(joint.axis, (0, 0, 1))
        lines.append(f'  <joint name="{name}" type="{joint_type}">')
        lines.append(f'    <parent link="{_slug(joint.parent, "parent_link")}"/>')
        lines.append(f'    <child link="{_slug(joint.child, "child_link")}"/>')
        lines.append(f'    <origin xyz="{_fmt_vec(origin)}" rpy="{_fmt_vec(rpy)}"/>')
        if joint_type in {"revolute", "continuous", "prismatic"}:
            lines.append(f'    <axis xyz="{_fmt_vec(axis)}"/>')
        if joint_type in {"revolute", "prismatic"}:
            limit = joint.limit or {}
            lower = float(limit.get("lower", -1.57))
            upper = float(limit.get("upper", 1.57))
            effort = float(limit.get("effort", 10))
            velocity = float(limit.get("velocity", 1))
            lines.append(
                f'    <limit lower="{lower:.6g}" upper="{upper:.6g}" effort="{effort:.6g}" velocity="{velocity:.6g}"/>'
            )
        lines.append("  </joint>")

    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def compile_xacro(document: CadDocument) -> str:
    urdf = compile_urdf(document)
    body = "\n".join(urdf.splitlines()[1:-1])
    robot = get_robot_description(document)
    return (
        '<?xml version="1.0"?>\n'
        f'<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="{_slug(robot.robotName, "textocad_robot")}">\n'
        f"{body}\n"
        "</robot>\n"
    )


def write_robot_package(
    document: CadDocument,
    build_dir: Path,
    *,
    urdf_text: str,
    xacro_text: str,
    mesh_files: list[Path],
) -> Path:
    robot = get_robot_description(document)
    package_name = _slug(robot.robotName or document.title, "textocad_robot")
    package_root = build_dir / package_name
    (package_root / "urdf").mkdir(parents=True, exist_ok=True)
    (package_root / "xacro").mkdir(parents=True, exist_ok=True)
    (package_root / "meshes").mkdir(parents=True, exist_ok=True)
    (package_root / "launch").mkdir(parents=True, exist_ok=True)

    (package_root / "package.xml").write_text(
        f"""<?xml version="1.0"?>
<package format="3">
  <name>{package_name}</name>
  <version>0.1.0</version>
  <description>TextoCAD generated robot description package.</description>
  <maintainer email="dev@textocad.local">TextoCAD</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz2</exec_depend>
</package>
""",
        encoding="utf-8",
    )
    (package_root / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.8)
project({package_name})
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf xacro meshes launch DESTINATION share/${{PROJECT_NAME}})
ament_package()
""",
        encoding="utf-8",
    )
    (package_root / "urdf" / "robot.urdf").write_text(urdf_text, encoding="utf-8")
    (package_root / "xacro" / "robot.xacro").write_text(xacro_text, encoding="utf-8")
    (package_root / "urdf" / f"{package_name}.urdf").write_text(urdf_text, encoding="utf-8")
    (package_root / "xacro" / f"{package_name}.urdf.xacro").write_text(xacro_text, encoding="utf-8")
    (package_root / "README.md").write_text(
        "# TextoCAD Robot Package\n\nGenerated from canonical TextoCAD robot JSON. Mesh files are per-link exports.\n",
        encoding="utf-8",
    )
    (package_root / "launch" / "display.launch.py").write_text(
        """from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen'),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', output='screen'),
    ])
""",
        encoding="utf-8",
    )
    (package_root / "textocad_robot.json").write_text(
        json.dumps(document.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    for mesh_file in mesh_files:
        target = package_root / "meshes" / mesh_file.name
        if mesh_file.exists() and target.resolve() != mesh_file.resolve():
            target.write_bytes(mesh_file.read_bytes())

    zip_path = build_dir / "model.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(build_dir).as_posix())
    return zip_path

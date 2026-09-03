"""MJCF mechanical inventory used as the IsaacLab asset source."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XML = ROOT / "src/mjlab_microduck/robot/microduck/robot_walk.xml"

ACTUATED_ORDER = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def build_report(xml_path: Path = DEFAULT_XML) -> dict[str, object]:
    root = ET.parse(xml_path).getroot()
    joints: list[dict[str, object]] = []
    for joint in root.iter("joint"):
        if joint.get("name") is None:
            continue
        item: dict[str, object] = {"name": joint.attrib["name"], "type": joint.get("type", "hinge")}
        if axis := joint.get("axis"):
            item["axis"] = [float(value) for value in axis.split()]
        if limit := joint.get("range"):
            item["range"] = [float(value) for value in limit.split()]
        joints.append(item)
    return {
        "source": str(xml_path.relative_to(ROOT)),
        "actuated_joints": [joint for joint in joints if joint["name"] in ACTUATED_ORDER],
        "all_named_joints": joints,
        "actuated_joint_order": list(ACTUATED_ORDER),
    }

#!/usr/bin/env python3
"""Drive the official Rust controller from a persistent MuJoCo process.

This is a transport/evidence harness only. Policy selection, command smoothing,
fall handling, gains, and target filtering remain in ``robotd``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, MICRODUCK_ROLLERS_XML, MICRODUCK_XML, PolicyInference  # noqa: E402

OFFICIAL_JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
MOUTH_INDEX = 9
CONTROL_HZ = 50
PHYSICS_STEPS = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_frame(data: mujoco.MjData, policy: PolicyInference, twist: list[float], *, init: bool = False, shutdown: bool = False) -> dict[str, Any]:
    positions = np.zeros(15, dtype=np.float64)
    velocities = np.zeros(15, dtype=np.float64)
    positions[:MOUTH_INDEX] = data.qpos[policy.joint_qpos_indices[:MOUTH_INDEX]]
    positions[MOUTH_INDEX + 1 :] = data.qpos[policy.joint_qpos_indices[MOUTH_INDEX:]]
    velocities[:MOUTH_INDEX] = data.qvel[policy.joint_qvel_indices[:MOUTH_INDEX]]
    velocities[MOUTH_INDEX + 1 :] = data.qvel[policy.joint_qvel_indices[MOUTH_INDEX:]]
    rotation = data.xmat[policy.trunk_body_id].reshape(3, 3)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    quat = data.xquat[policy.trunk_body_id]
    gyro = data.qvel[policy.trunk_dof_indices[:3]] if hasattr(policy, "trunk_dof_indices") else np.zeros(3)
    return {
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "currents_ma": [0.0] * 15,
        "gravity": gravity.tolist(),
        "gyro": np.asarray(gyro, dtype=float).tolist(),
        "quat": quat.tolist(),
        "twist": twist,
        "enabled": True,
        "init": init,
        "shutdown": shutdown,
    }


def read_json_line(stream, timeout: float = 3.0) -> dict[str, Any]:
    ready, _, _ = select.select([stream], [], [], timeout)
    if not ready:
        raise TimeoutError("official robotd produced no replay output")
    line = stream.readline()
    if not line:
        raise RuntimeError("official robotd exited before replay output")
    return json.loads(line)


def make_params(path: Path, policy: Path, roller: bool) -> None:
    mode = "roller" if roller else "walk"
    stand = "none" if roller else "/tmp/microduck-fork/policies/alpha_stand.onnx"
    text = f"""[policy]
enabled = true
mode = \"{mode}\"
walk = \"{policy}\"
stand = \"{stand}\"
sitstand = \"none\"
ground_pick = \"none\"
kick_left = \"none\"
kick_right = \"none\"
roulade = \"none\"

[control]
cmd_alpha = 1.0

[safety]
deadman_ms = 10000
"""
    path.write_text(text, encoding="utf-8")


def run_profile(args: argparse.Namespace, roller: bool) -> dict[str, Any]:
    xml = ROOT / (MICRODUCK_ROLLERS_XML if roller else MICRODUCK_XML)
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    policy_path = args.roller_policy if roller else args.walk_policy
    policy = PolicyInference(model, data, walking_onnx_path=str(policy_path), new_cmd_obs=True, use_projected_gravity=True)
    policy.trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    free_q = int(model.jnt_qposadr[free_id])
    data.qpos[free_q : free_q + 3] = [0.0, 0.0, 0.1385 if roller else 0.125]
    for i, qpos in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos] = DEFAULT_POSE[i]
    data.ctrl[:] = DEFAULT_POSE
    mujoco.mj_forward(model, data)
    start = data.xpos[policy.trunk_body_id].copy()
    executable = args.robotd.resolve()
    with tempfile.TemporaryDirectory(prefix="microduck-replay-") as temp:
        params = Path(temp) / "robotd.toml"
        make_params(params, policy_path.resolve(), roller)
        process = subprocess.Popen(
            [str(executable), "--replay-stdin", "--params", str(params)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        assert process.stdin is not None and process.stdout is not None
        outputs: list[dict[str, Any]] = []
        labels: list[str] = []
        tilt_trace: list[float] = []
        official_fall_seen = False
        official_recovery_seen = False
        gains: list[int] = []
        targets: list[list[float]] = []
        max_tilt = 0.0
        physical_fall_seen = False
        physical_recovery_seen = False
        contacts: set[str] = set()
        twists = ([0.0, 0.0, 0.0] if roller else [0.12, 0.0, 0.0], [0.0, 0.0, 0.0])
        ticks = args.ticks
        stop_sender = threading.Event()
        data_lock = threading.Lock()
        target_count = 0

        def pump_input() -> None:
            nonlocal target_count
            # The official RobotIo read is intentionally blocking. Keep it fed independently so
            # startup/bring-up ticks (which may not write a target) cannot deadlock the harness.
            try:
                for tick in range(ticks + 16):
                    if stop_sender.is_set():
                        break
                    with data_lock:
                        moving = target_count < max(0, ticks - args.zero_tail_ticks)
                        frame = official_frame(data, policy, list(twists[0] if moving else twists[1]), init=tick == 0)
                    process.stdin.write(json.dumps(frame) + "\n")
                    process.stdin.flush()
                    time.sleep(0.02)
                with data_lock:
                    frame = official_frame(data, policy, [0.0, 0.0, 0.0], shutdown=True)
                process.stdin.write(json.dumps(frame) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, ValueError):
                pass

        sender = threading.Thread(target=pump_input, daemon=True)
        sender.start()
        try:
            while len(targets) < ticks:
                message = read_json_line(process.stdout)
                outputs.append(message)
                if message.get("type") == "gain":
                    gains.append(int(message["kp"]))
                elif message.get("type") == "state":
                    state = message["state"]
                    labels.append(str(state.get("policy", "")))
                    safety = state.get("safety", {})
                    official_fall_seen = official_fall_seen or bool(safety.get("fallen", False))
                    official_recovery_seen = official_recovery_seen or (
                        official_fall_seen and not bool(safety.get("fallen", False))
                    )
                elif message.get("type") == "target":
                    target = [float(x) for x in message["positions"]]
                    if len(target) != 15 or not np.isfinite(target).all():
                        raise ValueError("official target is not finite 15D")
                    targets.append(target)
                    target_count = len(targets)
                    servo_target = np.asarray(target, dtype=np.float64)
                    with data_lock:
                        data.ctrl[:] = np.concatenate((servo_target[:MOUTH_INDEX], servo_target[MOUTH_INDEX + 1 :]))
                        for _ in range(PHYSICS_STEPS):
                            mujoco.mj_step(model, data)
                        rotation = data.xmat[policy.trunk_body_id].reshape(3, 3)
                        tilt = math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
                        tilt_trace.append(tilt)
                        max_tilt = max(max_tilt, tilt)
                        physical_fall_seen = physical_fall_seen or tilt > math.radians(65.0)
                        physical_recovery_seen = physical_recovery_seen or (
                            physical_fall_seen and tilt < math.radians(35.0)
                        )
                        for contact in data.contact[: data.ncon]:
                            for geom in (contact.geom1, contact.geom2):
                                contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or str(geom))
        finally:
            stop_sender.set()
            sender.join(timeout=2)
            process.stdin.close()
            process.wait(timeout=10)
        final_pos = data.xpos[policy.trunk_body_id].copy()
        return {
            "schema_version": 1,
            "profile": "rollers" if roller else "walk_all_collisions",
            "scene": str(xml),
            "official_commit": args.official_commit,
            "bridge": "robotd --replay-stdin",
            "policy_sha256": sha256(policy_path),
            "ticks": len(targets),
            "labels": sorted(set(labels)),
            "label_trace": labels,
            "tilt_trace_rad": tilt_trace,
            "gains": sorted(set(gains)),
            "target_count": len(targets),
            "max_target_discontinuity": float(np.max(np.abs(np.diff(np.asarray(targets), axis=0)))) if len(targets) > 1 else 0.0,
            "max_tilt_rad": max_tilt,
            "world_displacement_m": (final_pos - start).tolist(),
            "contact_geoms": sorted(contacts),
            "fall_seen": physical_fall_seen or official_fall_seen,
            "physical_fall_seen": physical_fall_seen,
            "official_fall_seen": official_fall_seen,
            "recovery_seen": physical_recovery_seen and official_recovery_seen,
            "physical_recovery_seen": physical_recovery_seen,
            "official_recovery_seen": official_recovery_seen,
            "reset_count": 0,
            "passed": bool(targets) and process.returncode == 0 and (
                not physical_fall_seen or (physical_recovery_seen and official_recovery_seen)
            ),
            "failure_reason": None if (
                not physical_fall_seen or (physical_recovery_seen and official_recovery_seen)
            ) else "mujoco_fall_without_recovery",
            "unsupported_edges": ["walk_to_rollers_cross_profile"],
            "raw_output_count": len(outputs),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotd", type=Path, required=True)
    parser.add_argument("--walk-policy", type=Path, default=Path("/tmp/microduck-fork/policies/alpha_walking.onnx"))
    parser.add_argument("--roller-policy", type=Path, default=Path("/tmp/microduck-fork/policies/roller.onnx"))
    parser.add_argument("--official-commit", default="66d4fa8")
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--zero-tail-ticks", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    if args.zero_tail_ticks < 0 or args.zero_tail_ticks >= args.ticks:
        parser.error("--zero-tail-ticks must be non-negative and smaller than --ticks")
    reports = [run_profile(args, roller=False), run_profile(args, roller=True)]
    result = {"schema_version": 1, "mode": "smoke" if args.ticks <= 20 else "full", "profiles": reports, "passed": all(r["passed"] for r in reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

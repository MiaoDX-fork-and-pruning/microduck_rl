#!/usr/bin/env python3
"""Run the accepted specialist switching scenarios in one continuous MuJoCo episode."""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, MICRODUCK_ROLLERS_XML, MICRODUCK_XML, PolicyInference
from specialist_scenario import load_scenario, scenario_events

PATHS = {p.name: str(p / "policy.onnx") for p in (ROOT / "artifacts/specialists").iterdir() if (p / "policy.onnx").exists()}
TILT_FAILURE = math.radians(65)


def run(scenario_path: Path, roller: bool, output: Path) -> dict:
    scenario, frames = load_scenario(scenario_path)
    model = mujoco.MjModel.from_xml_path(str(ROOT / (MICRODUCK_ROLLERS_XML if roller else MICRODUCK_XML)))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    kwargs = {"new_cmd_obs": True}
    mapping = {"velocity_flat":"walking_onnx_path", "velstand_flat":"standing_onnx_path", "sitstand_flat":"sitstand_onnx_path", "ground_pick_flat":"ground_pick_onnx_path", "ball_kick_flat":"kick_right_onnx_path", "roulade_flat":"roulade_onnx_path", "velocity_rollers":"walking_onnx_path", "roller_crouch":"roller_crouch_onnx_path"}
    for pid in dict.fromkeys(f.policy_id for f in frames):
        if pid not in PATHS or pid not in mapping: raise ValueError(f"missing compatible policy artifact: {pid}")
        kwargs[mapping[pid]] = PATHS[pid]
    policy = PolicyInference(model, data, **kwargs)
    policy.validate_specialist_policies(f.policy_id for f in frames)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    q = int(model.jnt_qposadr[free]); data.qpos[q:q+3] = [0, 0, 0.1385 if roller else 0.125]; data.qpos[q+3:q+7] = [1,0,0,0]
    for i, idx in enumerate(policy.joint_qpos_indices): data.qpos[idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose; mujoco.mj_forward(model, data)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    start = data.xpos[trunk].copy(); events = {f.step:f for f in scenario_events(frames)}
    segment = {}; labels=[]; tilts=[]; actions=[]; contacts=set(); previous=None; transitions=[]; finite=True
    for step, frame in enumerate(frames):
        if step in events:
            if previous is not None: transitions[-1]["end_step"] = step - 1
            policy.activate_specialist_policy(events[step].policy_id, events[step].command)
            previous = events[step].policy_id
            transitions.append({"step": step, "policy_id": previous, "command": list(events[step].command), "start_step": step})
            segment.setdefault(previous, {"frames": 0, "max_tilt_rad": 0.0, "max_action_jump": 0.0})
        obs = policy.get_observations(); previous_action = policy.last_action.copy(); action = policy.infer()
        finite = finite and bool(np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite: break
        jump = float(np.max(np.abs(action - previous_action)))
        policy.apply_action(action)
        for _ in range(4): mujoco.mj_step(model, data)
        quat = data.xquat[trunk]; tilt = float(2 * math.acos(np.clip(abs(float(quat[0])), 0, 1)))
        tilts.append(tilt); actions.append(action); labels.append(frame.policy_id)
        item = segment[frame.policy_id]; item["frames"] += 1; item["max_tilt_rad"] = max(item["max_tilt_rad"], tilt); item["max_action_jump"] = max(item["max_action_jump"], jump)
        policy.last_action = np.asarray(action, dtype=np.float32).copy()
        for c in data.contact[:data.ncon]:
            for g in (c.geom1, c.geom2): contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g))
    if transitions: transitions[-1]["end_step"] = len(labels) - 1
    displacement = (data.xpos[trunk] - start).tolist()
    report = {"schema_version": 1, "scenario": str(scenario_path), "track": scenario["track"], "scene": "rollers" if roller else "walk_all_collisions", "seed": scenario["seed"], "frames": len(labels), "expected_frames": len(frames), "reset_count": 0, "finite": finite, "policy_sequence": list(dict.fromkeys(labels)), "transitions": transitions, "segments": segment, "max_tilt_rad": max(tilts, default=None), "world_displacement_m": displacement, "contact_geoms": sorted(contacts), "max_action_jump": max((float(np.max(np.abs(actions[i]-actions[i-1]))) for i in range(1,len(actions))), default=0.0)}
    report["stability_gate_rad"] = TILT_FAILURE
    report["passed"] = finite and len(labels) == len(frames) and report["reset_count"] == 0 and len(report["policy_sequence"]) == len(set(f.policy_id for f in frames)) and report["max_tilt_rad"] < TILT_FAILURE
    report["failure_reason"] = None if report["passed"] else ("nonfinite_state_or_action" if not finite else "physical_fall_during_switching_sequence" if report["max_tilt_rad"] >= TILT_FAILURE else "incomplete_or_missing_policy_transition")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--track", choices=("A","B"), required=True); ap.add_argument("--output", type=Path, required=True); a=ap.parse_args()
    scenario = ROOT / ("docs/specialist_demo_scenario.json" if a.track == "A" else "docs/specialist_demo_track_b_scenario.json")
    result = run(scenario, a.track == "B", a.output); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__": main()

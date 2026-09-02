#!/usr/bin/env python3
"""Run the accepted specialist switching scenarios in one continuous MuJoCo episode."""
from __future__ import annotations

import argparse, json, math, sys
import re
from pathlib import Path
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, MICRODUCK_BALL_XML, MICRODUCK_ROLLERS_XML, MICRODUCK_XML, PolicyInference
from specialist_scenario import load_scenario, scenario_events

PATHS = {p.name: str(p / "policy.onnx") for p in (ROOT / "artifacts/specialists").iterdir() if (p / "policy.onnx").exists()}
TILT_FAILURE = math.radians(65)


def run(scenario_path: Path, roller: bool, output: Path, video: Path | None = None) -> dict:
    scenario, frames = load_scenario(scenario_path)
    scene_id = scenario["compatibility"]["scene"]
    xml = MICRODUCK_ROLLERS_XML if roller else MICRODUCK_BALL_XML if "ball" in scene_id else MICRODUCK_XML
    model = mujoco.MjModel.from_xml_path(str(ROOT / xml))
    model.opt.timestep = 0.005
    if roller:
        # Match the rollers deployment rehearsal: passive wheel bearings have
        # no XML friction and receive this runtime friction explicitly.
        for joint_id in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name and re.match(r"^passive_.*", name):
                model.dof_frictionloss[model.jnt_dofadr[joint_id]] = 0.003
    data = mujoco.MjData(model)
    # Specialist artifacts were evaluated with projected gravity. Keep the
    # switching harness on the same observation contract; raw accelerometer
    # values are a materially different 3D input for these ONNX policies.
    kwargs = {"new_cmd_obs": True, "use_projected_gravity": True}
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
    writer = imageio.get_writer(str(video), fps=50, codec="libx264", quality=7) if video else None
    renderer = mujoco.Renderer(model, height=368, width=640) if video else None
    camera = mujoco.MjvCamera() if video else None
    if camera is not None:
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 1.0
        camera.azimuth = 135.0
        camera.elevation = -18.0
        camera.lookat[:] = data.xpos[trunk]
    font = ImageFont.load_default() if video else None
    start = data.xpos[trunk].copy(); events = {f.step:f for f in scenario_events(frames)}
    segment = {}; labels=[]; tilts=[]; actions=[]; contacts=set(); previous=None; transitions=[]; finite=True
    path_length = 0.0
    previous_position = data.xpos[trunk].copy()
    first_failure_step = None
    ball_start = None
    try:
      for step, frame in enumerate(frames):
        if step in events:
            if previous is not None: transitions[-1]["end_step"] = step - 1
            if previous is not None:
                transitions[-1]["end_position_m"] = data.xpos[trunk].tolist()
            policy.activate_specialist_policy(events[step].policy_id, events[step].command)
            if events[step].policy_id == "ball_kick_flat" and policy.ball_qpos_adr is not None:
                ball_start = data.qpos[policy.ball_qpos_adr : policy.ball_qpos_adr + 3].copy()
            previous = events[step].policy_id
            transitions.append({"step": step, "policy_id": previous, "command": list(events[step].command), "start_step": step})
            segment.setdefault(previous, {"frames": 0, "max_tilt_rad": 0.0, "max_action_jump": 0.0, "start_position_m": data.xpos[trunk].tolist()})
        # Phase-conditioned episodic policies use the same command contract as
        # deployment: phase 0 starts at the handoff and advances continuously.
        # A zero block is only the event seed, not the runtime command.
        active = transitions[-1]
        elapsed_s = (step - active["start_step"]) / 50.0
        if frame.policy_id == "roller_crouch":
            phase = elapsed_s / 5.0
            policy.command[:3] = (math.cos(2.0 * math.pi * phase), math.sin(2.0 * math.pi * phase), 0.0)
        elif frame.policy_id == "ground_pick_flat":
            phase = min(elapsed_s / 4.0, 0.7)
            policy.command[:3] = (math.cos(2.0 * math.pi * phase), math.sin(2.0 * math.pi * phase), 0.0)
        obs = policy.get_observations(); previous_action = policy.last_action.copy(); action = policy.infer()
        finite = finite and bool(np.isfinite(obs).all() and np.isfinite(action).all() and np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if not finite: break
        jump = float(np.max(np.abs(action - previous_action)))
        policy.apply_action(action)
        for _ in range(4): mujoco.mj_step(model, data)
        quat = data.xquat[trunk]; tilt = float(2 * math.acos(np.clip(abs(float(quat[0])), 0, 1)))
        current_position = data.xpos[trunk].copy()
        path_length += float(np.linalg.norm(current_position - previous_position))
        previous_position = current_position
        tilts.append(tilt); actions.append(action); labels.append(frame.policy_id)
        item = segment[frame.policy_id]; item["frames"] += 1; item["max_tilt_rad"] = max(item["max_tilt_rad"], tilt); item["max_action_jump"] = max(item["max_action_jump"], jump)
        item["end_position_m"] = current_position.tolist()
        if tilt >= TILT_FAILURE and first_failure_step is None:
            first_failure_step = step
        if renderer is not None:
            camera.lookat[:] = data.xpos[trunk]
            renderer.update_scene(data, camera=camera)
            frame_image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame_image, "RGBA")
            command = policy.command
            if frame.policy_id in {"velocity_flat", "velocity_rollers"}:
                command_text = f"velocity={float(command[0]):+.2f} m/s"
            elif frame.policy_id in {"ground_pick_flat", "roller_crouch"}:
                command_text = f"phase=({float(command[0]):+.2f}, {float(command[1]):+.2f})"
            elif frame.policy_id == "sitstand_flat":
                command_text = "command=SIT" if float(command[0]) >= 0.5 else "command=STAND"
            else:
                command_text = "command=zero"
            lines = [
                f"{frame.policy_id}",
                f"t={step / 50.0:05.1f}s  {command_text}",
                f"tilt={math.degrees(tilt):.1f} deg  action_jump={jump:.2f}",
            ]
            box = (12, 12, 365, 78)
            draw.rounded_rectangle(box, radius=6, fill=(0, 0, 0, 185))
            y = 18
            for line in lines:
                draw.text((22, y), line, font=font, fill=(255, 255, 255, 255))
                y += 18
            writer.append_data(np.asarray(frame_image))
        policy.last_action = np.asarray(action, dtype=np.float32).copy()
        for c in data.contact[:data.ncon]:
            for g in (c.geom1, c.geom2): contacts.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or str(g))
    finally:
      if writer is not None:
        writer.close()
      if renderer is not None:
        renderer.close()
    if transitions:
        transitions[-1]["end_step"] = len(labels) - 1
        transitions[-1]["end_position_m"] = data.xpos[trunk].tolist()
    displacement = (data.xpos[trunk] - start).tolist()
    dynamic_policy_ids = {"roulade_flat", "ball_kick_flat"}
    non_dynamic_tilts = [tilt for label, tilt in zip(labels, tilts) if label not in dynamic_policy_ids]
    last_dynamic_start = max(
        (t["start_step"] for t in transitions if t["policy_id"] in dynamic_policy_ids),
        default=None,
    )
    recovery_start = next(
        (t["start_step"] for t in transitions
         if t["policy_id"] == "velstand_flat"
         and last_dynamic_start is not None
         and t["start_step"] > last_dynamic_start),
        None,
    )
    recovery_max_tilt = (max(tilts[recovery_start:], default=None) if recovery_start is not None else None)
    stable_window_steps = 50
    recovery_stable_step = None
    if recovery_start is not None:
        for candidate in range(recovery_start, max(recovery_start, len(tilts) - stable_window_steps + 1)):
            if max(tilts[candidate : candidate + stable_window_steps], default=math.inf) < TILT_FAILURE:
                recovery_stable_step = candidate
                break
    recovery_settled_max_tilt = (
        max(tilts[recovery_stable_step:], default=None)
        if recovery_stable_step is not None else None
    )
    ball_displacement = None
    if ball_start is not None and policy.ball_qpos_adr is not None:
        ball_displacement = (
            data.qpos[policy.ball_qpos_adr : policy.ball_qpos_adr + 3] - ball_start
        ).tolist()
    for transition in transitions:
        start_step = transition["start_step"]
        end_step = transition["end_step"]
        transition_tilts = tilts[start_step : end_step + 1]
        transition["duration_s"] = len(transition_tilts) / 50.0
        transition["max_tilt_rad"] = max(transition_tilts, default=None)
    report = {"schema_version": 3, "scenario": str(scenario_path), "track": scenario["track"], "scene": scene_id, "seed": scenario["seed"], "frames": len(labels), "expected_frames": len(frames), "reset_count": 0, "finite": finite, "policy_sequence": list(dict.fromkeys(labels)), "transitions": transitions, "segments": segment, "max_tilt_rad": max(tilts, default=None), "max_non_dynamic_tilt_rad": max(non_dynamic_tilts, default=None), "recovery_max_tilt_rad": recovery_max_tilt, "recovery_stable_step": recovery_stable_step, "recovery_time_s": ((recovery_stable_step - recovery_start) / 50.0 if recovery_stable_step is not None and recovery_start is not None else None), "recovery_stable_window_s": stable_window_steps / 50.0, "recovery_settled_max_tilt_rad": recovery_settled_max_tilt, "first_failure_step": first_failure_step, "world_displacement_m": displacement, "path_length_m": path_length, "ball_displacement_m": ball_displacement, "contact_geoms": sorted(contacts), "max_action_jump": max((float(np.max(np.abs(actions[i]-actions[i-1]))) for i in range(1,len(actions))), default=0.0), "video": str(video) if video else None}
    report["stability_gate_rad"] = TILT_FAILURE
    non_recovery_non_dynamic_tilts = [
        tilt for step, (label, tilt) in enumerate(zip(labels, tilts))
        if label not in dynamic_policy_ids
        and (recovery_start is None or step < recovery_start or recovery_stable_step is not None and step >= recovery_stable_step)
    ]
    report["max_gated_non_dynamic_tilt_rad"] = max(non_recovery_non_dynamic_tilts, default=None)
    physical_ok = report["max_gated_non_dynamic_tilt_rad"] is None or report["max_gated_non_dynamic_tilt_rad"] < TILT_FAILURE
    recovery_ok = recovery_start is None or recovery_stable_step is not None
    report["passed"] = finite and len(labels) == len(frames) and report["reset_count"] == 0 and len(report["policy_sequence"]) == len(set(f.policy_id for f in frames)) and physical_ok and recovery_ok
    report["failure_reason"] = None if report["passed"] else ("nonfinite_state_or_action" if not finite else "physical_fall_outside_dynamic_action" if not physical_ok else "failed_post_action_recovery" if not recovery_ok else "incomplete_or_missing_policy_transition")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--track", choices=("A","B"), required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--video", type=Path); a=ap.parse_args()
    scenario = ROOT / ("docs/specialist_showcase_track_a.json" if a.track == "A" else "docs/specialist_showcase_track_b.json")
    result = run(scenario, a.track == "B", a.output, a.video); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["passed"] else 1)

if __name__ == "__main__": main()

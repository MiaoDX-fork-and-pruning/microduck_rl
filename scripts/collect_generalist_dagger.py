#!/usr/bin/env python3
"""Collect student-state samples relabeled by immutable ONNX teachers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import mujoco, numpy as np, onnxruntime as ort
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, PolicyInference
from mjlab_microduck.generalist_schema import ACTION_DIM, OBS_DIM, make_conditioned_observation, validate_batch
from mjlab_microduck.generalist_transition_graph import validate_transition
from mjlab_microduck.generalist_model import build_actor

_G0_POLICY_STATE = {"velstand_flat": "VELSTAND", "velocity_flat": "VELOCITY", "sitstand_flat": "SITSTAND"}


def extract_boundary_windows(report: dict, frames: list[dict], *, before: int = 25, after: int = 25) -> list[dict]:
    """Extract deterministic, labeled windows around the four proven Track A edges.

    ``frames`` are replay records containing at least ``step`` and ``policy_id``.
    The returned records retain the original fields and add edge/phase/bucket
    metadata; no samples are fabricated and direct velocity/sit-stand edges are
    rejected by the frozen transition graph.
    """
    if before < 0 or after < 0:
        raise ValueError("window sizes must be non-negative")
    if report.get("track") != "A" or report.get("reset_count") != 0:
        raise ValueError("boundary source must be a no-reset Track A report")
    ordered = sorted(frames, key=lambda row: int(row["step"]))
    by_step = {int(row["step"]): row for row in ordered}
    result: list[dict] = []
    seen_edges: set[tuple[str, str]] = set()
    transitions = report.get("transitions", [])
    for transition in transitions:
        start = int(transition.get("start_step", transition.get("step", -1)))
        policy = transition.get("policy_id")
        if policy not in _G0_POLICY_STATE:
            continue
        prior = max((t for t in transitions if int(t.get("start_step", t.get("step", -1))) < start and t.get("policy_id") in _G0_POLICY_STATE), key=lambda t: int(t.get("start_step", t.get("step", -1))), default=None)
        if prior is None:
            continue
        source, destination = _G0_POLICY_STATE[prior["policy_id"]], _G0_POLICY_STATE[policy]
        if source == destination:
            continue
        validate_transition(source, destination)
        seen_edges.add((source, destination))
        for step in range(start - before, start + after):
            frame = by_step.get(step)
            if frame is None:
                raise ValueError(f"missing frame at boundary step {step}")
            item = dict(frame)
            item.update({"transition_id": f"{source}->{destination}@{start}",
                         "transition_source": source, "transition_destination": destination,
                         "transition_phase": "pre" if step < start else "post",
                         "transition_bucket": f"{source}->{destination}"})
            result.append(item)
    expected = {("VELSTAND", "VELOCITY"), ("VELOCITY", "VELSTAND"),
                ("VELSTAND", "SITSTAND"), ("SITSTAND", "VELSTAND")}
    if seen_edges != expected:
        raise ValueError(f"Track A boundary coverage mismatch: {sorted(seen_edges)}")
    return result


def validate_replay_batch(data: dict[str, np.ndarray]) -> None:
    """Check replay fields needed to reproduce teacher labels deterministically."""
    required = ("observation", "requested_command", "raw_action")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"replay missing required fields: {', '.join(missing)}")
    observation = np.asarray(data["observation"])
    command = np.asarray(data["requested_command"])
    action = np.asarray(data["raw_action"])
    conditioned = make_conditioned_observation(observation, command, "stand")
    validate_batch(conditioned, action)
    if action.shape[1] != ACTION_DIM or conditioned.shape[1] != OBS_DIM:
        raise ValueError("replay ABI mismatch")
    lengths = {len(observation), len(command), len(action)}
    for key in ("previous_action", "episode_step", "transition_id"):
        if key in data:
            lengths.add(len(np.asarray(data[key])))
    if len(lengths) != 1:
        raise ValueError("replay fields have inconsistent lengths")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--student-run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--ticks',type=int,default=120); ap.add_argument('--beta',type=float,default=.5); ap.add_argument('--behavior', choices=('stand','locomotion','sit_stand'), default='stand'); args=ap.parse_args()
    import torch
    b=torch.load(args.student_run/'model.pt',weights_only=False)
    manifest = json.loads((args.student_run/'manifest.json').read_text()) if (args.student_run/'manifest.json').exists() else {}
    net=build_actor(manifest.get('metrics', {})); net.load_state_dict(b['state_dict']); net.eval()
    model=mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/scene.xml'); model.opt.timestep=.005
    xs=[]; ys=[]
    profiles = [('stand',0.,'artifacts/specialists/velstand_flat/policy.onnx'),('locomotion',.2,'artifacts/specialists/velocity_flat/policy.onnx'),('sit_stand',1.,'artifacts/specialists/sitstand_flat/policy.onnx')]
    for behavior,speed,teacher_path in [p for p in profiles if p[0] == args.behavior]:
        data=mujoco.MjData(model); teacher=PolicyInference(model,data,walking_onnx_path=teacher_path,new_cmd_obs=True,use_projected_gravity=True); teacher.command=np.array([speed,0,0]+[0]*10,np.float32)
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'trunk_base_freejoint'); qa=int(model.jnt_qposadr[jid]); data.qpos[qa:qa+3]=[0,0,.125]; data.qpos[qa+3:qa+7]=[1,0,0,0]
        for i,q in enumerate(teacher.joint_qpos_indices): data.qpos[q]=DEFAULT_POSE[i]
        mujoco.mj_forward(model,data)
        for _ in range(args.ticks):
            legacy=teacher.get_observations(); x=make_conditioned_observation(legacy[None],teacher.command[None],behavior)
            with torch.no_grad(): student=net(torch.from_numpy(x)).numpy()[0]
            teacher.last_action=teacher.last_action.copy(); ta=teacher.infer()
            xs.append(x[0]); ys.append(ta)
            action=args.beta*ta+(1-args.beta)*student; teacher.last_action=action.astype(np.float32); teacher.apply_action(action)
            for _ in range(5): mujoco.mj_step(model,data)
    args.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output,inputs=np.asarray(xs,np.float32),actions=np.asarray(ys,np.float32)); print({'samples':len(xs),'beta':args.beta,'output':str(args.output)})
if __name__=='__main__': main()

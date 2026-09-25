#!/usr/bin/env python3
"""Execution-ready-v2 VELSTAND causal probe.

This runner owns the frozen case list and uses the specialist ManagerBasedRlEnv
for every closed-loop arm.  Reset states are written through EntityData so the
reward/termination managers see the same state as the policy observation path.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch

from mjlab_microduck.generalist_schema import make_conditioned_observation
from mjlab_microduck.generalist_teachers import FrozenG0Teachers
from mjlab_microduck.generalist_model import build_actor

BUCKETS = ("upright", "seated", "recovery_face_up", "recovery_face_down",
           "recovery_left_side", "recovery_right_side", "recovery_crouched",
           "recovery_natural_fall", "post_recovery_upright")
COMMANDS = ("zero", "nominal_hold")
HORIZON = 1000

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def quat(axis: str, angle: float) -> list[float]:
    v = {"x": (1.,0.,0.), "y": (0.,1.,0.), "z": (0.,0.,1.)}[axis]
    s, c = math.sin(angle/2), math.cos(angle/2)
    return [c, v[0]*s, v[1]*s, v[2]*s]

def case_list(seed: int = 42, episodes: int = 32) -> list[dict]:
    if episodes != 32: raise ValueError("execution-ready-v2 freezes exactly 32 episodes")
    rows=[]
    for i in range(episodes):
        bucket = BUCKETS[i % len(BUCKETS)]
        command = COMMANDS[(i // len(BUCKETS)) % len(COMMANDS)]
        rows.append({"episode": i, "seed": seed+i, "bucket": bucket,
                     "command_bucket": command, "horizon_ticks": HORIZON})
    return rows

def reset_state(env, bucket: str, env_id: int = 0) -> None:
    robot = env.scene["robot"]
    device = env.device
    q = [1.,0.,0.,0.]; z = 0.12
    if bucket == "seated": z, q = .07, quat("y", math.radians(8))
    elif bucket == "recovery_face_up": z, q = .055, quat("y", math.pi)
    elif bucket == "recovery_face_down": z, q = .055, quat("x", math.pi)
    elif bucket == "recovery_left_side": z, q = .06, quat("x", math.pi/2)
    elif bucket == "recovery_right_side": z, q = .06, quat("x", -math.pi/2)
    elif bucket == "recovery_crouched": z, q = .09, quat("y", math.radians(18))
    elif bucket == "recovery_natural_fall": z, q = .10, quat("x", math.radians(55))
    elif bucket == "post_recovery_upright": z, q = .115, [1.,0.,0.,0.]
    pose = torch.tensor([[0.,0.,z,*q]], device=device, dtype=torch.float32)
    vel = torch.zeros((1,6), device=device)
    joints = robot.data.default_joint_pos[env_id:env_id+1].clone()
    robot.data.write_root_pose(pose, torch.tensor([env_id], device=device))
    robot.data.write_root_velocity(vel, torch.tensor([env_id], device=device))
    robot.data.write_joint_state(joints, torch.zeros_like(joints), env_ids=torch.tensor([env_id], device=device))
    env.scene.write_data_to_sim(); env.sim.forward(); env.sim.sense()
    env.observation_manager.compute(update_history=True)

def set_command(env, bucket: str, nominal: bool) -> None:
    term = env.command_manager.get_term("twist")
    value = .15 if nominal else 0.0
    term.vel_command_b[:, :3] = torch.tensor([value,0.,0.], device=env.device)
    if getattr(term, "vel_command_w", None) is not None: term.vel_command_w[:, :3] = term.vel_command_b[:, :3]

def termination_names(env, idx=0):
    return sorted(name for name in env.termination_manager.active_terms
                  if bool(env.termination_manager.get_term(name)[idx].item()))

def run_episode(env, session, student=None, bucket="upright", nominal=False, seed=42):
    obs, _ = env.reset(seed=seed); reset_state(env, bucket); set_command(env, bucket, nominal)
    active=True; actions=[]; observations=[]; tilts=[]; heights=[]; terms={}; reward_sum=0.; main=0.; finite=True
    for tick in range(HORIZON):
        actor = env.observation_manager.compute(update_history=False)["actor"][0].detach().cpu().numpy().astype(np.float32)
        observations.append(actor.copy())
        x = make_conditioned_observation(actor[None], np.r_[([.15,0,0] if nominal else [0,0,0]), np.zeros(10,np.float32)][None], "stand")
        if student is None:
            action = session.run(None, {session.get_inputs()[0].name: actor[None]})[0][0]
        else:
            with torch.inference_mode(): action = student(torch.from_numpy(x)).numpy()[0]
        action=np.asarray(action,np.float32); finite &= bool(np.isfinite(actor).all() and np.isfinite(action).all() and np.abs(action).max() <= 1.00001)
        actions.append(action.copy())
        obs, reward, done, _, _ = env.step(torch.from_numpy(action[None]).to(env.device))
        reward_sum += float(reward[0].item())
        for name, value in env.reward_manager.get_active_iterable_terms(0):
            if name == "track_linear_velocity": main += float(value[0]) * env.step_dt
        r=env.scene["robot"].data.root_link_pose_w[0].detach().cpu().numpy(); heights.append(float(r[2]))
        w,xq,yq,zq = r[3:7]; tilts.append(2*math.acos(min(1.,abs(float(w)))))
        if bool(done[0].item()):
            terms = {name: bool(env.termination_manager.get_term(name)[0].item()) for name in env.termination_manager.active_terms}; active=False; break
    reasons=sorted(k for k,v in terms.items() if v)
    return {"seed": seed, "bucket": bucket, "command_bucket": "nominal_hold" if nominal else "zero",
            "ticks": len(actions), "finite": finite, "observations": np.asarray(observations), "actions": np.asarray(actions),
            "action_max_abs": float(np.abs(actions).max()) if actions else 0.,
            "max_tilt_rad": max(tilts,default=None), "final_height_m": heights[-1] if heights else None,
            "termination_class": reasons or ["incomplete" if active else "none"],
            "terminated": not active, "main_task_metric": main, "total_reward": reward_sum,
            "success": bool(finite and max(tilts,default=99.) < math.radians(65) and not set(reasons)&{"nan_state","fallen_too_long"})}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--episodes",type=int,default=32); p.add_argument("--seed",type=int,default=42); p.add_argument("--ticks",type=int,default=HORIZON); args=p.parse_args()
    if args.ticks != HORIZON: raise ValueError("execution-ready-v2 freezes a 20 s / 50 Hz horizon")
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    cfg=load_env_cfg("Mjlab-VelStand-Flat-MicroDuck",play=True); cfg.scene.num_envs=args.episodes; cfg.seed=args.seed
    env=ManagerBasedRlEnv(cfg,device="cpu")
    onnx=Path("artifacts/specialists/velstand_flat/policy.onnx"); session=ort.InferenceSession(str(onnx),providers=["CPUExecutionProvider"])
    cases=case_list(args.seed,args.episodes); case_hash=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()
    env.reset(seed=args.seed)
    for index, case in enumerate(cases): reset_state(env, case["bucket"], index)
    nominal = torch.tensor([c["command_bucket"] == "nominal_hold" for c in cases], dtype=torch.bool)
    command = env.command_manager.get_term("twist")
    command.vel_command_b[:, :3] = torch.zeros((args.episodes,3))
    command.vel_command_b[nominal, 0] = .15
    if getattr(command, "vel_command_w", None) is not None: command.vel_command_w[:, :3] = command.vel_command_b[:, :3]
    native=[{"seed":c["seed"],"bucket":c["bucket"],"command_bucket":c["command_bucket"],"ticks":0,"finite":True,"observations":[],"actions":[],"tilts":[],"heights":[],"termination_class":[],"terminated":False,"main_task_metric":0.,"total_reward":0.,"success":True} for c in cases]
    reconstructed_teacher = FrozenG0Teachers(device="cpu")
    parity_deltas=[]
    active = np.ones(args.episodes, dtype=bool)
    for _ in range(HORIZON):
        actor = env.observation_manager.compute(update_history=False)["actor"].detach().cpu().numpy().astype(np.float32)
        action = np.concatenate([
            session.run(None, {session.get_inputs()[0].name: actor[i:i+1]})[0]
            for i in range(args.episodes)
        ]).astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        # Use the exact command block emitted by this observation, including
        # head/body command fields. Reconstructing from the command manager can
        # be stale by one manager update and is not a valid parity comparison.
        conditioned = make_conditioned_observation(actor, actor[:, 48:61], "stand")
        reconstructed = reconstructed_teacher(torch.from_numpy(conditioned)).numpy()
        parity_deltas.append(np.abs(reconstructed - action))
        obs, reward, done, _, _ = env.step(torch.from_numpy(action).to(env.device))
        for i in np.flatnonzero(active):
            native[i]["ticks"] += 1; native[i]["observations"].append(actor[i]); native[i]["actions"].append(action[i])
            native[i]["finite"] &= bool(np.isfinite(actor[i]).all() and np.isfinite(action[i]).all() and np.abs(action[i]).max() <= 1.00001)
            root=env.scene["robot"].data.root_link_pose_w[i].detach().cpu().numpy(); w=float(root[3]); native[i]["heights"].append(float(root[2])); native[i]["tilts"].append(2*math.acos(min(1.,abs(w))))
            native[i]["total_reward"] += float(reward[i].item())
            for name,value in env.reward_manager.get_active_iterable_terms(i):
                if name == "track_linear_velocity": native[i]["main_task_metric"] += float(value[0])*env.step_dt
            if bool(done[i].item()):
                native[i]["termination_class"] = termination_names(env,i); native[i]["terminated"] = True; native[i]["success"] = bool(native[i]["finite"] and max(native[i]["tilts"],default=99.) < math.radians(65) and not set(native[i]["termination_class"]) & {"nan_state","fallen_too_long"}); active[i]=False
        if not active.any(): break
    for row in native:
        tilts = row.pop("tilts"); heights = row.pop("heights")
        row["max_tilt_rad"] = max(tilts,default=None); row["final_height_m"] = heights[-1] if heights else None
        row["actions"] = np.asarray(row["actions"]); row["observations"] = np.asarray(row["observations"])
    env.close()
    # Phase-A action parity is checked on exactly the observations used by the
    # native closed-loop rollout, never on an open-loop replay.
    delta=np.concatenate(parity_deltas) if parity_deltas else np.zeros((0,14),np.float32)
    compatibility={"finite":bool(np.isfinite(delta).all()),"max_abs":float(delta.max(initial=0.)),"mean_abs":float(delta.mean() if delta.size else 0.),"passed":bool(delta.max(initial=0.)<=1e-4 and delta.mean()<=1e-5),"samples":int(delta.shape[0])}
    clean=[{k:v for k,v in r.items() if k not in ("actions","observations")} for r in native]
    buckets={b:{"count":sum(r["bucket"]==b for r in native),"success_rate":float(np.mean([r["success"] for r in native if r["bucket"]==b]))} for b in BUCKETS}
    report={"schema":"g0-velstand-causal-probe-phase-a-b","phase":"A","seed":args.seed,"episodes":args.episodes,"horizon_seconds":20,"control_hz":50,"case_list":cases,"case_list_sha256":case_hash,"native_teacher":{"episodes":clean,"per_bucket":buckets,"success_rate":float(np.mean([r["success"] for r in native])),"main_task_metric":float(np.mean([r["main_task_metric"] for r in native]))},"teacher_compatibility":compatibility,"frozen_inputs":{"teacher_onnx":str(onnx),"teacher_onnx_sha256":sha256(onnx)}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(args.output),"case_list_sha256":case_hash,"per_bucket":buckets},indent=2))

if __name__ == "__main__": main()

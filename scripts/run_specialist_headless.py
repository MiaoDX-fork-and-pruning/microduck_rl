#!/usr/bin/env python3
"""Run a canonical specialist scenario without opening the MuJoCo viewer."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from infer_policy import (MICRODUCK_ROLLERS_XML, MICRODUCK_XML, PolicyInference)
from specialist_scenario import load_scenario, scenario_events


def run(scenario_path: Path, output: Path, policies: dict[str, str], roller: bool = False) -> dict:
    scenario, frames = load_scenario(scenario_path)
    xml = MICRODUCK_ROLLERS_XML if roller else MICRODUCK_XML
    model = mujoco.MjModel.from_xml_path(xml); model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    kwargs = {"new_cmd_obs": True}
    mapping = {"velocity_flat":"walking_onnx_path", "velstand_flat":"standing_onnx_path",
               "sitstand_flat":"sitstand_onnx_path", "ground_pick_flat":"ground_pick_onnx_path",
               "ball_kick_flat":"kick_right_onnx_path", "roulade_flat":"roulade_onnx_path"}
    for pid, path in policies.items():
        if pid in mapping: kwargs[mapping[pid]] = path
    if "roller_crouch" in policies:
        kwargs["roller_crouch_onnx_path"] = policies["roller_crouch"]
    if "velocity_rollers" in policies:
        kwargs["walking_onnx_path"] = policies["velocity_rollers"]
    policy = PolicyInference(model, data, **kwargs)
    policy.validate_specialist_policies(frame.policy_id for frame in frames)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = int(model.jnt_qposadr[free]); data.qpos[adr:adr+3] = [0, 0, 0.1385 if roller else 0.125]
    data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
    for i, q in enumerate(policy.joint_qpos_indices): data.qpos[q] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose; mujoco.mj_forward(model, data)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=50, codec="libx264", quality=7)
    renderer = mujoco.Renderer(model, height=368, width=640)
    events = {f.step:f for f in scenario_events(frames)}; actions=[]; observations=[]; policy_ids=[]; resets=0
    try:
        for step, frame in enumerate(frames):
            if step in events: policy.activate_specialist_policy(events[step].policy_id, events[step].command)
            obs = policy.get_observations(); action = policy.infer(); policy.apply_action(action)
            observations.append(obs); actions.append(action); policy_ids.append(frame.policy_id)
            for _ in range(4): mujoco.mj_step(model, data)
            renderer.update_scene(data); writer.append_data(renderer.render())
    finally: writer.close()
    renderer.close()
    report = {"schema_version":1,"mode":"headless_onnx","scenario":str(scenario_path),"seed":scenario["seed"],"frames":len(frames),"command_rate_hz":50,"events":[{"step":f.step,"policy_id":f.policy_id,"command":list(f.command)} for f in events.values()],"reset_count":resets,"action_dim":14,"observation_dim":int(policy.get_observations().size),"reel":str(output)}
    trace = output.with_suffix('.npz')
    np.savez(trace, observations=np.asarray(observations, dtype=np.float32),
             onnx_actions=np.asarray(actions, dtype=np.float32),
             policy_ids=np.asarray(policy_ids))
    report["trace"] = str(trace)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenario',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--policy',action='append',default=[],metavar='ID=ONNX'); p.add_argument('--roller',action='store_true'); a=p.parse_args(); policies=dict(x.split('=',1) for x in a.policy); print(json.dumps(run(a.scenario,a.output,policies,a.roller),indent=2))
if __name__=='__main__': main()

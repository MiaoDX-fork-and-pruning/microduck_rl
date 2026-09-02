#!/usr/bin/env python3
"""Collect student-state samples relabeled by immutable ONNX teachers."""
from __future__ import annotations
import argparse
from pathlib import Path
import mujoco, numpy as np, onnxruntime as ort
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import DEFAULT_POSE, PolicyInference
from mjlab_microduck.generalist_schema import make_conditioned_observation

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--student-run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--ticks',type=int,default=120); ap.add_argument('--beta',type=float,default=.5); args=ap.parse_args()
    import torch
    b=torch.load(args.student_run/'model.pt',weights_only=False); net=torch.nn.Sequential(torch.nn.Linear(71,256),torch.nn.Tanh(),torch.nn.Linear(256,256),torch.nn.Tanh(),torch.nn.Linear(256,14)); net.load_state_dict(b['state_dict']); net.eval()
    model=mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/scene.xml'); model.opt.timestep=.005
    xs=[]; ys=[]
    for behavior,speed,teacher_path in [('stand',0.,'artifacts/specialists/velstand_flat/policy.onnx'),('locomotion',.2,'artifacts/specialists/velocity_flat/policy.onnx')]:
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

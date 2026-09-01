#!/usr/bin/env python3
"""Compare checkpoint-backed PyTorch actions against a headless ONNX trace."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np

def compare(trace: Path, policies: dict[str, tuple[str, str]], output: Path, device: str = "cpu") -> dict:
    import torch
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    data = np.load(trace, allow_pickle=False)
    observations, expected, ids = data["observations"], data["onnx_actions"], data["policy_ids"]
    if observations.shape[1] != 61 or expected.shape != (len(observations), 14): raise ValueError("trace ABI must be [N,61] -> [N,14]")
    reports=[]
    for policy_id, indices in ((pid, np.flatnonzero(ids == pid)) for pid in sorted(set(ids.tolist()))):
        if policy_id not in policies: raise ValueError(f"missing PT policy mapping: {policy_id}")
        task, checkpoint = policies[policy_id]
        env_cfg=load_env_cfg(task, play=True); env_cfg.scene.num_envs=1
        agent_cfg=load_rl_cfg(task); env=RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg,device=device), clip_actions=agent_cfg.clip_actions)
        runner=(load_runner_cls(task) or OnPolicyRunner)(env, asdict(agent_cfg), device=device)
        runner.load(str(Path(checkpoint).resolve()), load_cfg={"actor":True}, strict=True, map_location=device)
        actor=runner.get_inference_policy(device=device)
        with torch.inference_mode(): actual=actor({"actor":torch.from_numpy(observations[indices]).to(device)}).detach().cpu().numpy()
        errors=np.abs(actual-expected[indices]); reports.append({"policy_id":policy_id,"task":task,"frames":int(len(indices)),"max_abs_error":float(errors.max(initial=0.0)),"mean_abs_error":float(errors.mean()),"passed":bool(np.allclose(actual,expected[indices],atol=1e-4,rtol=1e-3))})
        env.close()
    result={"schema_version":1,"trace":str(trace),"frames":len(observations),"observation_dim":61,"action_dim":14,"policies":reports,"passed":all(r["passed"] for r in reports)}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trace',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--policy',action='append',default=[],metavar='ID=TASK=CHECKPOINT'); p.add_argument('--device',default='cpu'); a=p.parse_args(); mappings={}
    for raw in a.policy:
        pid,task,checkpoint=raw.split('=',2); mappings[pid]=(task,checkpoint)
    print(json.dumps(compare(a.trace,mappings,a.output,a.device),indent=2))
if __name__=='__main__': main()

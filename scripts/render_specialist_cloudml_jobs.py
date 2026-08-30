#!/usr/bin/env python3
"""Render specialist CloudML jobs from a reviewed JSON wave definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

IMAGE = "micr.cloud.mioffice.cn/cc-proxy/thelastfoot-openpi-g2-training:microduck-rl-cuda128-20260829-0503"
SOURCE = "/dongxu/microduck_rl/source/652b7ce-20260829T140000Z"
VOLUME = "robot-intelligent-planning-data"
CLUSTER = "wlcb-cloudml"
RESOURCE = "cloudml.ng1r49-8-8.13-107"


def render(job: dict) -> dict:
    slug = job["slug"]
    wave = job.get("wave", "a1")
    run_name = f"{wave}-{slug}-652b7ce"
    return {
        "jobName": f"microduck-specialist-{wave}-{slug}-652b7ce",
        "description": f"Specialist {job['track']} {slug} training from commit 652b7ce1b19f",
        "accessType": "PRIVATE",
        "imageConfig": {
            "imageUrl": IMAGE,
            "imageCommand": "\n".join([
                "set -euo pipefail",
                "export PYTHONUNBUFFERED=1",
                "export PYTHONPATH=/mnt/cloudml/source/src",
                "cd /mnt/cloudml/output",
                f"/opt/microduck_rl/.venv/bin/train {job['task']} \\",
                "  --env.scene.num-envs 4096 \\",
                f"  --agent.max_iterations {job['iterations']} \\",
                "  --agent.logger tensorboard \\",
                f"  --agent.run_name {run_name}",
            ]),
        },
        "juiceFsMountConfigs": [
            {"volume": VOLUME, "juiceFsCluster": CLUSTER, "subPath": SOURCE,
             "mountPath": "/mnt/cloudml/source", "readOnly": True},
            {"volume": VOLUME, "juiceFsCluster": CLUSTER,
             "subPath": f"/dongxu/microduck_rl/runs/specialist-{wave}/{slug}-652b7ce-v1",
             "mountPath": "/mnt/cloudml/output", "readOnly": False},
        ],
        "queueId": "11759",
        "priority": 5,
        "preemptible": False,
        "framework": "pytorch",
        "resourceConfigs": [{
            "nodeRole": "worker", "nodeNumber": 1,
            "perNodeResourceSpec": {"resourcePriority": "GUARANTEED",
                                    "resourceName": RESOURCE, "resourceNumber": 1},
        }],
        "enableTensorboard": False,
        "enableMetrics": False,
        "retryConfig": {"enableRetry": False, "maxRetryTimes": 0,
                        "policySets": ["JobFailed"]},
        "diagOptions": [{"name": "HangDetection", "enable": False}],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wave", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("cloudml/generated"))
    args = parser.parse_args()
    jobs = json.loads(args.wave.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        wave = job.get("wave", "a1")
        path = args.output_dir / f"microduck-specialist-{wave}-{job['slug']}-652b7ce.yaml"
        path.write_text(yaml.safe_dump(render(job), sort_keys=False), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

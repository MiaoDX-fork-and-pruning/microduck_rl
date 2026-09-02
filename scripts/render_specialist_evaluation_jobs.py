#!/usr/bin/env python3
"""Render fixed-seed S2 CloudML evaluation jobs from checkpoint inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

IMAGE = "micr.cloud.mioffice.cn/cc-proxy/thelastfoot-openpi-g2-training:microduck-rl-cuda128-20260829-0503"
VOLUME = "robot-intelligent-planning-data"
CLUSTER = "wlcb-cloudml"
RESOURCE = "cloudml.ng1r49-8-8.13-107"


def render(job: dict, *, source_prefix: str, revision: str) -> dict:
    job_slug = job["id"].replace("_", "-")
    output_prefix = (
        f"/dongxu/microduck_rl/runs/specialist-s2-eval/{job_slug}-{revision}-v1"
    )
    checkpoint = f"/mnt/cloudml/checkpoint/{job['run_prefix']}/{job['checkpoint']}"
    command = "\n".join(
        [
            "set -euo pipefail",
            "export PYTHONUNBUFFERED=1",
            "export PYTHONPATH=/mnt/cloudml/source/src",
            "cd /mnt/cloudml/output",
            f"/opt/microduck_rl/.venv/bin/python /mnt/cloudml/source/scripts/evaluate_specialist_policy.py {job['task']} \\",
            f"  --checkpoint {checkpoint} \\",
            "  --output /mnt/cloudml/output/evaluation.json \\",
            "  --seed 42 \\",
            "  --episodes 32 \\",
            "  --device cuda:0 \\",
            f"  --main-task-term {job['main_task_term']} \\",
            "  --success-threshold 0 \\",
            "  --minimum-success-rate 0 \\",
            "  --minimum-main-task-metric 0 \\",
            "  --video-output /mnt/cloudml/output/diagnostic.mp4 \\",
            "  --video-seconds 15 \\",
            "  --video-width 640 \\",
            "  --video-height 480 \\",
            "  --report-only",
        ]
    )
    return {
        "jobName": f"microduck-specialist-s2-{job_slug}-{revision}",
        "description": f"Specialist S2 {job['id']} calibration evaluation from {revision}",
        "accessType": "PRIVATE",
        "imageConfig": {"imageUrl": IMAGE, "imageCommand": command},
        "juiceFsMountConfigs": [
            {
                "volume": VOLUME,
                "juiceFsCluster": CLUSTER,
                "subPath": source_prefix,
                "mountPath": "/mnt/cloudml/source",
                "readOnly": True,
            },
            {
                "volume": VOLUME,
                "juiceFsCluster": CLUSTER,
                "subPath": job["output_prefix"],
                "mountPath": "/mnt/cloudml/checkpoint",
                "readOnly": True,
            },
            {
                "volume": VOLUME,
                "juiceFsCluster": CLUSTER,
                "subPath": output_prefix,
                "mountPath": "/mnt/cloudml/output",
                "readOnly": False,
            },
        ],
        "queueId": "11759",
        "priority": 5,
        "preemptible": True,
        "framework": "pytorch",
        "resourceConfigs": [
            {
                "nodeRole": "worker",
                "nodeNumber": 1,
                "perNodeResourceSpec": {
                    "resourcePriority": "GUARANTEED",
                    "resourceName": RESOURCE,
                    "resourceNumber": 1,
                },
            }
        ],
        "enableTensorboard": False,
        "enableMetrics": False,
        "retryConfig": {
            "enableRetry": False,
            "maxRetryTimes": 0,
            "policySets": ["JobFailed"],
        },
        "diagOptions": [{"name": "HangDetection", "enable": False}],
    }


def load_jobs(matrix_path: Path, inventory_path: Path) -> list[dict]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    inventory = {
        item["id"]: item
        for item in json.loads(inventory_path.read_text(encoding="utf-8"))
    }
    missing = sorted(item["id"] for item in matrix if item["id"] not in inventory)
    if missing:
        raise ValueError(f"evaluation ids missing from checkpoint inventory: {missing}")
    return [{**inventory[item["id"]], **item} for item in matrix]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--source-prefix", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("cloudml/generated"))
    args = parser.parse_args()

    jobs = load_jobs(args.matrix, args.inventory)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        slug = job["id"].replace("_", "-")
        path = args.output_dir / f"microduck-specialist-s2-{slug}-{args.revision}.yaml"
        path.write_text(
            yaml.safe_dump(
                render(job, source_prefix=args.source_prefix, revision=args.revision),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

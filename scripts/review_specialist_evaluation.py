#!/usr/bin/env python3
"""Record a human diagnostic-video review in an S2 evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_specialist_policy import apply_video_review, reassess_evaluation_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--review")
    parser.add_argument("--success-threshold", type=float)
    parser.add_argument("--minimum-success-rate", type=float)
    parser.add_argument("--minimum-main-task-metric", type=float)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    threshold_args = (args.success_threshold, args.minimum_success_rate)
    if any(value is not None for value in threshold_args):
        if any(value is None for value in threshold_args):
            parser.error(
                "--success-threshold and --minimum-success-rate are required together"
            )
        reassess_evaluation_report(
            report,
            success_threshold=args.success_threshold,
            minimum_success_rate=args.minimum_success_rate,
            minimum_main_task_metric=(
                args.success_threshold
                if args.minimum_main_task_metric is None
                else args.minimum_main_task_metric
            ),
        )
    elif args.minimum_main_task_metric is not None:
        parser.error("--minimum-main-task-metric requires the other threshold options")
    if args.review is not None:
        apply_video_review(report, args.review)
    if args.review is None and args.success_threshold is None:
        parser.error("provide a video review, reassessment thresholds, or both")
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": report["accepted"], "report": str(args.report)}))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

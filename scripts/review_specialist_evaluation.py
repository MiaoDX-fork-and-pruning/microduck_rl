#!/usr/bin/env python3
"""Record a human diagnostic-video review in an S2 evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_specialist_policy import apply_video_review


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--review", required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    apply_video_review(report, args.review)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"accepted": report["accepted"], "report": str(args.report)}))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

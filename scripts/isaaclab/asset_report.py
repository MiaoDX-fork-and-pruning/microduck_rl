"""Emit the MJCF mechanical inventory used as the IsaacLab asset source."""

from __future__ import annotations

import json
from isaaclab_microduck.assets import build_report


if __name__ == "__main__":
    print(json.dumps(build_report(), indent=2, sort_keys=True))

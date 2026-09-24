"""Headless runtime probe executed by the selected IsaacLab launcher."""

from __future__ import annotations

import json
import argparse
import platform

from isaaclab.app import AppLauncher


def main() -> None:
    print("ISAACLAB_PROBE:begin", flush=True)
    parser = argparse.ArgumentParser(description="Probe the IsaacLab runtime.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    print("ISAACLAB_PROBE:app_created", flush=True)
    simulation_app = app_launcher.app
    try:
        import isaaclab
        import torch

        result = {
            "cuda_available": torch.cuda.is_available(),
            "isaaclab": getattr(isaaclab, "__version__", "unknown"),
            "python": platform.python_version(),
            "torch": torch.__version__,
        }
        print("ISAACLAB_PROBE:" + json.dumps(result, sort_keys=True), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

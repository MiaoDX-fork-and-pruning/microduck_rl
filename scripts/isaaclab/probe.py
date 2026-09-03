"""Headless runtime probe executed by the selected IsaacLab launcher."""

from __future__ import annotations

import json
import platform

from isaaclab.app import AppLauncher


def main() -> None:
    app_launcher = AppLauncher(headless=True)
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
        print(json.dumps(result, sort_keys=True))
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

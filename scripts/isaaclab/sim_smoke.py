"""Start, step, and close Isaac Sim through IsaacLab's public API."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Microduck IsaacLab simulator smoke test")
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> None:
    from isaaclab.sim import SimulationCfg, SimulationContext

    sim = SimulationContext(SimulationCfg(dt=0.02, device=args.device))
    sim.reset()
    for _ in range(args.steps):
        sim.step()
    print(f"ISAACLAB_SIM_SMOKE:steps={args.steps}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

"""Convert a Microduck MJCF using Isaac Sim's explicitly enabled importer."""

from __future__ import annotations

import argparse
import os
import shutil

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})


def main() -> None:
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    config = MJCFImporterConfig(
        mjcf_path=os.path.abspath(args.input),
        usd_path=os.path.dirname(output),
        import_scene=True,
        merge_mesh=False,
        collision_from_visuals=False,
        allow_self_collision=True,
        fix_base=False,
        run_asset_transformer=False,
        run_multi_physics_conversion=True,
    )
    generated = MJCFImporter(config).import_mjcf()
    if not os.path.isfile(generated):
        raise FileNotFoundError(f"MJCF importer returned missing USD: {generated}")
    if os.path.abspath(generated) != output:
        shutil.copyfile(generated, output)
    print(f"ISAACLAB_MJCF_CONVERTED:{args.output} source={generated}", flush=True)


try:
    main()
finally:
    app.close()

"""Convert a Microduck MJCF using Isaac Sim's explicitly enabled importer."""

from __future__ import annotations

import argparse
import os

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--headless", action="store_true")
args = parser.parse_args()
app = SimulationApp({"renderer": "RaytracedLighting", "headless": args.headless})


def main() -> None:
    import omni.kit.app
    import omni.kit.commands

    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    if not status or import_config is None:
        raise RuntimeError("MJCF importer did not return an import configuration")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = False
    import_config.distance_scale = 1.0
    status, prim_path = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=os.path.abspath(args.input),
        import_config=import_config,
        dest_path=os.path.abspath(args.output),
        prim_path="/microduck",
    )
    if not status:
        raise RuntimeError(f"MJCF importer failed for {args.input}")
    stage = omni.usd.get_context().get_stage()
    stage.Export(os.path.abspath(args.output))
    print(f"ISAACLAB_MJCF_CONVERTED:{args.output} prim={prim_path}", flush=True)


try:
    main()
finally:
    app.close()

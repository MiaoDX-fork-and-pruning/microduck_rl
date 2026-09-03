"""Convert a Microduck MJCF using Isaac Sim's explicitly enabled importer."""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main() -> None:
    import omni.kit.app
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
    omni.kit.app.get_app().update()
    cfg = MjcfConverterCfg(
        asset_path=os.path.abspath(args.input),
        usd_dir=os.path.dirname(os.path.abspath(args.output)),
        usd_file_name=os.path.basename(args.output),
        force_usd_conversion=True,
    )
    converter = MjcfConverter(cfg)
    print(f"ISAACLAB_MJCF_CONVERTED:{converter.usd_path}", flush=True)


try:
    main()
finally:
    app.close()

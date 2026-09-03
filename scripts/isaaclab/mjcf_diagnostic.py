"""Diagnose Isaac Sim MJCF importer command availability."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
print("MJCF_DIAGNOSTIC:before_app", flush=True)
app = AppLauncher(args).app
print("MJCF_DIAGNOSTIC:after_app", flush=True)

try:
    import omni.kit.app
    import omni.kit.commands

    manager = omni.kit.app.get_app().get_extension_manager()
    path = manager.get_extension_path_by_name("isaacsim.asset.importer.mjcf")
    print(f"MJCF_EXTENSION_PATH:{path}", flush=True)
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.mjcf", True)
    omni.kit.app.get_app().update()
    result = omni.kit.commands.execute("MJCFCreateImportConfig")
    print(f"MJCF_IMPORT_CONFIG:{result!r}", flush=True)
finally:
    app.close()

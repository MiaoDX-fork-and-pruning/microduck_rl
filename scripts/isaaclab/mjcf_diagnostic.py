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
    from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

    print(f"MJCF_IMPORTER:{MJCFImporter.__name__}", flush=True)
    print(f"MJCF_IMPORTER_CONFIG:{MJCFImporterConfig.__name__}", flush=True)
finally:
    app.close()

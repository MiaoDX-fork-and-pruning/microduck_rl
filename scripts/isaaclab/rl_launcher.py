"""Launch IsaacLab's official RL entrypoints with Microduck tasks pre-registered.

IsaacLab 3.0 resolves the task while selecting the RL backend.  Registering the
repository tasks before dispatch keeps that lookup deterministic and still
leaves argument parsing, Hydra config resolution, and RSL-RL execution to the
official entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _app_launcher_args(argv: list[str]) -> dict[str, Any]:
    """Extract Kit launcher arguments before importing simulator-backed tasks.

    IsaacLab 3.0 resolves the task config before its backend entrypoint calls
    ``launch_simulation``.  Microduck's task config necessarily imports an
    articulation class, which can reach ``pxr`` before Kit exists.  Starting
    Kit from this small, task-independent argument subset makes that import
    safe; the official entrypoint then observes the already-running Kit and
    does not launch a second application.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--livestream", type=int, default=None)
    parser.add_argument("--visualizer", "--viz", dest="visualizer", default=None)
    parser.add_argument("--experience", default=None)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--kit_args", default=None)
    parser.add_argument("--xr", action="store_true", default=False)
    parser.add_argument("--enable_cameras", action="store_true", default=False)
    parser.add_argument("--anim_recording_enabled", action="store_true", default=False)
    parser.add_argument("--anim_recording_start_time", type=float, default=None)
    parser.add_argument("--anim_recording_stop_time", type=float, default=None)
    parser.add_argument("--max_visible_envs", type=int, default=None)

    # Kit arguments can begin with ``-`` and therefore need the same fusion as
    # IsaacLab's official dispatcher before argparse sees the token pair.
    from isaaclab.app import AppLauncher

    fused = AppLauncher._fuse_kit_args(argv)
    parsed, _ = parser.parse_known_args(fused)
    values = vars(parsed)
    result: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or value is False:
            continue
        result[key] = value
    if parsed.visualizer is not None:
        result["visualizer"] = [item.strip() for item in parsed.visualizer.split(",") if item.strip()]
        result["visualizer_explicit"] = True
    return result


def _backend_args(argv: list[str]) -> list[str]:
    """Remove launcher-only flags that IsaacLab's Hydra parser must not see."""

    # IsaacLab exposes ``headless`` through AppLauncher configuration and the
    # ``HEADLESS`` environment variable, but does not register ``--headless``
    # on the official RL parser.  We consume it during prelaunch above.
    return [argument for argument in argv if argument != "--headless"]


def _run_with_prelaunched_kit(action: str, backend_args: list[str]) -> int:
    """Launch Kit once, then delegate task parsing and execution to IsaacLab."""

    from isaaclab.app import AppLauncher

    forwarded_args = _backend_args(backend_args)
    app_launcher = AppLauncher(_app_launcher_args(backend_args))
    try:
        # Importing the repository registry after Kit startup is intentional:
        # task config imports may resolve pxr-backed IsaacLab classes.
        from isaaclab_microduck.tasks import register_tasks

        register_tasks()
        if action == "train":
            from isaaclab_rl.entrypoints import run_train_cli

            return int(run_train_cli(forwarded_args))

        from isaaclab_rl.entrypoints import run_play_cli

        return int(run_play_cli(forwarded_args))
    finally:
        app_launcher.app.close()


def main(argv: list[str] | None = None) -> int:
    """Run ``train`` or ``play`` with the remaining arguments unchanged."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"train", "play"}:
        print("usage: rl_launcher.py {train|play} [IsaacLab RL arguments]", file=sys.stderr)
        return 2

    action, backend_args = args[0], args[1:]
    return _run_with_prelaunched_kit(action, backend_args)


if __name__ == "__main__":
    raise SystemExit(main())

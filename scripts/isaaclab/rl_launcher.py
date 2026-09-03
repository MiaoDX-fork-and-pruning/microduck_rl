"""Launch IsaacLab's official RL entrypoints with Microduck tasks pre-registered.

IsaacLab 3.0 resolves the task while selecting the RL backend.  Registering the
repository tasks before dispatch keeps that lookup deterministic and still
leaves argument parsing, Hydra config resolution, and RSL-RL execution to the
official entrypoint.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Run ``train`` or ``play`` with the remaining arguments unchanged."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in {"train", "play"}:
        print("usage: rl_launcher.py {train|play} [IsaacLab RL arguments]", file=sys.stderr)
        return 2

    action, backend_args = args[0], args[1:]
    from isaaclab_microduck.tasks import register_tasks

    register_tasks()
    if action == "train":
        from isaaclab_rl.entrypoints import run_train_cli

        return int(run_train_cli(backend_args))

    from isaaclab_rl.entrypoints import run_play_cli

    return int(run_play_cli(backend_args))


if __name__ == "__main__":
    raise SystemExit(main())

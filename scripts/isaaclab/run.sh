#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/isaaclab/run.sh <command> [args...]

Run a command in an externally installed Isaac Sim/IsaacLab environment.

Select exactly one runtime:
  ISAACLAB_LAUNCHER  Path to an IsaacLab isaaclab.sh launcher
  ISAACSIM_PYTHON    Path to the Python executable bundled with Isaac Sim

Examples:
  ISAACLAB_LAUNCHER=/opt/IsaacLab/isaaclab.sh \
    scripts/isaaclab/run.sh -p scripts/isaaclab/probe.py
  ISAACSIM_PYTHON=/opt/isaac-sim/python.sh \
    scripts/isaaclab/run.sh scripts/isaaclab/probe.py
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ -n "${ISAACLAB_LAUNCHER:-}" && -n "${ISAACSIM_PYTHON:-}" ]]; then
  echo "error: set only one of ISAACLAB_LAUNCHER or ISAACSIM_PYTHON" >&2
  exit 2
fi

if [[ -n "${ISAACLAB_LAUNCHER:-}" ]]; then
  if [[ ! -x "$ISAACLAB_LAUNCHER" ]]; then
    echo "error: ISAACLAB_LAUNCHER is not executable: $ISAACLAB_LAUNCHER" >&2
    exit 2
  fi
  exec "$ISAACLAB_LAUNCHER" "$@"
fi

if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
  if [[ ! -x "$ISAACSIM_PYTHON" ]]; then
    echo "error: ISAACSIM_PYTHON is not executable: $ISAACSIM_PYTHON" >&2
    exit 2
  fi
  exec "$ISAACSIM_PYTHON" "$@"
fi

cat >&2 <<'EOF'
error: no IsaacLab runtime selected
Set ISAACLAB_LAUNCHER to IsaacLab's isaaclab.sh, or ISAACSIM_PYTHON to the
Python launcher bundled with Isaac Sim. See scripts/isaaclab/runtime.toml for
the required runtime contract.
EOF
exit 2

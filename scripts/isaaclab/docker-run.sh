#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim publishes the IsaacLab-compatible runtime as a large, GPU-enabled
# container. Keep it separate from the repository's mjlab uv environment.
image="${ISAACLAB_DOCKER_IMAGE:-nvcr.io/nvidia/isaac-sim:5.0.0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v "${repo_root}:/workspace/microduck_rl:ro" \
  -w /workspace/microduck_rl \
  "$image" "$@"

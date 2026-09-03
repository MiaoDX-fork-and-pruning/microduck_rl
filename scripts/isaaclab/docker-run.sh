#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim publishes the IsaacLab-compatible runtime as a large, GPU-enabled
# container. Keep it separate from the repository's mjlab uv environment.
image="${ISAACLAB_DOCKER_IMAGE:-microduck-isaaclab:2.2.0-isaacsim5.0.0}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
isaaclab_source="${repo_root}/.cache/IsaacLab-v2.2.0"

if [[ ! -d "$isaaclab_source" ]]; then
  echo "error: pinned IsaacLab source is missing; run scripts/isaaclab/fetch_source.sh" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec docker run --rm --gpus all --network host \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e TERM=xterm \
  -v "${repo_root}:/workspace/microduck_rl:ro" \
  -v "${isaaclab_source}:/workspace/IsaacLab:ro" \
  -w /workspace/microduck_rl \
  --entrypoint /bin/bash \
  "$image" "$@"

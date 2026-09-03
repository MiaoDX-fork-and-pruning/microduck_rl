#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${ISAACLAB_DOCKER_IMAGE:-microduck-isaaclab:2.2.0-isaacsim5.0.0}"

exec docker build -t "$image" -f "$repo_root/scripts/isaaclab/Dockerfile" "$repo_root"

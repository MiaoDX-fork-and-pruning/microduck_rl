#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${ISAACLAB_DOCKER_IMAGE:-microduck-isaaclab:3.0.0-isaacsim6.0.1}"

exec docker build -t "$image" -f "$repo_root/scripts/isaaclab/Dockerfile" "$repo_root"

#!/usr/bin/env bash
set -euo pipefail

destination="${ISAACLAB_SOURCE_DIR:-.cache/IsaacLab-v2.2.0}"
revision="46dff135f44683f031edf346e544fcfd8456b2bb"

if [[ -e "$destination/.git" ]]; then
  git -C "$destination" fetch --depth 1 origin "refs/tags/v2.2.0"
  git -C "$destination" checkout --detach "$revision"
else
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 --branch v2.2.0 \
    https://github.com/isaac-sim/IsaacLab.git "$destination"
  git -C "$destination" checkout --detach "$revision"
fi

echo "IsaacLab source ready at $destination ($revision)"

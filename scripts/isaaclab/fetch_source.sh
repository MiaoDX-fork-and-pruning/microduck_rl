#!/usr/bin/env bash
set -euo pipefail

destination="${ISAACLAB_SOURCE_DIR:-.cache/IsaacLab-v3.0.0-beta2.patch1}"
revision="ffff603eafc6b74264a5261cc0183d6a65390d78"

if [[ -e "$destination/.git" ]]; then
  git -C "$destination" fetch --depth 1 origin "refs/tags/v3.0.0-beta2.patch1"
  git -C "$destination" checkout --detach "$revision"
else
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 --branch v3.0.0-beta2.patch1 \
    https://github.com/isaac-sim/IsaacLab.git "$destination"
  git -C "$destination" checkout --detach "$revision"
fi

echo "IsaacLab source ready at $destination ($revision)"

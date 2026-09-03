#!/usr/bin/env bash
set -euo pipefail

destination="${ISAACLAB_SOURCE_DIR:-.cache/IsaacLab-v3.0.0}"
source_ref="release/3.0.0"
revision="c7fd163736878a4a348a63880ff6001ea8b3143e"

if [[ -e "$destination/.git" ]]; then
  git -C "$destination" fetch --depth 1 origin "refs/heads/${source_ref}"
  git -C "$destination" checkout --detach "$revision"
else
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 --branch "$source_ref" \
    https://github.com/isaac-sim/IsaacLab.git "$destination"
  git -C "$destination" checkout --detach "$revision"
fi

echo "IsaacLab source ready at $destination ($revision)"

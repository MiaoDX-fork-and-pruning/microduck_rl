#!/usr/bin/env bash
set -euo pipefail

task_id=$1
output_root=$2
windows=${3:-5}
window_iterations=${4:-100}

for window in $(seq 1 "$windows"); do
  args=(
    "$task_id"
    --env.scene.num-envs 4096
    --agent.max-iterations "$window_iterations"
    --agent.save-interval "$window_iterations"
    --agent.logger tensorboard
    --agent.experiment-name "$output_root/adaptive-run"
    --agent.run-name "window-${window}"
  )
  if (( window > 1 )); then
    args+=(--agent.resume True --agent.load-run '.*' --agent.load-checkpoint 'model_.*.pt')
  fi
  /opt/microduck_rl/.venv/bin/train "${args[@]}"

  checkpoint=$(find "$output_root/adaptive-run" -name 'model_*.pt' -type f | sort -V | tail -n 1)
  test -n "$checkpoint"
  onnx="$output_root/model_${window}.onnx"
  battery="$output_root/battery/window-${window}"
  /opt/microduck_rl/.venv/bin/python /mnt/cloudml/source/scripts/export.py \
    Mjlab-Velocity-Flat-Adaptive-MicroDuck \
    --checkpoint-file "$checkpoint" --onnx-file "$onnx" --device cpu
  /opt/microduck_rl/.venv/bin/python /mnt/cloudml/source/scripts/run_adaptive_capability_battery.py \
    --onnx "$onnx" --output "$battery" --duration-seconds 6.0
  /opt/microduck_rl/.venv/bin/python /mnt/cloudml/source/scripts/advance_adaptive_curriculum.py \
    --battery "$battery/capability.json" \
    --state "$output_root/adaptive-state.json" \
    --stage-file "$output_root/adaptive-stage.json" \
    --step "$((window * window_iterations))" --checkpoint "$checkpoint"
done

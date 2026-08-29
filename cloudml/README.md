# CloudML training

The container pins the CUDA and Python dependency environment. Source code can
be replaced at runtime from JuiceFS, so normal experiment iterations do not
require rebuilding the image.

## Local image smoke test

Build once:

```bash
docker build -t microduck-rl:cuda128 .
```

Run the repository's required smoke test with the current working tree mounted
over the package installed in the image:

```bash
mkdir -p "$PWD/docker-output"
docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/src \
  -v "$PWD:/workspace:ro" \
  -v "$PWD/docker-output:/outputs" \
  microduck-rl:cuda128 \
  Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --agent.max_iterations 5 \
  --agent.logger tensorboard \
  --agent.run_name docker-smoke
```

Checkpoints and TensorBoard data are written below
`docker-output/logs/rsl_rl/`.

## JuiceFS source snapshot

Prepare a clean source snapshot (including non-ignored local edits) and upload
it to a unique immutable directory:

```bash
rm -rf /tmp/microduck-source-snapshot
mkdir -p /tmp/microduck-source-snapshot
tar \
  --exclude='./.git' --exclude='./.venv' --exclude='./.serena' \
  --exclude='./logs' --exclude='./wandb' --exclude='./docker-output' \
  --exclude='*.pt' --exclude='*.onnx' --exclude='*.pyc' \
  -cf - . | tar -xf - -C /tmp/microduck-source-snapshot

/home/mi/executor/exe storage juicefs upload \
  --local-dir /tmp/microduck-source-snapshot \
  --url 'https://cloud.mioffice.cn/juicefs/vol-detail?cluster=wlcb-cloudml&name=robot-intelligent-planning-data&path=/dongxu/microduck_rl/source/<snapshot>/' \
  --json
```

The Executor upload target recursively uploads the complete local directory;
the excludes above are therefore required when the source is prepared from a
working tree.

Copy `microduck-smoke.yaml.example`, replace the image, snapshot, and run
placeholders, then inspect and submit it using the Executor CML passthrough:

```bash
/home/mi/executor/exe compute cloudml cml -- \
  custom_train submit --filename cloudml/microduck-smoke.yaml
```

Submitting consumes CloudML resources and should only be done after reviewing
the resolved image, source snapshot, output path, task ID, and training size.

### Parallel specialist waves

Specialist policies are independent jobs and should be submitted in parallel
after one gait and one episodic pilot has established the real throughput. The
current target for the R49 queue is up to **8 GUARANTEED single-GPU jobs**,
with a best-effort expansion to **16 total jobs** when queue capacity and
workspace quota are confirmed.

Before each wave, inspect both queue capacity and quota through the Executor
CloudML passthrough:

```bash
/home/mi/executor/exe compute cloudml cml -- \
  queue list --format RESOURCETYPE,RESOURCEPRIORITY,RESOURCESPEC,RESOURCEFREE,RESOURCETOTAL
/home/mi/executor/exe compute cloudml cml -- \
  resource quota list --output json
```

`RESOURCEFREE` is queue-wide capacity, not a guarantee that this workspace can
consume it. `BEST_EFFORT_PUBLIC` jobs can be preempted; do not also set the
training task's `--preemptible` flag. Use a separate immutable JuiceFS source
snapshot and output prefix per task/run, and resume only from a verified latest
checkpoint after preemption.

Recommended wave order:

```text
P0  1 gait + 1 episodic pilot
A1  remaining Track A teachers (up to 8 guaranteed slots)
B1  Track B specialists in unused guaranteed/best-effort slots
R1  targeted retries/resumes for failed or preempted jobs
```

Each submitted job must be recorded with its job ID, lane, resource name,
source snapshot, output path, checkpoint, and retry count. The detached monitor
should poll every 30 minutes and stop launching new jobs when queue/quota
preconditions fail; it must not stop jobs already running.

WandB and Hugging Face are not required. `--agent.logger tensorboard` avoids
WandB authentication, and checkpoints persist directly on the writable JuiceFS
mount under `logs/rsl_rl/<experiment>/<run>/model_*.pt`.

## Background monitor

Run a detached monitor that checks the job and JuiceFS every 30 minutes:

```bash
setsid nohup scripts/monitor_cloudml_job.sh <JOB_ID> 1800 \
  </dev/null >>/tmp/microduck-monitor.out 2>&1 &
```

The monitor writes a summary to `cloudml/monitor-<JOB_ID>.log` and stops
automatically when the job reaches a terminal state.

## Video gallery

Collect rollout videos from one or more task directories and build a standalone
HTML review page (no server or Python package required). Put each task under a
separate child directory so its task name is retained in the gallery:

```bash
python scripts/build_video_gallery.py /path/to/rollouts/<TASK_ID> \
  -o artifacts/video-gallery --title "Microduck policy review"
```

For the specialist handoff, build from the validated manifest so each clip is
indexed with its task, acceptance metrics, penalty values, video review,
failure note, and artifact hashes:

```bash
python scripts/build_video_gallery.py \
  --manifest artifacts/specialist/manifest.json \
  -o artifacts/video-gallery --title "Microduck specialist policy review"
```

You can also pass several task directories in one command. The resulting
`index.html` provides task filtering, filename search, native video controls,
and manifest evidence; copy the complete `artifacts/video-gallery/` directory
to the LAN machine and open the HTML file directly.

Open `artifacts/video-gallery/index.html` locally, or copy that directory to
the LAN machine. The page provides task filtering, filename search, and native
video controls.

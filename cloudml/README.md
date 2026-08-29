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

WandB and Hugging Face are not required. `--agent.logger tensorboard` avoids
WandB authentication, and checkpoints persist directly on the writable JuiceFS
mount under `logs/rsl_rl/<experiment>/<run>/model_*.pt`.

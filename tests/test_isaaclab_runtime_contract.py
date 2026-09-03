from __future__ import annotations

import os
import subprocess
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "isaaclab" / "run.sh"
DOCKER_RUNNER = ROOT / "scripts" / "isaaclab" / "docker-run.sh"
DOCKERFILE = ROOT / "scripts" / "isaaclab" / "Dockerfile"
MANIFEST = ROOT / "scripts" / "isaaclab" / "runtime.toml"


def test_runtime_manifest_keeps_isaaclab_external() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    assert manifest["installation"] == "official_docker_image"
    assert "ISAACLAB_LAUNCHER" in manifest["launcher_env"]
    assert manifest["validation"]["headless_probe"].startswith(
        "scripts/isaaclab/docker-run.sh"
    )


def test_docker_runner_is_pinned_and_executable() -> None:
    assert DOCKER_RUNNER.stat().st_mode & 0o111
    assert "microduck-isaaclab:3.0.0-beta2.patch1-isaacsim6.0.1" in DOCKER_RUNNER.read_text()
    assert "/workspace/IsaacLab" in DOCKER_RUNNER.read_text()
    assert "ISAACLAB_DOCKER_WRITE" in DOCKER_RUNNER.read_text()


def test_docker_image_installs_isaaclab_lazy_import_dependency() -> None:
    assert "lazy-loader==0.4" in DOCKERFILE.read_text()


def test_source_pin_is_reproducible() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    assert manifest["isaaclab_source_tag"] == "v3.0.0-beta2.patch1"
    assert len(manifest["isaaclab_source_revision"]) == 40


def test_runner_fails_clearly_without_selected_runtime() -> None:
    env = os.environ.copy()
    env.pop("ISAACLAB_LAUNCHER", None)
    env.pop("ISAACSIM_PYTHON", None)

    result = subprocess.run(
        [str(RUNNER), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "no IsaacLab runtime selected" in result.stderr

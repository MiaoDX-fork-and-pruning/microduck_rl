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
RL_LAUNCHER = ROOT / "scripts" / "isaaclab" / "rl_launcher.py"


def test_runtime_manifest_keeps_isaaclab_external() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    assert manifest["installation"] == "official_docker_image"
    assert "ISAACLAB_LAUNCHER" in manifest["launcher_env"]
    assert manifest["validation"]["headless_probe"].startswith(
        "scripts/isaaclab/docker-run.sh"
    )
    assert manifest["validation"]["status"] == "passed"
    assert "velocity_flat_ppo_smoke" in manifest["validation"]


def test_docker_runner_is_pinned_and_executable() -> None:
    assert DOCKER_RUNNER.stat().st_mode & 0o111
    source = DOCKER_RUNNER.read_text()
    assert "microduck-isaaclab:3.0.0-isaacsim6.0.1" in source
    assert "/workspace/IsaacLab" in source
    assert "ISAACLAB_DOCKER_WRITE" in source
    assert "GIT_PYTHON_REFRESH=quiet" in source
    assert 'chmod a+rwx "${repo_root}/logs"' in source


def test_docker_image_installs_isaaclab_lazy_import_dependency() -> None:
    assert "lazy-loader==0.4" in DOCKERFILE.read_text()


def test_docker_image_supports_official_playback_exports_and_video() -> None:
    source = DOCKERFILE.read_text()

    assert "moviepy==1.0.3" in source
    assert "imageio-ffmpeg==0.6.0" in source
    assert "onnx==1.20.0" in source
    assert "onnxscript==0.5.7" in source


def test_docker_image_pins_trainer_dependencies_without_resolving_torch() -> None:
    source = DOCKERFILE.read_text()
    assert "torch==2.10.0" in source
    assert "rsl-rl-lib==5.4.1" in source
    assert "tensordict==0.10.0" in source
    assert "hydra-core==1.3.2" in source
    assert "tensorboard==2.20.0" in source
    assert "rich==14.2.0" in source
    assert "packaging==26.0" in source
    assert "tqdm==4.67.1" in source
    assert "warp-lang==1.16.0" in source
    assert "newton==1.5.1" in source
    assert source.count("--no-deps") >= 2


def test_source_pin_is_reproducible() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    assert manifest["isaaclab_version"] == "3.0.0"
    assert manifest["isaaclab_source_ref"] == "release/3.0.0"
    assert len(manifest["isaaclab_source_revision"]) == 40
    assert "3.0.0-isaacsim6.0.1" in manifest["runtime"]


def test_runtime_commands_include_backend_probe_dependencies() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())
    for key, command in manifest["validation"].items():
        if key == "status":
            continue
        assert "/workspace/IsaacLab/source/isaaclab_rl" in command
        assert "/workspace/IsaacLab/source/isaaclab_newton" in command
        assert "/workspace/IsaacLab/source/isaaclab_ovphysx" in command
        assert "/workspace/IsaacLab/source/isaaclab_visualizers" in command


def test_rl_launcher_registers_tasks_before_official_dispatch() -> None:
    source = RL_LAUNCHER.read_text()

    assert 'args[0] not in {"train", "play"}' in source
    assert "_backend_args(backend_args)" in source
    assert 'argument != "--headless"' in source
    assert "register_tasks()" in source
    assert "run_train_cli(forwarded_args)" in source
    assert "run_play_cli(forwarded_args)" in source


def test_ppo_smoke_uses_repository_rl_launcher() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    command = manifest["validation"]["velocity_flat_ppo_smoke"]
    assert "scripts/isaaclab/rl_launcher.py train" in command
    assert "--rl_library rsl_rl" in command
    assert "--external_callback" not in command


def test_playback_video_command_selects_a_recording_visualizer() -> None:
    manifest = tomllib.loads(MANIFEST.read_text())

    command = manifest["validation"]["velocity_flat_play_video"]
    assert "scripts/isaaclab/rl_launcher.py play" in command
    assert "--video_length 32" in command
    assert "--viz kit" in command


def test_command_battery_has_fixed_required_scenarios() -> None:
    source = (ROOT / "scripts/isaaclab/velocity_flat_command_battery.py").read_text()

    for name in ("zero", "forward", "lateral", "yaw"):
        assert f'"{name}"' in source
    assert "checkpoint_sha256" in source
    assert "friction_bridge" in source
    assert "mean_actual_vel_xy_m_s" in source
    assert "mean_actual_vel_yaw_rad_s" in source
    assert "RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)" in source
    assert "clip_actions=True" not in source
    assert '"clip_actions": agent_cfg.clip_actions' in source
    assert "handle_deprecated_rsl_rl_cfg(agent_cfg, \"5.4.1\")" in source
    assert "from velocity_flat_battery_spec import" in source


def test_asset_declares_imported_articulation_root_path() -> None:
    source = (ROOT / "src/isaaclab_microduck/assets/microduck.py").read_text()
    assert 'articulation_root_prim_path="/Geometry/trunk_base"' in source


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

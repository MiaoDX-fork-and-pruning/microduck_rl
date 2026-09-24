from __future__ import annotations

from pathlib import Path


def test_fixed_root_dynamics_probe_keeps_both_reference_modes_and_shared_settings() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/isaaclab/fixed_root_dynamics_parity.py"
    ).read_text()
    for token in (
        '"fixed_root_same_state_bam_dynamics"',
        '"mujoco_bam"',
        '"mujoco_motor_only"',
        '"isaaclab_physx"',
        '"external_load"',
        "DELAY = 3",
        "NOMINAL_VOLTAGE = 7.5",
        "DROP_GAIN = 0.1",
        "ROOT_Z = 0.5",
        'cfg.events.randomize_mass_inertia.params["alpha_range"] = (0.0, 0.0)',
        'cfg.events.randomize_armature.params["ranges"] = (1.0, 1.0)',
        "model.dof_frictionloss[dof_ids] = budget",
        "robot.write_root_pose_to_sim_index",
        "robot.write_root_velocity_to_sim_index",
        "scene.write_data_to_sim()",
        "sim.step()",
    ):
        assert token in source


def test_fixed_root_probe_records_trajectory_error_against_each_reference() -> None:
    source = (
        Path(__file__).parents[1] / "scripts/isaaclab/fixed_root_dynamics_parity.py"
    ).read_text()
    assert '"q_max_abs_error_vs_mujoco_bam"' in source
    assert '"qdot_max_abs_error_vs_mujoco_bam"' in source
    assert '"q_max_abs_error_vs_motor_only"' in source
    assert '"qdot_max_abs_error_vs_motor_only"' in source

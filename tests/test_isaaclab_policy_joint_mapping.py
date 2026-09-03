import numpy as np

from isaaclab_microduck.policy_abi import POLICY_JOINT_ORDER, reorder_policy_joints


def test_policy_target_reorders_to_physx_traversal_order() -> None:
    sim_order = [
        "left_hip_yaw", "neck_pitch", "right_hip_yaw", "left_hip_roll",
        "head_pitch", "right_hip_roll", "left_hip_pitch", "head_yaw",
        "right_hip_pitch", "left_knee", "head_roll", "right_knee",
        "left_ankle", "right_ankle",
    ]
    policy = np.arange(14, dtype=np.float32).reshape(1, -1)
    mapped = reorder_policy_joints(policy, sim_order)
    expected = np.array([[0, 5, 9, 1, 6, 10, 2, 7, 11, 3, 8, 12, 4, 13]], dtype=np.float32)
    np.testing.assert_array_equal(mapped, expected)
    assert len(POLICY_JOINT_ORDER) == 14

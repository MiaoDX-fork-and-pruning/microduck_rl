from __future__ import annotations

from isaaclab_microduck.assets import ACTUATED_ORDER, build_report


def test_mjcf_asset_report_has_expected_actuated_order() -> None:
    report = build_report()

    assert report["actuated_joint_order"] == list(ACTUATED_ORDER)
    assert [joint["name"] for joint in report["actuated_joints"]] == list(ACTUATED_ORDER)
    assert all(len(joint["axis"]) == 3 for joint in report["actuated_joints"])

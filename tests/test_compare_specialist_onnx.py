import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "compare_specialist_onnx",
    Path(__file__).parents[1] / "scripts" / "compare_specialist_onnx.py",
)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_actions = _MODULE.compare_actions


def _inputs():
    expected = np.zeros((3, 14), dtype=np.float32)
    actual = expected.copy()
    cases = np.array(["zero_command", "command_extreme:min", "command_extreme:max"])
    return expected, actual, cases


def test_reports_passing_parity_by_case():
    expected, actual, cases = _inputs()
    actual[2, 4] = 5e-6
    report = compare_actions(expected, actual, cases, atol=1e-5, rtol=1e-4)
    assert report["passed"] is True
    assert report["failed_values"] == 0
    assert report["cases"][2]["max_abs_error"] == pytest.approx(5e-6)


def test_reports_failed_values_without_hiding_other_cases():
    expected, actual, cases = _inputs()
    actual[1, 0] = 0.1
    report = compare_actions(expected, actual, cases, atol=1e-5, rtol=1e-4)
    assert report["passed"] is False
    assert report["failed_values"] == 1
    assert [case["passed"] for case in report["cases"]] == [True, False, True]


@pytest.mark.parametrize("missing", ["zero_command", "command_extreme"])
def test_requires_deployment_boundary_cases(missing):
    expected, actual, cases = _inputs()
    cases = np.array([case for case in cases if not case.startswith(missing)])
    with pytest.raises(ValueError, match=missing):
        compare_actions(expected[:len(cases)], actual[:len(cases)], cases, atol=1e-5, rtol=1e-4)


def test_rejects_nonfinite_or_wrong_action_shapes():
    expected, actual, cases = _inputs()
    with pytest.raises(ValueError, match="shape"):
        compare_actions(expected[:, :-1], actual[:, :-1], cases, atol=1e-5, rtol=1e-4)
    actual[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_actions(expected, actual, cases, atol=1e-5, rtol=1e-4)

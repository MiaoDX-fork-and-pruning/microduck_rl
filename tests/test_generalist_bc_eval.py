from pathlib import Path
import importlib.util


def test_bc_evaluator_script_is_present():
    path = Path(__file__).parents[1] / "scripts" / "evaluate_generalist_bc.py"
    assert path.exists()
    spec = importlib.util.spec_from_file_location("evaluate_generalist_bc", path)
    assert spec and spec.loader

from pathlib import Path


def test_converter_uses_current_mjcf_python_api() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/convert_mjcf.py").read_text()

    assert "MJCFImporter" in source
    assert "MJCFImporterConfig" in source
    assert "shutil.copyfile" in source
    assert "SimulationApp" in source

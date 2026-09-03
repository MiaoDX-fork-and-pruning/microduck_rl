from pathlib import Path


def test_mjcf_diagnostic_checks_extension_and_config_command() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/mjcf_diagnostic.py").read_text()

    assert "isaacsim.asset.importer.mjcf" in source
    assert "MJCFCreateImportConfig" in source

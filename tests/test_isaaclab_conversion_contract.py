from pathlib import Path


def test_converter_enables_mjcf_extension() -> None:
    source = (Path(__file__).parents[1] / "scripts/isaaclab/convert_mjcf.py").read_text()

    assert "isaacsim.asset.importer.mjcf" in source
    assert "MjcfConverterCfg" in source

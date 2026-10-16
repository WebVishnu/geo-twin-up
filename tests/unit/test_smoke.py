import importlib

import pytest


@pytest.mark.parametrize("module", ["numpy", "pandas", "scipy", "xarray", "yaml", "xgboost", "geopandas"])
def test_imports(module: str) -> None:
    importlib.import_module(module)


def test_config_loads() -> None:
    import yaml

    with open("configs/base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["region"]["name"] == "Uttar Pradesh"

def test_imports():
    import numpy, pandas, scipy, xarray, yaml  # noqa: F401


def test_config_loads():
    import yaml

    cfg = yaml.safe_load(open("configs/base.yaml", encoding="utf-8"))
    assert cfg["region"]["name"] == "Uttar Pradesh"

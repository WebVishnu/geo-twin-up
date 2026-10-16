"""Year-blocked train/val/test splits (ticket T7). Never random.

A sample is assigned by the year of its *target* date. Features only look backwards, so a
train sample (target year in train years) can never contain information from val/test years.
"""

from __future__ import annotations

import pandas as pd
import yaml


def load_split_years(cfg_path: str = "configs/base.yaml") -> dict[str, list[int]]:
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return {"val": list(cfg["splits"]["val_years"]), "test": list(cfg["splits"]["test_years"])}


def assign_split(target_dates: pd.Series, val_years: list[int], test_years: list[int]) -> pd.Series:
    yr = pd.to_datetime(target_dates).dt.year
    out = pd.Series("train", index=target_dates.index)
    out[yr.isin(val_years)] = "val"
    out[yr.isin(test_years)] = "test"
    first_holdout = min(val_years + test_years)
    out[(yr > first_holdout) & ~yr.isin(val_years + test_years)] = (
        "gap"  # years after holdout start but unassigned: unused
    )
    return out

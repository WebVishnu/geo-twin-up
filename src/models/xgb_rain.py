"""Pooled XGBoost hurdle model for district rainfall (ticket T9).

Rain is mostly zero and heavy-tailed, so: (1) classifier P(rain >= wet_mm), (2) regressor on
log1p(amount) trained on wet days only. Expected rain = P * E[amount | wet]. One model per lead.
Hyper-parameters are tuned on the VALIDATION split only; the test split is touched once.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.build import FEATURE_COLS

WET_MM = 1.0


class HurdleXGB:
    def __init__(
        self,
        wet_mm: float = WET_MM,
        seed: int = 42,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ):
        self.wet_mm, self.seed = wet_mm, seed
        common = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
        }
        self.clf = xgb.XGBClassifier(eval_metric="logloss", **common)
        self.reg = xgb.XGBRegressor(**common)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> HurdleXGB:
        y = np.asarray(y, float)
        wet = y >= self.wet_mm
        self.clf.fit(X, wet.astype(int))
        self.reg.fit(X[wet], np.log1p(y[wet]))
        return self

    def predict_proba_wet(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.clf.predict_proba(X)[:, 1], dtype=np.float64)

    def predict_amount_if_wet(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(np.expm1(np.clip(self.reg.predict(X), 0, None)), dtype=np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Expected rainfall (mm)."""
        return np.asarray(self.predict_proba_wet(X) * self.predict_amount_if_wet(X), dtype=np.float64)


def fit_per_lead(
    feats: pd.DataFrame, train_mask: np.ndarray, cols: list[str] | None = None, **kw: Any
) -> dict[int, HurdleXGB]:
    """One hurdle model per lead day. `cols` defaults to the lag/seasonal features; MOS adds the NWP column."""
    use = cols or FEATURE_COLS
    models: dict[int, HurdleXGB] = {}
    for k, g in feats[train_mask].groupby("lead"):
        models[int(str(k))] = HurdleXGB(**kw).fit(g[use], g["y"].to_numpy())
    return models


def predict_per_lead(models: dict[int, HurdleXGB], feats: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Returns feats[[district_id, issue_date, lead]] + p_wet + rain_mm_pred, aligned to feats' rows."""
    use = cols or FEATURE_COLS
    out = feats[["district_id", "issue_date", "lead"]].copy()
    out["p_wet"], out["rain_mm_pred"] = np.nan, np.nan
    for k, m in models.items():
        idx = feats.index[feats["lead"] == k]
        X = feats.loc[idx, use]
        out.loc[idx, "p_wet"] = m.predict_proba_wet(X)
        out.loc[idx, "rain_mm_pred"] = m.predict(X)
    return out

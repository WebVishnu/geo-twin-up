import numpy as np
import pandas as pd

from src.features.build import FEATURE_COLS
from src.models import xgb_rain


def toy(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.random((n, len(FEATURE_COLS))), columns=FEATURE_COLS)
    wet = rng.random(n) < (0.1 + 0.7 * X["rain_lag0"])      # signal: yesterday predicts today
    y = np.where(wet, rng.gamma(2.0, 6.0, n) * (0.5 + X["rain_lag0"]), 0.0)
    return X, y


def test_nonnegative_and_deterministic():
    X, y = toy()
    a = xgb_rain.HurdleXGB(n_estimators=40).fit(X, y).predict(X)
    b = xgb_rain.HurdleXGB(n_estimators=40).fit(X, y).predict(X)
    assert (a >= 0).all() and np.allclose(a, b)


def test_learns_the_signal():
    X, y = toy()
    tr, te = slice(0, 2000), slice(2000, None)
    m = xgb_rain.HurdleXGB(n_estimators=80).fit(X.iloc[tr], y[tr])
    p = m.predict(X.iloc[te])
    clim = np.full(len(p), y[tr].mean())
    mse = lambda f: np.mean((y[te] - f) ** 2)
    assert mse(p) < mse(clim)


def test_per_lead_models_and_alignment():
    X, y = toy(1200)
    f = X.copy()
    f["y"], f["lead"] = y, np.where(np.arange(len(X)) % 2 == 0, 1, 3)
    f["district_id"], f["issue_date"] = 0, pd.Timestamp("2020-01-01")
    models = xgb_rain.fit_per_lead(f, np.ones(len(f), bool), n_estimators=20)
    assert set(models) == {1, 3}
    out = xgb_rain.predict_per_lead(models, f)
    assert out.rain_mm_pred.notna().all() and out.index.equals(f.index)

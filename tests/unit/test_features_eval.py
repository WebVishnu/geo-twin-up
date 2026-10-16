import numpy as np
import pandas as pd

from src.eval import baselines, metrics, splits
from src.features import build


def synthetic(years=(2019, 2020, 2021, 2022, 2023, 2024, 2025), n_dist=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    rows = []
    for d in range(n_dist):
        rain = rng.gamma(0.4, 8, len(dates)) * (rng.random(len(dates)) < 0.3)
        rows.append(pd.DataFrame({"district_id": d, "date": dates, "rain_mm": rain}))
    df = pd.concat(rows, ignore_index=True)
    cen = pd.DataFrame({"district_id": range(n_dist), "lat": 26 + np.arange(n_dist), "lon": 80.0})
    return df, cen


def test_no_future_information_in_features():
    df, cen = synthetic()
    f = build.build_features(df, cen, leads=(1, 3))
    P = build.to_wide(df)
    for _ in range(200):
        r = f.sample(1, random_state=_).iloc[0]
        t, dist = r.issue_date, int(r.district_id)
        assert r.rain_lag0 == P.loc[t, dist]                               # today
        assert r.rain_lag3 == P.loc[t - pd.Timedelta(days=3), dist]        # past
        assert r.y == P.loc[t + pd.Timedelta(days=int(r.lead)), dist]      # target
        assert np.isclose(r.rain_sum7, P.loc[t - pd.Timedelta(days=6): t, dist].sum())  # window ends at t
    assert (f.target_date > f.issue_date).all()


def test_changing_the_future_does_not_change_features():
    df, cen = synthetic()
    f1 = build.build_features(df, cen, leads=(2,))
    df2 = df.copy()
    cut = pd.Timestamp("2024-06-01")
    df2.loc[df2.date > cut, "rain_mm"] += 100.0     # tamper with the future only
    f2 = build.build_features(df2, cen, leads=(2,))
    key = ["district_id", "issue_date"]
    a = f1[f1.issue_date <= cut].set_index(key)[build.FEATURE_COLS].sort_index()
    b = f2[f2.issue_date <= cut].set_index(key)[build.FEATURE_COLS].sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_split_order_and_no_overlap():
    df, cen = synthetic()
    f = build.build_features(df, cen, leads=(1, 7))
    f["split"] = splits.assign_split(f.target_date, val_years=[2023], test_years=[2024, 2025])
    yr = f.target_date.dt.year
    assert yr[f.split == "train"].max() < yr[f.split == "val"].min() <= yr[f.split == "val"].max() < yr[f.split == "test"].min()
    assert set(f.split) <= {"train", "val", "test", "gap"}


def test_climatology_uses_train_years_only():
    df, _ = synthetic()
    W = build.to_wide(df)
    c1 = baselines.fit_climatology(W, train_years=[2019, 2020, 2021])
    W2 = W.copy()
    W2.loc[W2.index.year >= 2023] += 500           # corrupt non-train years
    c2 = baselines.fit_climatology(W2, train_years=[2019, 2020, 2021])
    pd.testing.assert_frame_equal(c1, c2)


def test_persistence_is_lag0():
    df, cen = synthetic()
    f = build.build_features(df, cen, leads=(1,))
    assert np.array_equal(baselines.persistence(f), f.rain_lag0.to_numpy())


def test_perfect_forecast_scores():
    y = np.array([0, 0, 3, 20, 70.0])
    c = metrics.categorical(y, y, 2.5)
    assert c["pod"] == 1 and c["far"] == 0 and c["csi"] == 1
    assert metrics.rmse(y, y) == 0


def test_all_dry_forecast_has_zero_pod():
    y = np.array([0, 5, 20, 70.0])
    assert metrics.categorical(y, np.zeros(4), 2.5)["pod"] == 0


def test_skill_score_of_reference_is_zero():
    assert metrics.skill_score(3.0, 3.0) == 0
    assert metrics.skill_score(2.0, 4.0) == 0.5

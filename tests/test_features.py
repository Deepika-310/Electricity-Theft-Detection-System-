import pandas as pd
import pytest

from model.features import FEATURE_COLUMNS, build_features


def _series(values_by_hour, days, meter="M00"):
    ts = pd.date_range("2026-08-01", periods=days * 24, freq="h")
    return pd.DataFrame(
        {
            "meter_id": meter,
            "timestamp": ts,
            "voltage": 230.0,
            "consumption": [values_by_hour(t) for t in ts],
            "is_theft": 0,
        }
    )


def test_warmup_rows_get_neutral_features_not_flags():
    feats = build_features(_series(lambda t: 10.0, days=2))
    first_day = feats.iloc[:24]
    assert not first_day["has_history"].any()
    assert (first_day["seasonal_ratio"] == 1.0).all()
    assert (first_day["zscore"] == 0.0).all()


def test_baseline_never_uses_future_readings():
    df = _series(lambda t: 10.0, days=10)
    changed = df.copy()
    changed.loc[changed.index[-1], "consumption"] = 999.0

    a, b = build_features(df), build_features(changed)
    earlier = a.index[:-1]
    for col in FEATURE_COLUMNS:
        assert (a.loc[earlier, col] == b.loc[earlier, col]).all()


def test_reading_does_not_contribute_to_its_own_baseline():
    df = _series(lambda t: 10.0, days=10)
    df.loc[df.index[-1], "consumption"] = 40.0
    last = build_features(df).iloc[-1]
    assert last["seasonal_ratio"] == pytest.approx(4.0, rel=1e-3)


def test_persistent_theft_does_not_poison_the_baseline():
    # Days 4-7 stolen. Day 7's baseline is days 0-6, three of which are already
    # stolen: a mean baseline drifts to ~7 (ratio 0.43), the median stays at 10.
    def usage(t):
        day = (t - pd.Timestamp("2026-08-01")).days
        return 3.0 if 4 <= day <= 7 else 10.0

    df = _series(usage, days=8)
    day7 = build_features(df)
    day7 = day7[day7["timestamp"] >= pd.Timestamp("2026-08-08")]
    assert (day7["seasonal_ratio"] < 0.35).all()


def test_features_are_computed_per_meter():
    a = _series(lambda t: 10.0, days=10, meter="M00")
    b = _series(lambda t: 500.0, days=10, meter="M01")
    feats = build_features(pd.concat([a, b], ignore_index=True))
    assert (feats.loc[feats["has_history"], "seasonal_ratio"].round(6) == 1.0).all()


import numpy as np

from database.init_db import generate_readings
from model.evaluate import score_flags, split_by_time
from model.features import build_features


def test_generation_is_reproducible():
    a = generate_readings(seed=3)
    b = generate_readings(seed=3)
    assert a.equals(b)
    assert not a.equals(generate_readings(seed=4))


def test_theft_labels_are_sane():
    df = generate_readings(seed=11)
    rate = df["is_theft"].mean()
    assert 0.01 < rate < 0.15
    assert df.loc[df["is_theft"] == 1, "theft_type"].notna().all()
    assert df.loc[df["is_theft"] == 0, "theft_type"].isna().all()
    first_3_days = df[df["timestamp"] < df["timestamp"].min() + np.timedelta64(72, "h")]
    assert first_3_days["is_theft"].sum() == 0


def test_theft_appears_in_both_train_and_test_periods():
    for seed in range(1, 6):
        train, test, _ = split_by_time(build_features(generate_readings(seed=seed)))
        assert train["is_theft"].sum() > 0
        assert test["is_theft"].sum() > 0


def test_split_is_chronological_with_no_overlap():
    train, test, cutoff = split_by_time(build_features(generate_readings(seed=2)))
    assert train["timestamp"].max() < cutoff <= test["timestamp"].min()


def test_score_flags_perfect_and_confusion_counts():
    y = np.array([0, 0, 1, 1, 0, 1])
    res = score_flags(y, y.copy(), y.astype(float))
    assert res["precision"] == res["recall"] == res["f1"] == 1.0
    assert res["confusion"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}

    flags = np.array([1, 0, 1, 0, 0, 0])
    res = score_flags(y, flags, flags.astype(float))
    assert res["confusion"] == {"tn": 2, "fp": 1, "fn": 2, "tp": 1}
    assert res["precision"] == 0.5
    assert abs(res["recall"] - 1 / 3) < 1e-9

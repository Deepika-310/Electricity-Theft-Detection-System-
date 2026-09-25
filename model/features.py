import numpy as np
import pandas as pd

# Model inputs. Chosen on separate development seeds (100-104): raw consumption/voltage
# added noise for tree and distance models, and log-ratios fixed heavy-tailed ratios.
FEATURE_COLUMNS = ["log_seasonal_ratio", "zscore"]

SHORT_WINDOW = 24  # hours of recent history for the short-term baseline
SEASONAL_DAYS = 7  # same-hour readings from the previous N days
EPS = 1e-6


def build_features(df):
    """
    Per-meter behavioural features. Every baseline uses only *earlier* readings
    (shift(1)), so a reading never contributes to its own baseline and scoring
    can never leak the future.

    spike_ratio    consumption / median of the previous 24 readings
    seasonal_ratio consumption / median of the same hour over the previous 7 days
    log_seasonal_ratio  log of the above (drops and surges become symmetric)
    zscore         robust z-score against that same-hour baseline (median / MAD)

    Medians and MAD are used instead of means and std so that a few days of theft
    do not drag the baseline down and make persistent theft look normal.

    Rows without enough history get neutral values (ratio 1, z-score 0) rather
    than being flagged, and are marked with has_history=False.
    """
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour

    by_meter = df.groupby("meter_id")["consumption"]
    recent_median = by_meter.transform(lambda s: s.shift(1).rolling(SHORT_WINDOW, min_periods=6).median())

    by_meter_hour = df.groupby(["meter_id", "hour"])["consumption"]
    seasonal_median = by_meter_hour.transform(lambda s: s.shift(1).rolling(SEASONAL_DAYS, min_periods=3).median())
    seasonal_mad = by_meter_hour.transform(
        lambda s: s.shift(1).rolling(SEASONAL_DAYS, min_periods=3).apply(_scaled_mad, raw=True)
    )

    df["has_history"] = seasonal_median.notna()
    df["expected_consumption"] = seasonal_median.fillna(recent_median).fillna(df["consumption"])

    df["spike_ratio"] = (df["consumption"] / (recent_median + EPS)).fillna(1.0)
    df["seasonal_ratio"] = (df["consumption"] / (seasonal_median + EPS)).fillna(1.0)
    df["log_seasonal_ratio"] = np.log(df["seasonal_ratio"].clip(0.02, 50))

    scale = np.maximum(seasonal_mad.fillna(0.0), 0.1 * seasonal_median.fillna(0.0)) + EPS
    df["zscore"] = ((df["consumption"] - seasonal_median) / scale).clip(-10, 10).fillna(0.0)

    return df.drop(columns=["hour"])


def _scaled_mad(values):
    """Median absolute deviation scaled to be comparable to a standard deviation."""
    return 1.4826 * np.median(np.abs(values - np.median(values)))

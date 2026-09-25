"""
Honest evaluation of the candidate detectors.

* Models are trained WITHOUT labels, on the earlier part of the timeline only.
* They are scored on the later part (time-based hold-out). Features use only past
  data, so nothing from the test period leaks into training.
* Labels (is_theft) are used only here, to measure the result.
* Because the data is simulated, every number describes performance on *this
  simulator*, not on real utility data.
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from database.db import get_connection, load_usage
from database.init_db import generate_readings
from model.config import (
    CONTAMINATION,
    DEPLOYED_MODEL,
    METRICS_PATH,
    RANDOM_STATE,
    REPORT_PATH,
    RESULTS_DIR,
)
from model.features import FEATURE_COLUMNS, build_features
from model.models import make_models

TRAIN_FRAC = 0.7
ZSCORE_RULE = "Z-score rule (baseline)"
ZSCORE_THRESHOLD = 3.0
ROBUSTNESS_SEEDS = [1, 2, 3, 4, 5]
SWEEP = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]


def split_by_time(features, train_frac=TRAIN_FRAC):
    start, end = features["timestamp"].min(), features["timestamp"].max()
    cutoff = start + (end - start) * train_frac
    return features[features["timestamp"] < cutoff], features[features["timestamp"] >= cutoff], cutoff


def score_flags(y_true, flags, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, flags, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "f1": float(f1_score(y_true, flags, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run_models(train, test, contamination=CONTAMINATION, seed=RANDOM_STATE):
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_test = test["is_theft"].to_numpy()
    results = {}

    for name, model in make_models(contamination, seed).items():
        t0 = time.perf_counter()
        model.fit(X_train)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        flags = (model.predict(X_test) == -1).astype(int)
        scores = -model.score_samples(X_test)
        predict_s = time.perf_counter() - t0

        res = score_flags(y_test, flags, scores)
        res["fit_seconds"] = round(fit_s, 3)
        res["predict_seconds"] = round(predict_s, 3)
        res["recall_by_type"] = recall_by_type(test, flags)
        results[name] = res

    rule_scores = test["zscore"].abs().to_numpy()
    rule_flags = (rule_scores > ZSCORE_THRESHOLD).astype(int)
    res = score_flags(y_test, rule_flags, rule_scores)
    res.update({"fit_seconds": 0.0, "predict_seconds": 0.0, "recall_by_type": recall_by_type(test, rule_flags)})
    results[ZSCORE_RULE] = res
    return results


def recall_by_type(test, flags):
    theft = test.assign(flag=flags)
    theft = theft[theft["is_theft"] == 1]
    return {t: round(float(g["flag"].mean()), 4) for t, g in theft.groupby("theft_type")}


def contamination_sweep(train, test, seed=RANDOM_STATE):
    X_train, X_test = train[FEATURE_COLUMNS], test[FEATURE_COLUMNS]
    y_test = test["is_theft"].to_numpy()
    rows = []
    for c in SWEEP:
        model = make_models(c, seed)["Isolation Forest"].fit(X_train)
        flags = (model.predict(X_test) == -1).astype(int)
        rows.append(
            {
                "contamination": c,
                "precision": float(precision_score(y_test, flags, zero_division=0)),
                "recall": float(recall_score(y_test, flags, zero_division=0)),
                "f1": float(f1_score(y_test, flags, zero_division=0)),
            }
        )
    return rows


def robustness(seeds=ROBUSTNESS_SEEDS):
    """Repeat the whole experiment on independently simulated datasets."""
    collected = {}
    for seed in seeds:
        feats = build_features(generate_readings(seed=seed))
        train, test, _ = split_by_time(feats)
        if test["is_theft"].nunique() < 2:
            continue
        for name, res in run_models(train, test, seed=seed).items():
            for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
                collected.setdefault(name, {}).setdefault(metric, []).append(res[metric])

    return {
        name: {
            metric: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
            for metric, v in metrics.items()
        }
        for name, metrics in collected.items()
    }


def evaluate(db_path=None, write=True):
    conn = get_connection(db_path)
    try:
        df = load_usage(conn)
    finally:
        conn.close()

    features = build_features(df)
    train, test, cutoff = split_by_time(features)
    if test["is_theft"].nunique() < 2:
        raise ValueError("Hold-out period has no theft rows; regenerate the data.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "deployed_model": DEPLOYED_MODEL,
        "config": {
            "contamination": CONTAMINATION,
            "train_fraction": TRAIN_FRAC,
            "features": FEATURE_COLUMNS,
            "zscore_threshold": ZSCORE_THRESHOLD,
        },
        "dataset": {
            "rows": int(len(features)),
            "meters": int(features["meter_id"].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split_time": cutoff.isoformat(),
            "test_theft_rate": round(float(test["is_theft"].mean()), 4),
        },
        "models": run_models(train, test),
        "contamination_sweep": contamination_sweep(train, test),
        "robustness": {"seeds": ROBUSTNESS_SEEDS, "models": robustness()},
    }

    if write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(payload, f, indent=2)
        with open(REPORT_PATH, "w") as f:
            f.write(render_report(payload))
    return payload


def render_report(p):
    d = p["dataset"]
    lines = [
        "# Model comparison (time-based hold-out)",
        "",
        f"Train rows: {d['train_rows']} | Test rows: {d['test_rows']} | "
        f"Test theft rate: {d['test_theft_rate']:.1%} (a model flagging at random scores PR-AUC ~ {d['test_theft_rate']:.2f})",
        "",
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Fit (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in p["models"].items():
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['fit_seconds']:.2f} |"
        )
    lines += ["", f"## Robustness over {len(p['robustness']['seeds'])} independent simulated datasets (mean +/- std)", ""]
    lines += ["| Model | F1 | ROC-AUC | PR-AUC |", "|---|---|---|---|"]
    for name, r in p["robustness"]["models"].items():
        lines.append(
            f"| {name} | {r['f1']['mean']:.3f} +/- {r['f1']['std']:.3f} | "
            f"{r['roc_auc']['mean']:.3f} +/- {r['roc_auc']['std']:.3f} | "
            f"{r['pr_auc']['mean']:.3f} +/- {r['pr_auc']['std']:.3f} |"
        )
    lines += ["", "## Isolation Forest: contamination sensitivity", "", "| contamination | precision | recall | F1 |", "|---|---|---|---|"]
    for row in p["contamination_sweep"]:
        lines.append(f"| {row['contamination']:.2f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |")
    lines += ["", "## Recall by theft type (fraction of each theft type caught)", ""]
    types = sorted({t for r in p["models"].values() for t in r["recall_by_type"]})
    lines += ["| Model | " + " | ".join(types) + " |", "|---|" + "---|" * len(types)]
    for name, r in p["models"].items():
        lines.append(f"| {name} | " + " | ".join(f"{r['recall_by_type'].get(t, 0):.2f}" for t in types) + " |")
    lines += ["", "_All numbers are from simulated data and describe this simulator, not real utility data._", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    result = evaluate()
    print(render_report(result))
    print(f"Saved -> {METRICS_PATH}")

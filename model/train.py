import argparse
import os
from datetime import datetime, timezone

import joblib
import sklearn

from database.db import get_connection, load_usage
from model.config import CONTAMINATION, DEPLOYED_MODEL, MODEL_PATH, RANDOM_STATE
from model.features import FEATURE_COLUMNS, build_features
from model.models import make_models


def train(db_path=None, model_path=MODEL_PATH, model_name=DEPLOYED_MODEL, contamination=CONTAMINATION):
    """
    Fit the chosen detector on ALL available readings (unsupervised: labels are
    never used) and save it with the metadata needed to serve it safely.
    Honest performance numbers come from model/evaluate.py (time-based hold-out).
    """
    conn = get_connection(db_path)
    try:
        df = load_usage(conn)
    finally:
        conn.close()

    features = build_features(df)[FEATURE_COLUMNS]
    model = make_models(contamination, RANDOM_STATE)[model_name]
    model.fit(features)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "features": FEATURE_COLUMNS,
            "contamination": contamination,
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_rows": len(features),
        },
        model_path,
    )
    print(f"Trained {model_name} on {len(features)} readings -> {model_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save the fraud detector.")
    parser.add_argument("--model", default=DEPLOYED_MODEL, choices=list(make_models()))
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    args = parser.parse_args()
    train(model_name=args.model, contamination=args.contamination)

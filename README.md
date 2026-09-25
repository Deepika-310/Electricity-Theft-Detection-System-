# Electricity Theft Detection System

An end-to-end machine-learning system that flags suspicious electricity usage from smart-meter readings: a simulated data source, an unsupervised anomaly detector, a REST API, and an interactive dashboard, with a reproducible evaluation that compares several models against a non-ML baseline.

> **Scope note.** The data is simulated, so every result below describes this simulator, not real utility data. The simulator exists so that theft is *labelled*, which real utility data almost never is, and that makes honest measurement possible.

## The problem

Utilities lose revenue to meter tampering, bypassing and illegal connections. Confirmed theft labels are rare (theft is only confirmed by physical inspection), so a supervised classifier has nothing to train on. This project treats theft as **anomaly detection**: learn what a meter's normal behaviour looks like, and flag readings that deviate from it.

## Architecture

```
database/  SQLite + simulator (labelled theft episodes)
   |
model/     features -> candidate detectors -> time-based evaluation -> model.pkl
   |
backend/   FastAPI: /detect /fraud-cases /meters /predict /metrics
   |
frontend/  Streamlit dashboard (calls the API over HTTP)
```

## How detection works

**Data.** `database/init_db.py` simulates 10 meters over 30 days of hourly readings (residential and commercial load shapes, weekday/weekend effects, noise). Theft episodes are injected into 40% of meters and labelled (`is_theft`, `theft_type`):

| Type | Behaviour | Duration |
|---|---|---|
| `meter_bypass` | consumption recorded at 25-55% of normal | 24-72 h |
| `meter_stall` | consumption near zero | 12-48 h |
| `abnormal_surge` | consumption 2.5-4x normal | 2-6 h |

Most theft is a *drop* in recorded usage, so a "big number = fraud" rule would miss it.

**Features** (`model/features.py`). Every reading is compared with *that meter's own past*, using only earlier data (no leakage):
- `seasonal_ratio` / `log_seasonal_ratio`: consumption vs. the median of the same hour over the previous 7 days
- `zscore`: robust z-score (median / MAD) against that same-hour baseline
- Medians and MAD are used instead of means and standard deviations so that a few days of theft cannot drag the baseline down and make persistent theft look normal.

**Model.** Isolation Forest (unsupervised; labels are never used for training). It is compared with Local Outlier Factor, One-Class SVM, Elliptic Envelope and a plain robust z-score threshold rule.

## Evaluation

Run with `python -m model.evaluate`. Detectors are trained without labels on the first 70% of the timeline and scored on the last 30% (time-based split). Labels are used only to measure the result.

Held-out results (seed 42; test set is 3.9% theft, so a random scorer has PR-AUC of about 0.04):

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Isolation Forest** | 0.941 | 0.565 | 0.706 | 0.982 | 0.782 |
| Local Outlier Factor | 0.177 | 0.259 | 0.211 | 0.622 | 0.101 |
| One-Class SVM | 0.512 | 0.518 | 0.515 | 0.569 | 0.524 |
| Elliptic Envelope | 0.703 | 0.529 | 0.604 | 0.973 | 0.706 |
| Z-score rule (baseline) | 0.316 | 0.882 | 0.466 | 0.977 | 0.758 |

Across 5 independently simulated datasets (mean +/- std): Isolation Forest ROC-AUC 0.970 +/- 0.019 and PR-AUC 0.792 +/- 0.112; Elliptic Envelope is statistically tied (0.969 / 0.784); the z-score rule reaches 0.924 / 0.674. Full tables, including contamination sensitivity and recall by theft type, are in `model/results/comparison.md`.

### What the experiments showed
- **Feature engineering mattered more than model choice.** With raw consumption and voltage as inputs, a plain z-score rule beat every ML model. Feature selection (dropping noisy raw features, log-transforming ratios) was done on separate development seeds (100-104), not on the seeds reported above.
- **The decision threshold matters.** Isolation Forest's `contamination` sets the operating point: at 0.05 precision is 0.94 and recall 0.57; at 0.10 recall rises to 0.91 and precision drops to 0.37. The right value depends on the cost of a missed theft versus a wasted inspection.
- **Known weakness: meter bypass.** Isolation Forest catches all surges and stalls but none of the bypass episodes in the seed-42 test set at the default threshold (the z-score rule catches 73%, at low precision). A partial reduction sits in the middle of the distribution, which is exactly where isolation-based methods are weakest.

## API

| Endpoint | Description |
|---|---|
| `GET /` | health check |
| `GET /detect?meter_id=&limit=` | readings with anomaly label (`-1` anomaly, `1` normal) and score |
| `GET /fraud-cases?meter_id=&limit=` | only flagged readings, most suspicious first |
| `GET /meters` | per-meter flag rates |
| `POST /predict` | score a reading; pass `meter_id` to score against that meter's history |
| `GET /metrics` | hold-out evaluation results |

```json
POST /predict   {"consumption": 0.0, "voltage": 230, "meter_id": "M03"}
->              {"prediction": -1, "is_fraud": true, "anomaly_score": 0.741, "context": "history", ...}
```

Interactive docs at `http://127.0.0.1:8000/docs`. Invalid bodies return 422, unknown meters 404.

## Project structure

```
backend/    main.py (routes), services.py (FraudDetector), schemas.py (Pydantic models)
database/   schema.sql, db.py, init_db.py (simulator)
model/      features.py, models.py, train.py, evaluate.py, config.py, results/
frontend/   app.py (Streamlit)
tests/      feature/leakage, data generation, evaluation metrics, API
```

Training and serving share `build_features`, so there is no training/serving skew.

## Run it

```bash
pip install -r requirements.txt        # requirements-dev.txt adds pytest
python -m database.init_db             # simulate data (seeded, reproducible)
python -m model.train                  # fit and save model/model.pkl
python -m model.evaluate               # write model/results/
uvicorn backend.main:app --reload      # API on :8000
streamlit run frontend/app.py          # dashboard
python -m pytest                       # tests
```

### Docker

```bash
docker compose up --build   # API on :8000, dashboard on :8501
```

The image simulates data and trains the model at build time, so the saved model always matches the installed scikit-learn. GitHub Actions (`.github/workflows/ci.yml`) runs the tests and builds the image on every push.

Run the modules with `python -m` from the project root, otherwise the package imports fail.

## Limitations

- Simulated data only; the simulator's theft patterns are simpler than real theft.
- One evaluation dataset plus 5 repeats: numbers carry sampling noise (see the +/- above).
- The model is trained on the whole history and loaded once at API start; retraining needs a restart.
- No authentication, and SQLite is not suited to concurrent, high-volume ingestion.
- Anomalies are not proof of theft: legitimate causes (new appliances, vacancies) look similar. The output is a shortlist for inspection.

## Next steps

Validate on a real dataset (e.g. the SGCC theft dataset), add weather and holiday features, retrain on a schedule with drift monitoring, and move storage to PostgreSQL/TimescaleDB.

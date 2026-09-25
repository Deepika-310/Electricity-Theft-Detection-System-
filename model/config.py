import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "model", "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "comparison.md")

CONTAMINATION = 0.05
RANDOM_STATE = 42
DEPLOYED_MODEL = "Isolation Forest"

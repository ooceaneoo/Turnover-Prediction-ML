from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
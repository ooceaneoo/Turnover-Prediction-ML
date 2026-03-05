import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from turnover_ml.data_prep import load_raw_data, clean_dataset
from turnover_ml.features import add_engineered_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


def main():
    print("TRAIN BEST MODEL (with preprocessing)")

    df_raw = load_raw_data("data")
    df = clean_dataset(df_raw)
    df = add_engineered_features(df)

    y = df["a_quitte_l_entreprise_num"].copy()
    X = df.drop(columns=["a_quitte_l_entreprise_num"], errors="ignore").copy()

        
    raw_feature_names = list(X.columns)

    categorical_levels = {}
    for col in X.select_dtypes(include="object").columns:
        vals = (
            X[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
            .tolist()
        )
        categorical_levels[col] = sorted(vals)[:50] 

    example_features = X.iloc[0].to_dict()

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("under", RandomUnderSampler(random_state=42)),
        ("logreg", LogisticRegression(max_iter=5000, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grid = {
        "logreg__C": [0.01, 0.1, 1, 5, 10],
        "logreg__solver": ["liblinear", "lbfgs"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=10,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, y_proba)

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2*(precision*recall)/(precision+recall+1e-12)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    metrics = {
        "model": "LogReg + UnderSampling + Preprocessing",
        "cv_best_average_precision": float(search.best_score_),
        "test_average_precision": float(ap),
        "best_threshold_max_f1": float(best_threshold),
        "best_params": search.best_params_,
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "raw_feature_names": raw_feature_names,
        "categorical_levels": categorical_levels,
        "example_features": example_features,
    }

    model_path = MODELS_DIR / "pipeline.joblib"
    metrics_path = REPORTS_DIR / "metrics.json"

    joblib.dump(best_model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved pipeline to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Average precision (test): {ap:.4f}")
    print(f"Best threshold (max F1): {best_threshold:.4f}")


if __name__ == "__main__":
    main()
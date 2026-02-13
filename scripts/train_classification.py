import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spotify_ml.config import (  # noqa: E402
    CLASSIFICATION_TARGET,
    MODEL_DIR,
    RAW_DATA_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
)
from spotify_ml.evaluation.metrics import classification_metrics  # noqa: E402
from spotify_ml.models.builders import (  # noqa: E402
    build_classification_baseline,
    build_classification_improved,
    build_preprocessor,
)
from spotify_ml.preprocessing.dataset import (  # noqa: E402
    get_feature_columns,
    prepare_dataframe,
)
from spotify_ml.preprocessing.splits import train_val_test_split  # noqa: E402
from spotify_ml.training.trainers import train_model, tune_model  # noqa: E402
from spotify_ml.utils.io import ensure_dir, load_csv, save_json  # noqa: E402
from spotify_ml.utils.seed import set_seed  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-engineering", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--optimize-threshold", action="store_true")
    return parser.parse_args()


def choose_threshold(y_true, y_score):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        metrics = classification_metrics(y_true, y_pred, y_score)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = t
    return float(best_t), float(best_f1)


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    df = load_csv(RAW_DATA_PATH)
    df = prepare_dataframe(df, feature_engineering=args.feature_engineering)

    numeric_cols, categorical_cols = get_feature_columns(
        feature_engineering=args.feature_engineering
    )
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    feature_cols = numeric_cols + categorical_cols

    df = df.dropna(subset=[CLASSIFICATION_TARGET])
    x = df[feature_cols]
    y = df[CLASSIFICATION_TARGET].values

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x, y, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    baseline = build_classification_baseline(preprocessor)
    baseline = train_model(baseline, x_train, y_train)
    val_score = baseline.predict_proba(x_val)[:, 1]
    val_pred = (val_score >= 0.5).astype(int)
    baseline_metrics = classification_metrics(y_val, val_pred, val_score)

    ensure_dir(MODEL_DIR)
    joblib.dump(baseline, MODEL_DIR / "classification_baseline.joblib")

    improved = build_classification_improved(preprocessor)
    best_params = {}
    if args.tune:
        param_grid = {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
        improved, best_params, best_score = tune_model(
            improved,
            param_grid,
            x_train,
            y_train,
            scoring="f1",
            n_iter=args.n_iter,
            random_state=RANDOM_SEED,
        )
    else:
        improved = train_model(improved, x_train, y_train)

    improved_val_score = improved.predict_proba(x_val)[:, 1]
    threshold = 0.5
    threshold_f1 = None
    if args.optimize_threshold:
        threshold, best_f1 = choose_threshold(y_val, improved_val_score)
        threshold_f1 = best_f1

    improved_val_pred = (improved_val_score >= threshold).astype(int)
    improved_metrics = classification_metrics(
        y_val, improved_val_pred, improved_val_score
    )
    joblib.dump(improved, MODEL_DIR / "classification_improved.joblib")

    results = {
        "baseline_val": baseline_metrics,
        "improved_val": improved_metrics,
        "best_params": best_params,
        "threshold": threshold,
        "threshold_optimized": bool(args.optimize_threshold),
        "threshold_f1": threshold_f1,
        "feature_engineering": args.feature_engineering,
        "tuned": args.tune,
    }
    save_json(results, RESULTS_DIR / "classification" / "training_summary.json")


if __name__ == "__main__":
    main()

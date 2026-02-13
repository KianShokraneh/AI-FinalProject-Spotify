import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spotify_ml.config import (  # noqa: E402
    MODEL_DIR,
    RAW_DATA_PATH,
    REGRESSION_TARGET,
    RESULTS_DIR,
    RANDOM_SEED,
)
from spotify_ml.evaluation.metrics import regression_metrics  # noqa: E402
from spotify_ml.models.builders import (  # noqa: E402
    build_preprocessor,
    build_regression_baseline,
    build_regression_improved,
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
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional experiment name. When set, saves models/results under run-specific paths.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)
    run_suffix = f"__{args.run_name}" if args.run_name else ""
    results_base = (
        RESULTS_DIR / "regression" / args.run_name
        if args.run_name
        else RESULTS_DIR / "regression"
    )

    df = load_csv(RAW_DATA_PATH)
    df = prepare_dataframe(df, feature_engineering=args.feature_engineering)

    numeric_cols, categorical_cols = get_feature_columns(
        feature_engineering=args.feature_engineering
    )
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    feature_cols = numeric_cols + categorical_cols
    df = df.dropna(subset=[REGRESSION_TARGET])
    x = df[feature_cols]
    y = df[REGRESSION_TARGET].values

    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        x, y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    baseline = build_regression_baseline(preprocessor)
    baseline = train_model(baseline, x_train, y_train)
    val_pred = baseline.predict(x_val)
    baseline_metrics = regression_metrics(y_val, val_pred)

    ensure_dir(MODEL_DIR)
    joblib.dump(
        baseline,
        MODEL_DIR / f"regression_baseline{run_suffix}.joblib",
    )

    improved = build_regression_improved(preprocessor)
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
            scoring="neg_mean_absolute_error",
            n_iter=args.n_iter,
            random_state=RANDOM_SEED,
        )
    else:
        improved = train_model(improved, x_train, y_train)

    improved_val_pred = improved.predict(x_val)
    improved_metrics = regression_metrics(y_val, improved_val_pred)
    joblib.dump(
        improved,
        MODEL_DIR / f"regression_improved{run_suffix}.joblib",
    )

    results = {
        "baseline_val": baseline_metrics,
        "improved_val": improved_metrics,
        "best_params": best_params,
        "feature_engineering": args.feature_engineering,
        "tuned": args.tune,
    }
    save_json(results, results_base / "training_summary.json")


if __name__ == "__main__":
    main()

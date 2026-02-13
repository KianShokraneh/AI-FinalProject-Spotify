import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spotify_ml.config import (  # noqa: E402
    RAW_DATA_PATH,
    REGRESSION_TARGET,
    RESULTS_DIR,
    RANDOM_SEED,
)
from spotify_ml.evaluation.error_analysis import regression_error_analysis  # noqa: E402
from spotify_ml.evaluation.metrics import regression_metrics  # noqa: E402
from spotify_ml.evaluation.plots import plot_regression_diagnostics  # noqa: E402
from spotify_ml.preprocessing.dataset import (  # noqa: E402
    get_feature_columns,
    prepare_dataframe,
)
from spotify_ml.preprocessing.splits import train_val_test_split  # noqa: E402
from spotify_ml.utils.io import load_csv, save_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-engineering", action="store_true")
    parser.add_argument("--model", choices=["baseline", "improved", "both"], default="both")
    return parser.parse_args()


def evaluate_model(model_path: Path, x_test, y_test, df_test, output_dir: Path):
    model = joblib.load(model_path)
    y_pred = model.predict(x_test)
    metrics = regression_metrics(y_test, y_pred)
    plot_regression_diagnostics(y_test, y_pred, output_dir / "figures")
    regression_error_analysis(
        df_test,
        y_test,
        y_pred,
        output_dir / "error_analysis.csv",
    )
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def main():
    args = parse_args()
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
    df_test = df.loc[x_test.index]

    results = {}
    if args.model in {"baseline", "both"}:
        results["baseline"] = evaluate_model(
            Path("models/regression_baseline.joblib"),
            x_test,
            y_test,
            df_test,
            RESULTS_DIR / "regression" / "baseline",
        )
    if args.model in {"improved", "both"}:
        results["improved"] = evaluate_model(
            Path("models/regression_improved.joblib"),
            x_test,
            y_test,
            df_test,
            RESULTS_DIR / "regression" / "improved",
        )

    save_json(results, RESULTS_DIR / "regression" / "metrics.json")


if __name__ == "__main__":
    main()

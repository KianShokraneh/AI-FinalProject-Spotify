from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spotify_ml.config import (  # noqa: E402
    CLASSIFICATION_TARGET,
    RAW_DATA_PATH,
    RESULTS_DIR,
    RANDOM_SEED,
)
from spotify_ml.evaluation.error_analysis import classification_error_analysis  # noqa: E402
from spotify_ml.evaluation.metrics import (  # noqa: E402
    classification_confusion,
    classification_metrics,
)
from spotify_ml.evaluation.plots import plot_confusion_matrix, plot_roc_curve  # noqa: E402
from spotify_ml.preprocessing.dataset import (  # noqa: E402
    get_feature_columns,
    prepare_dataframe,
)
from spotify_ml.preprocessing.splits import train_val_test_split  # noqa: E402
from spotify_ml.utils.io import load_csv, save_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-engineering", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional experiment name. When set, loads run-specific models and writes run-specific results.",
    )
    parser.add_argument(
        "--training-summary",
        type=str,
        default=None,
        help="Path to classification training summary JSON (contains optimized threshold).",
    )
    parser.add_argument(
        "--force-threshold",
        action="store_true",
        help="Always use --threshold, even if a trained threshold exists.",
    )
    parser.add_argument("--model", choices=["baseline", "improved", "both"], default="both")
    return parser.parse_args()


def _load_trained_threshold(training_summary_path: Path) -> float | None:
    if not training_summary_path.exists():
        return None
    try:
        import json

        payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    threshold = payload.get("threshold")
    try:
        return float(threshold) if threshold is not None else None
    except Exception:
        return None


def evaluate_model(model_path: Path, x_test, y_test, df_test, output_dir: Path, threshold: float):
    model = joblib.load(model_path)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = classification_metrics(y_test, y_pred, y_score)
    conf = classification_confusion(y_test, y_pred)
    plot_confusion_matrix(conf, output_dir / "figures" / "confusion_matrix.png")
    plot_roc_curve(y_test, y_score, output_dir / "figures" / "roc_curve.png")
    classification_error_analysis(
        df_test,
        y_test,
        y_pred,
        y_score,
        output_dir / "error_analysis.csv",
    )
    metrics_with_threshold = dict(metrics)
    metrics_with_threshold["threshold"] = float(threshold)
    save_json(metrics_with_threshold, output_dir / "metrics.json")
    return metrics_with_threshold


def main():
    args = parse_args()
    run_suffix = f"__{args.run_name}" if args.run_name else ""
    results_base = (
        RESULTS_DIR / "classification" / args.run_name
        if args.run_name
        else RESULTS_DIR / "classification"
    )
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
    df_test = df.loc[x_test.index]

    default_training_summary = results_base / "training_summary.json"
    training_summary_path = (
        Path(args.training_summary) if args.training_summary else default_training_summary
    )
    trained_threshold = None if args.force_threshold else _load_trained_threshold(training_summary_path)

    results = {}
    if args.model in {"baseline", "both"}:
        results["baseline"] = evaluate_model(
            Path("models") / f"classification_baseline{run_suffix}.joblib",
            x_test,
            y_test,
            df_test,
            results_base / "baseline",
            args.threshold,
        )
    if args.model in {"improved", "both"}:
        improved_threshold = args.threshold if trained_threshold is None else trained_threshold
        results["improved"] = evaluate_model(
            Path("models") / f"classification_improved{run_suffix}.joblib",
            x_test,
            y_test,
            df_test,
            results_base / "improved",
            improved_threshold,
        )

    save_json(results, results_base / "metrics.json")


if __name__ == "__main__":
    main()

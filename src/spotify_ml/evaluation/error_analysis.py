from __future__ import annotations

from pathlib import Path

import pandas as pd

from spotify_ml.utils.io import save_csv


def classification_error_analysis(
    df: pd.DataFrame,
    y_true,
    y_pred,
    y_score,
    output_path: Path,
    top_k: int = 50,
) -> None:
    result = df.copy()
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["y_score"] = y_score
    result["error_type"] = "correct"
    result.loc[(result["y_true"] == 0) & (result["y_pred"] == 1), "error_type"] = "false_positive"
    result.loc[(result["y_true"] == 1) & (result["y_pred"] == 0), "error_type"] = "false_negative"

    errors = result[result["error_type"] != "correct"].copy()
    errors["abs_score_diff"] = (errors["y_score"] - 0.5).abs()
    errors = errors.sort_values("abs_score_diff", ascending=False).head(top_k)
    save_csv(errors, output_path)


def regression_error_analysis(
    df: pd.DataFrame,
    y_true,
    y_pred,
    output_path: Path,
    top_k: int = 50,
) -> None:
    result = df.copy()
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    result["abs_error"] = (result["y_true"] - result["y_pred"]).abs()
    errors = result.sort_values("abs_error", ascending=False).head(top_k)
    save_csv(errors, output_path)

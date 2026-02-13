from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

from spotify_ml.utils.io import ensure_dir


def plot_regression_diagnostics(y_true, y_pred, output_dir: Path) -> None:
    ensure_dir(output_dir)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_actual.png", dpi=200)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, s=10, alpha=0.6)
    plt.axhline(0, color="red", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_pred.png", dpi=200)
    plt.close()


def plot_confusion_matrix(conf_matrix, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(4, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true, y_score, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    disp = RocCurveDisplay.from_predictions(y_true, y_score)
    disp.figure_.set_size_inches(5, 4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

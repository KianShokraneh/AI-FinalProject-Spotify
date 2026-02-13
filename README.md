# Spotify ML Project

End-to-end machine learning pipeline for Spotify Persian artists data with two tasks:

- Regression: predict `popularity`
- Classification: predict `is_sonnati`

The project includes modular preprocessing, training, evaluation, error analysis, plots, and a Streamlit demo.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Project Structure

- `src/spotify_ml/config.py`: paths, targets, feature lists, seed
- `src/spotify_ml/preprocessing/`: cleaning, dataset prep, train/val/test split
- `src/spotify_ml/features/`: engineered features + `is_sonnati` label creation
- `src/spotify_ml/models/`: preprocessor + baseline/improved model builders
- `src/spotify_ml/training/`: fit + randomized tuning helpers
- `src/spotify_ml/evaluation/`: metrics, plots, error analysis
- `scripts/`: train/eval entry points
- `notebooks/`: EDA + Phase 2 result notebook
- `app/streamlit_app.py`: interactive inference UI
- `results/`: metrics, figures, error-analysis artifacts
- `models/`: saved model artifacts (`.joblib`)

## Data

Default dataset path:

- `data/raw/Spotfiy_Persian_Artists.csv`

Fallback path (if needed):

- `Spotfiy_Persian_Artists.csv` (project root)

## Training

### Regression

```bash
python scripts/train_regression.py
python scripts/train_regression.py --feature-engineering
python scripts/train_regression.py --tune --n-iter 20
python scripts/train_regression.py --feature-engineering --tune --n-iter 20
```

### Classification

```bash
python scripts/train_classification.py
python scripts/train_classification.py --feature-engineering
python scripts/train_classification.py --tune --n-iter 20
python scripts/train_classification.py --optimize-threshold
python scripts/train_classification.py --feature-engineering --tune --n-iter 20 --optimize-threshold
```

## Evaluation

### Regression

```bash
python scripts/eval_regression.py --model both
python scripts/eval_regression.py --model baseline
python scripts/eval_regression.py --model improved
```

### Classification

```bash
python scripts/eval_classification.py --model both
python scripts/eval_classification.py --threshold 0.5
python scripts/eval_classification.py --threshold 0.45 --force-threshold
```

## Experiment Tracking with `--run-name`

All train/eval scripts support `--run-name` to avoid overwriting artifacts.

Example:

```bash
python scripts/train_classification.py --feature-engineering --tune --n-iter 20 --optimize-threshold --run-name cls_fe_tuned
python scripts/eval_classification.py --feature-engineering --run-name cls_fe_tuned
```

Artifacts will be saved as:

- models:
  - `models/classification_baseline__cls_fe_tuned.joblib`
  - `models/classification_improved__cls_fe_tuned.joblib`
- results:
  - `results/classification/cls_fe_tuned/training_summary.json`
  - `results/classification/cls_fe_tuned/metrics.json`
  - `results/classification/cls_fe_tuned/improved/figures/...`

Same pattern works for regression runs.

## Metrics

- Regression: `MAE`, `RMSE`, `R2`
- Classification: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC` (+ threshold)

## Results and Reports

- Main run summaries:
  - `results/regression/metrics.json`
  - `results/classification/metrics.json`
- Comparison tables:
  - `results/regression_model_comparison.md`
  - `results/classification_model_comparison.md`
- Notebook report:
  - `notebooks/phase-2_results.ipynb`

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app loads saved models and allows manual feature input for predictions.

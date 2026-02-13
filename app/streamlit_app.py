import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from spotify_ml.config import RAW_DATA_PATH  # noqa: E402
from spotify_ml.features.core import add_engineered_features  # noqa: E402
from spotify_ml.preprocessing.dataset import (  # noqa: E402
    get_feature_columns,
    prepare_dataframe,
)


st.set_page_config(page_title="Spotify ML Demo", layout="wide")
st.title("Spotify ML Demo")

model_base = ROOT / "models"
baseline_reg = model_base / "regression_baseline.joblib"
improved_reg = model_base / "regression_improved.joblib"
baseline_cls = model_base / "classification_baseline.joblib"
improved_cls = model_base / "classification_improved.joblib"


@st.cache_data
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    df = prepare_dataframe(df, feature_engineering=False)
    return df


@st.cache_data
def load_training_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource
def load_model(path: Path):
    import joblib

    return joblib.load(path)


df = load_data()
numeric_cols, categorical_cols = get_feature_columns(feature_engineering=False)
feature_cols = [c for c in numeric_cols + categorical_cols if c in df.columns]


def input_form(key_prefix: str):
    values = {}
    st.subheader("Input Features")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        default = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
        values[col] = st.number_input(col, value=default, key=f"{key_prefix}_num_{col}")
    for col in categorical_cols:
        if col not in df.columns:
            continue
        options = sorted(df[col].dropna().unique().tolist())
        values[col] = st.selectbox(col, options, key=f"{key_prefix}_cat_{col}")
    return values


def prepare_input_for_model(model, input_df: pd.DataFrame) -> pd.DataFrame:
    try:
        required_cols = list(model.named_steps["preprocess"].feature_names_in_)
    except Exception:
        required_cols = list(input_df.columns)

    if any(col in required_cols for col in ["duration_min", "energy_x_danceability", "loudness_sq", "tempo_log"]):
        input_df = add_engineered_features(input_df)

    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    return input_df[required_cols]


tab1, tab2 = st.tabs(["Regression: Popularity", "Classification: Is Sonnati"])

with tab1:
    values = input_form("regression")
    input_df = pd.DataFrame([values])
    model_path = st.selectbox(
        "Model",
        options=[str(baseline_reg), str(improved_reg)],
        index=1,
        key="reg_model",
    )
    if st.button("Predict Popularity"):
        if Path(model_path).exists():
            model = load_model(Path(model_path))
            input_df = prepare_input_for_model(model, input_df)
            pred = model.predict(input_df)[0]
            st.success(f"Predicted popularity: {pred:.2f}")
        else:
            st.error("Model not found. Run training scripts first.")

with tab2:
    values = input_form("classification")
    input_df = pd.DataFrame([values])
    model_path = st.selectbox(
        "Model ",
        options=[str(baseline_cls), str(improved_cls)],
        index=1,
        key="cls_model",
    )
    default_threshold = 0.5
    summary_path = ROOT / "results" / "classification" / "training_summary.json"
    summary = load_training_summary(summary_path) if summary_path.exists() else {}
    trained_threshold = summary.get("threshold", None)
    try:
        trained_threshold = float(trained_threshold) if trained_threshold is not None else None
    except Exception:
        trained_threshold = None

    is_improved = Path(model_path).name == improved_cls.name
    use_trained_threshold_default = bool(is_improved and trained_threshold is not None)
    use_trained_threshold = st.checkbox(
        "Use trained threshold (from results/classification/training_summary.json)",
        value=use_trained_threshold_default,
        disabled=trained_threshold is None,
    )
    threshold = st.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(trained_threshold if (use_trained_threshold and trained_threshold is not None) else default_threshold),
        step=0.01,
    )

    if st.button("Predict Sonnati"):
        if Path(model_path).exists():
            model = load_model(Path(model_path))
            input_df = prepare_input_for_model(model, input_df)
            score = model.predict_proba(input_df)[0, 1]
            pred = 1 if score >= threshold else 0
            st.success(f"Predicted label: {pred} (score={score:.3f}, threshold={threshold:.2f})")
        else:
            st.error("Model not found. Run training scripts first.")



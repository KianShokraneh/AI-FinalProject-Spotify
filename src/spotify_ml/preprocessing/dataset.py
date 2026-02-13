from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from spotify_ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from spotify_ml.features.core import add_engineered_features, add_is_sonnati_label
from spotify_ml.preprocessing.cleaning import clean_raw_data


def prepare_dataframe(df: pd.DataFrame, feature_engineering: bool = False) -> pd.DataFrame:
    df = clean_raw_data(df)
    df = add_is_sonnati_label(df)
    if feature_engineering:
        df = add_engineered_features(df)
    return df


def get_feature_columns(feature_engineering: bool = False) -> Tuple[List[str], List[str]]:
    numeric = list(NUMERIC_FEATURES)
    if feature_engineering:
        numeric.extend(["duration_min", "energy_x_danceability", "loudness_sq", "tempo_log"])
    categorical = list(CATEGORICAL_FEATURES)
    return numeric, categorical


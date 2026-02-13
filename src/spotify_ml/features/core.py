from __future__ import annotations

import numpy as np
import pandas as pd

SONNATI_ARTISTS = [
    "Salar Aghili",
    "Mohammadreza Shajarian",
    "Hesameddin Seraj",
    "Mohammad Esfahani",
    "Abdolhosein Mokhtabad",
    "Hossein Alizadeh",
    "Kayhan Kalhor",
    "Alireza Eftekhari",
    "Iraj Bastami",
    "Alireza Ghorbani",
    "Parviz Meshkatian",
    "Mohammad Reza Lotfi",
    "Ali Zand Vakili",
    "Kaveh Deylami",
    "Hatam Asgari",
    "Homayoun Shajarian",
    "Shahram Nazeri",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000.0
    if {"energy", "danceability"}.issubset(df.columns):
        df["energy_x_danceability"] = df["energy"] * df["danceability"]
    if "loudness" in df.columns:
        df["loudness_sq"] = df["loudness"] ** 2
    if "tempo" in df.columns:
        df["tempo_log"] = np.log1p(df["tempo"])
    return df


def add_is_sonnati_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_sonnati"] = df["artist_name"].apply(
        lambda name: 1 if name in SONNATI_ARTISTS else 0
    )
    return df

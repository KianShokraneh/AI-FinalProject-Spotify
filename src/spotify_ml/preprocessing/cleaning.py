from __future__ import annotations

import pandas as pd


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "album_release_date" in df.columns:
        df["album_release_date"] = pd.to_datetime(
            df["album_release_date"], errors="coerce"
        )
        if "album_release_year" in df.columns:
            missing_year = df["album_release_year"].isna()
            df.loc[missing_year, "album_release_year"] = df.loc[
                missing_year, "album_release_date"
            ].dt.year

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].fillna(0).astype(int)

    return df

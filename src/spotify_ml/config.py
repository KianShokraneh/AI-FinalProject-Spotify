from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Spotfiy_Persian_Artists.csv"
if not RAW_DATA_PATH.exists():
    RAW_DATA_PATH = PROJECT_ROOT / "Spotfiy_Persian_Artists.csv"

REGRESSION_TARGET = "popularity"
CLASSIFICATION_TARGET = "is_sonnati"

NUMERIC_FEATURES = [
    "disc_number",
    "duration_ms",
    "explicit",
    "track_number",
    "album_total_tracks",
    "album_release_year",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

CATEGORICAL_FEATURES = [
    "key_name",
    "mode_name",
    "key_mode",
]

MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

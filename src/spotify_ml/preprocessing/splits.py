from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    stratify: np.ndarray | None = None,
) -> Tuple[np.ndarray, ...]:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=stratify,
    )
    relative_val_size = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1 - relative_val_size,
        random_state=random_state,
        stratify=y_temp if stratify is not None else None,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test

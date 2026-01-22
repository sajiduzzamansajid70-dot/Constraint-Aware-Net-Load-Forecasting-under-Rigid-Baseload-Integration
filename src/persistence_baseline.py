import numpy as np
import pandas as pd


def predict_persistence(df: pd.DataFrame, target_col: str, lag_hours: int) -> pd.Series:
    """
    Persistence baseline:
      y_hat(t) = y(t - lag_hours)
    Assumes df is time-ordered.
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col '{target_col}' not found in df")

    return df[target_col].shift(lag_hours)


def make_persistence_predictions(df_full: pd.DataFrame, target_col: str) -> dict:
    """
    Returns two persistence baselines:
      - A4_Persist_24h
      - A5_Persist_168h
    """
    return {
        "A4_Persist_24h": predict_persistence(df_full, target_col, 24),
        "A5_Persist_168h": predict_persistence(df_full, target_col, 168),
    }

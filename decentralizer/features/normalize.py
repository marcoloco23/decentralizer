"""Quantile normalization and rolling window features for Numerai compatibility."""

from __future__ import annotations

import pandas as pd
import numpy as np


def quantile_normalize(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Cross-sectional quantile normalize features to [0, 1].

    For each date: rank all symbols, divide by count to get uniform [0, 1].
    This matches Numerai's normalization — ensures features are comparable across time.

    Args:
        df: DataFrame with 'date' and 'symbol' columns plus feature columns
        feature_cols: List of column names to normalize

    Returns:
        DataFrame with normalized feature columns (prefixed with 'feature_')
    """
    df = df.copy()

    if "date" in df.columns:
        # Cross-sectional normalization: rank within each date
        for col in feature_cols:
            if col not in df.columns:
                continue
            normalized_col = f"feature_{col}"
            df[normalized_col] = df.groupby("date")[col].transform(
                lambda x: x.rank(method="average", na_option="keep") / x.count()
                if x.count() > 1
                else 0.5
            )
    else:
        # Single cross-section (no date grouping)
        for col in feature_cols:
            if col not in df.columns:
                continue
            s = df[col]
            normalized_col = f"feature_{col}"
            if s.count() > 1:
                df[normalized_col] = s.rank(method="average", na_option="keep") / s.count()
            else:
                df[normalized_col] = 0.5

    # Fill NaN normalized values with 0.5 (neutral)
    norm_cols = [f"feature_{c}" for c in feature_cols if c in df.columns]
    df[norm_cols] = df[norm_cols].fillna(0.5)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    raw_cols: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling average features per symbol.

    Computes rolling means for each raw metric at specified windows.
    Also computes acceleration ratio (short_window / long_window).

    Args:
        df: DataFrame with 'date', 'symbol', and raw feature columns.
            Must be sorted by date within each symbol.
        raw_cols: Raw metric column names to compute rolling features for
        windows: Rolling window sizes in days. Defaults to [20, 60] (Numerai standard).

    Returns:
        DataFrame with added rolling columns:
        - {col}_avg_{w}d for each window
        - {col}_accel for short/long ratio (if 2+ windows)
    """
    if windows is None:
        windows = [20, 60]

    df = df.copy()
    df = df.sort_values(["symbol", "date"])

    for col in raw_cols:
        if col not in df.columns:
            continue

        for w in windows:
            roll_col = f"{col}_avg_{w}d"
            df[roll_col] = df.groupby("symbol")[col].transform(
                lambda x: x.rolling(window=w, min_periods=1).mean()
            )

        # Acceleration: short / long ratio
        if len(windows) >= 2:
            short_w, long_w = windows[0], windows[-1]
            short_col = f"{col}_avg_{short_w}d"
            long_col = f"{col}_avg_{long_w}d"
            accel_col = f"{col}_accel"
            df[accel_col] = df[short_col] / df[long_col].replace(0, np.nan)
            df[accel_col] = df[accel_col].fillna(1.0)

    return df

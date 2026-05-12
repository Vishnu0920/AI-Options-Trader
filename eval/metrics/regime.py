"""
eval/metrics/regime.py

Regime-split metric: VIX median split with per-regime directional accuracy
and Wilson confidence intervals. No separate thresholds for high-VIX:
the same universal thresholds apply, and CIs capture regime uncertainty.
"""

import pandas as pd

from eval.metrics.wilson_ci import wilson_score_interval


def regime_split(
    df: pd.DataFrame, vix_col: str = "vix"
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, float, float, float, float]:
    """
    Split evaluation results by VIX regime (median split) and compare
    directional accuracy with Wilson 95% confidence intervals.

    Returns
    -------
    high_vix_df, low_vix_df, median_vix,
    high_acc, high_ci_low, high_ci_high,
    low_acc, low_ci_low, low_ci_high
    """
    if vix_col not in df.columns:
        raise ValueError(f"DataFrame must contain column {vix_col!r}")

    median_vix = float(df[vix_col].median())
    high_vix_df = df[df[vix_col] > median_vix].copy()
    low_vix_df = df[df[vix_col] <= median_vix].copy()

    high_acc = float("nan")
    high_ci_low, high_ci_high = float("nan"), float("nan")
    low_acc = float("nan")
    low_ci_low, low_ci_high = float("nan"), float("nan")

    if "predicted_direction" in df.columns and "actual_direction" in df.columns:
        if len(high_vix_df) > 0:
            high_correct = int(
                (high_vix_df["predicted_direction"] == high_vix_df["actual_direction"]).sum()
            )
            high_n = len(high_vix_df)
            high_acc = high_correct / high_n
            high_ci_low, high_ci_high = wilson_score_interval(high_correct, high_n)

        if len(low_vix_df) > 0:
            low_correct = int(
                (low_vix_df["predicted_direction"] == low_vix_df["actual_direction"]).sum()
            )
            low_n = len(low_vix_df)
            low_acc = low_correct / low_n
            low_ci_low, low_ci_high = wilson_score_interval(low_correct, low_n)

    return (
        high_vix_df, low_vix_df, median_vix,
        high_acc, high_ci_low, high_ci_high,
        low_acc, low_ci_low, low_ci_high,
    )

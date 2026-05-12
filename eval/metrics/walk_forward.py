"""
eval/metrics/walk_forward.py

Walk-forward directional accuracy with Wilson confidence intervals.
"""

from typing import Any

import pandas as pd

from eval.metrics.wilson_ci import wilson_score_interval

WINDOW_SIZE = 5  # days per walk-forward block


def walk_forward_accuracy(
    df: pd.DataFrame, window_size: int = WINDOW_SIZE
) -> pd.DataFrame:
    """
    Compute directional accuracy per contiguous walk-forward window with
    Wilson confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded results. Must contain 'day', 'predicted_direction',
        'actual_direction'.
    window_size : int
        Days per window (default 5).

    Returns
    -------
    pd.DataFrame with columns:
        window, n, correct, accuracy, ci_low, ci_high
    """
    required = {"day", "predicted_direction", "actual_direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()
    df["correct"] = df["predicted_direction"] == df["actual_direction"]

    days = sorted(df["day"].unique())
    windows: list[dict[str, Any]] = []

    for start_idx in range(0, len(days), window_size):
        block_days = days[start_idx : start_idx + window_size]
        window_df = df[df["day"].isin(block_days)]

        n = len(window_df)
        correct = int(window_df["correct"].sum())
        accuracy = correct / n if n > 0 else float("nan")
        ci_low, ci_high = wilson_score_interval(correct, n)

        label = f"days_{block_days[0]}-{block_days[-1]}"
        windows.append(
            {
                "window": label,
                "n": n,
                "correct": correct,
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    return pd.DataFrame(windows)

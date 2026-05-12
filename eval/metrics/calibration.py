"""
eval/metrics/calibration.py

Conviction calibration metric: Expected Calibration Error (ECE) with a
circuit-breaker rule for the highest-conviction bin.
"""

from typing import Any

import pandas as pd

from eval.thresholds import get_threshold

N_BINS = 5
CIRCUIT_BREAKER_BIN_EDGE = 0.70
CIRCUIT_BREAKER_MIN_ACC = 0.50


def conviction_calibration(
    df: pd.DataFrame, n_bins: int = N_BINS
) -> tuple[pd.DataFrame, float, bool | None, bool]:
    """
    Compute Expected Calibration Error (ECE) and per-bin reliability.

    Returns
    -------
    bin_df : pd.DataFrame
        One row per bin with mean_conviction, actual_accuracy, n, weight.
    ece : float
    passed : bool | None
        True if ece <= max_ece, None if threshold unset.
    circuit_broken : bool
        True if the highest-conviction bin (>0.70) has accuracy < 50%.
    """
    required = {"conviction", "predicted_direction", "actual_direction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()
    df["correct"] = (df["predicted_direction"] == df["actual_direction"]).astype(int)

    df = df.dropna(subset=["conviction"])
    if len(df) == 0:
        empty = pd.DataFrame(
            columns=["bin", "mean_conviction", "actual_accuracy", "n", "weight"]
        )
        return empty, float("nan"), None, False

    df["bin"] = pd.cut(
        df["conviction"], bins=n_bins, include_lowest=True, precision=2
    )

    rows: list[dict[str, Any]] = []
    total = len(df)
    ece = 0.0
    circuit_broken = False

    for interval, sub in df.groupby("bin", observed=False):
        if len(sub) == 0:
            continue
        mean_conv = float(sub["conviction"].mean())
        actual_acc = float(sub["correct"].mean())
        n = len(sub)
        weight = n / total
        ece += weight * abs(mean_conv - actual_acc)

        right_edge = float(interval.right) if hasattr(interval, "right") else 0.0
        if right_edge > CIRCUIT_BREAKER_BIN_EDGE and actual_acc < CIRCUIT_BREAKER_MIN_ACC:
            circuit_broken = True

        rows.append(
            {
                "bin": f"{max(0.0, float(interval.left)):.2f}-{float(interval.right):.2f}",
                "mean_conviction": mean_conv,
                "actual_accuracy": actual_acc,
                "n": n,
                "weight": weight,
            }
        )

    bin_df = pd.DataFrame(rows)
    try:
        passed = ece <= get_threshold("max_ece")
    except ValueError:
        passed = None

    return bin_df, ece, passed, circuit_broken

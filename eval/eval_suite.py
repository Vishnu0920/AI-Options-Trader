"""
eval/eval_suite.py

Pure-analysis evaluation suite for the QS AI-SLM Signal Pod.
Reads orchestrator output JSONL and computes every committed metric.

Zero dependencies on models, training code, or Kaggle.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from eval.thresholds import (
    THRESHOLDS,
    get_threshold,
    all_thresholds_set,
    missing_thresholds,
)

# ── Constants ────────────────────────────────────────────────────────────

REASON_CODES = {
    "ADX_SUPPRESSED",
    "PARSE_FAILURE",
    "LOW_CONVICTION",
    "PASSED",
}

WINDOW_SIZE = 5  # days per walk-forward block
N_BINS = 5       # default for ECE computation
CIRCUIT_BREAKER_BIN_EDGE = 0.70
CIRCUIT_BREAKER_MIN_ACC = 0.50


# ── 1. Load results ──────────────────────────────────────────────────────

def load_results(filepath: str | Path) -> pd.DataFrame:
    """
    Read orchestrator output JSONL into a pandas DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the JSONL file (e.g. results/orchestrator_outputs.jsonl).

    Returns
    -------
    pd.DataFrame
        One row per record.

    Raises
    ------
    FileNotFoundError
    ValueError
        If no valid records are found.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    records: list[dict[str, Any]] = []
    bad_lines = 0
    with filepath.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
                warnings.warn(f"Malformed JSON on line {line_no}; skipping.")

    if not records:
        raise ValueError(
            f"No valid records found in {filepath}. "
            f"({bad_lines} malformed lines)"
        )

    df = pd.DataFrame(records)

    # Derive schema_valid if missing
    if "schema_valid" not in df.columns and "reason_code" in df.columns:
        df["schema_valid"] = df["reason_code"] != "PARSE_FAILURE"

    return df


# ── 2. Wilson score confidence interval ─────────────────────────────────

def wilson_score_interval(
    k: int, n: int, alpha: float = 0.05
) -> tuple[float, float]:
    """
    Compute the Wilson score confidence interval for a proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Total number of trials.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    (ci_low, ci_high) : both in [0.0, 1.0]
    """
    if n == 0:
        return 0.0, 1.0

    if k < 0 or k > n:
        raise ValueError(f"k ({k}) must be between 0 and n ({n})")

    p = k / n
    z = _z_score(1 - alpha / 2)

    denom = 1 + (z ** 2) / n
    centre = (p + (z ** 2) / (2 * n)) / denom
    margin = (
        z
        * math.sqrt(p * (1 - p) / n + (z ** 2) / (4 * n ** 2))
        / denom
    )

    ci_low = max(0.0, centre - margin)
    ci_high = min(1.0, centre + margin)
    return ci_low, ci_high


def _z_score(quantile: float) -> float:
    """Inverse normal CDF (approximation)."""
    # Acklam approximation
    if quantile <= 0 or quantile >= 1:
        raise ValueError("quantile must be in (0,1)")
    p = quantile
    if p > 0.5:
        r = 1 - p
        sign = 1
    else:
        r = p
        sign = -1
    t = math.sqrt(-2.0 * math.log(r))
    c0, c1, c2, d1, d2, d3 = (
        2.515517, 0.802853, 0.010328,
        1.432788, 0.189269, 0.001308,
    )
    z = t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)
    return sign * z


# ── 3. Walk-forward accuracy ───────────────────────────────────────────

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

    # Only consider rows that were actually executed by the orchestrator
    # (i.e. not suppressed, not parse failures, not downgrades).
    # However, the design doc says accuracy denominator = all rows where
    # predicted_direction and actual_direction can be compared.
    # NEUTRAL == NEUTRAL counts as correct.
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


# ── 4. Schema pass rate ────────────────────────────────────────────────

def schema_pass_rate(df: pd.DataFrame) -> tuple[float, bool | None]:
    """
    Compute the fraction of rows with valid JSON/schema output.

    Returns
    -------
    rate : float in [0.0, 1.0]
    passed : bool | None
        True if rate >= min_schema_pass_rate, None if threshold unset.
    """
    if "schema_valid" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'schema_valid' or 'reason_code'"
        )

    rate = float(df["schema_valid"].mean())
    try:
        passed = rate >= get_threshold("min_schema_pass_rate")
    except ValueError:
        passed = None

    return rate, passed


# ── 5. Conviction calibration (ECE) ────────────────────────────────────

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

    # Drop rows with missing conviction
    df = df.dropna(subset=["conviction"])
    if len(df) == 0:
        empty = pd.DataFrame(
            columns=["bin", "mean_conviction", "actual_accuracy", "n", "weight"]
        )
        return empty, float("nan"), None, False

    # Bin by conviction
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

        # Circuit breaker: highest conviction bin edge > 0.70
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


# ── 6. Orchestrator rates ──────────────────────────────────────────────

def orchestrator_rates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute suppression, parse-failure, low-conviction downgrade, and pass
    rates as fractions of the total dataset.

    Returns
    -------
    dict with keys:
        adx_suppressed, parse_failure, low_conviction, passed, n
    """
    if "reason_code" not in df.columns:
        raise ValueError("DataFrame must contain 'reason_code'")

    n = len(df)
    if n == 0:
        return {
            "adx_suppressed": float("nan"),
            "parse_failure": float("nan"),
            "low_conviction": float("nan"),
            "passed": float("nan"),
            "n": 0,
        }

    rates = {
        "adx_suppressed": float((df["reason_code"] == "ADX_SUPPRESSED").mean()),
        "parse_failure": float((df["reason_code"] == "PARSE_FAILURE").mean()),
        "low_conviction": float((df["reason_code"] == "LOW_CONVICTION").mean()),
        "passed": float((df["reason_code"] == "PASSED").mean()),
        "n": n,
    }

    # Sanity check: they should sum to ~1.0
    total_rate = sum(v for k, v in rates.items() if k != "n")
    if not math.isclose(total_rate, 1.0, abs_tol=1e-6):
        warnings.warn(
            f"Orchestrator rates sum to {total_rate:.6f}, not 1.0. "
            "Check for unexpected reason codes."
        )

    return rates


# ── 7. Regime split (VIX) ──────────────────────────────────────────────

def regime_split(
    df: pd.DataFrame, vix_col: str = "vix"
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float, float]:
    """
    Split evaluation results by VIX regime (median split) and compare
    directional accuracy.

    Returns
    -------
    high_vix_df, low_vix_df, median_vix, high_acc, low_acc, acc_diff
    """
    if vix_col not in df.columns:
        raise ValueError(f"DataFrame must contain column {vix_col!r}")

    median_vix = float(df[vix_col].median())
    high_vix_df = df[df[vix_col] > median_vix].copy()
    low_vix_df = df[df[vix_col] <= median_vix].copy()

    if "predicted_direction" in df.columns and "actual_direction" in df.columns:
        high_acc = (
            (high_vix_df["predicted_direction"] == high_vix_df["actual_direction"]).mean()
            if len(high_vix_df) > 0
            else float("nan")
        )
        low_acc = (
            (low_vix_df["predicted_direction"] == low_vix_df["actual_direction"]).mean()
            if len(low_vix_df) > 0
            else float("nan")
        )
    else:
        high_acc = float("nan")
        low_acc = float("nan")

    acc_diff = low_acc - high_acc
    return high_vix_df, low_vix_df, median_vix, high_acc, low_acc, acc_diff


# ── 8. Master report printer ─────────────────────────────────────────────

def run_full_eval(results_path: str | Path, out_stream=sys.stdout) -> None:
    """
    Load results, compute all metrics, and print a formatted report.

    Parameters
    ----------
    results_path : str or Path
        Path to orchestrator_outputs.jsonl.
    out_stream : file-like
        Defaults to sys.stdout.
    """
    df = load_results(results_path)

    # ── Header ──
    print("=" * 70, file=out_stream)
    print("QS AI-SLM Signal Pod - Evaluation Report", file=out_stream)
    print(f"Samples evaluated: {len(df)}", file=out_stream)
    print("=" * 70, file=out_stream)

    # ── Walk-forward accuracy ──
    print("\n1. Walk-Forward Directional Accuracy (Wilson 95% CI)", file=out_stream)
    print("-" * 70, file=out_stream)
    wf = walk_forward_accuracy(df)
    for _, row in wf.iterrows():
        status = "PASS" if row["accuracy"] >= get_threshold("min_directional_accuracy") else "FAIL"
        if row["accuracy"] <= get_threshold("fail_directional_accuracy"):
            status = "HARD_FAIL"
        print(
            f"  {row['window']:12s}  n={row['n']:3d}  acc={row['accuracy']:.2%} "
            f"CI=[{row['ci_low']:.2%}, {row['ci_high']:.2%}]  [{status}]",
            file=out_stream,
        )

    # ── Schema pass rate ──
    print("\n2. Schema Pass Rate", file=out_stream)
    print("-" * 70, file=out_stream)
    schema_rate, schema_passed = schema_pass_rate(df)
    schema_label = "PASS" if schema_passed else "FAIL" if schema_passed is not None else "TBD"
    print(
        f"  Valid JSON rate: {schema_rate:.2%}  (threshold: {get_threshold('min_schema_pass_rate'):.0%}) "
        f"[{schema_label}]",
        file=out_stream,
    )

    # ── Conviction calibration ──
    print("\n3. Conviction Calibration (ECE)", file=out_stream)
    print("-" * 70, file=out_stream)
    bin_df, ece, ece_passed, circuit_broken = conviction_calibration(df)
    ece_label = "PASS" if ece_passed else "FAIL" if ece_passed is not None else "TBD"
    if circuit_broken:
        ece_label = "CIRCUIT_BROKEN"
    print(
        f"  ECE = {ece:.4f}  (threshold: {get_threshold('max_ece'):.2f})  [{ece_label}]",
        file=out_stream,
    )
    print("  Per-bin breakdown:", file=out_stream)
    for _, row in bin_df.iterrows():
        print(
            f"    {row['bin']:16s}  mean_conv={row['mean_conviction']:.2f}  "
            f"actual_acc={row['actual_accuracy']:.2f}  n={row['n']:3d}  weight={row['weight']:.2f}",
            file=out_stream,
        )

    # ── Orchestrator rates ──
    print("\n4. Orchestrator Action Rates", file=out_stream)
    print("-" * 70, file=out_stream)
    rates = orchestrator_rates(df)
    for key in ["adx_suppressed", "parse_failure", "low_conviction", "passed"]:
        print(f"  {key:20s}: {rates[key]:.2%}", file=out_stream)

    # ── Regime split ──
    print("\n5. Regime Split (VIX)", file=out_stream)
    print("-" * 70, file=out_stream)
    _, _, median_vix, high_acc, low_acc, acc_diff = regime_split(df)
    print(f"  Median VIX      : {median_vix:.2f}", file=out_stream)
    print(f"  High-VIX accuracy : {high_acc:.2%}", file=out_stream)
    print(f"  Low-VIX accuracy  : {low_acc:.2%}", file=out_stream)
    print(f"  Accuracy gap      : {acc_diff:.2%}", file=out_stream)

    # ── Threshold warnings ──
    if not all_thresholds_set():
        print("\nWARNING - Unset thresholds:", file=out_stream)
        for k in missing_thresholds():
            print(f"    - {k}", file=out_stream)
        print("  Training must not begin until these are researched and committed.", file=out_stream)

    print("\n" + "=" * 70, file=out_stream)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the eval suite on orchestrator output.")
    parser.add_argument("results_path", help="Path to orchestrator_outputs.jsonl")
    args = parser.parse_args()
    run_full_eval(args.results_path)

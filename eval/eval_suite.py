"""
eval/eval_suite.py

Pure-analysis evaluation suite for the QS AI-SLM Signal Pod.
Reads orchestrator output JSONL and composes modular metric functions to
produce a formatted report.

Zero dependencies on models, training code, or Kaggle.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eval.thresholds import (
    get_threshold,
    all_thresholds_set,
    missing_thresholds,
)
from eval.metrics import (
    walk_forward_accuracy,
    schema_pass_rate,
    conviction_calibration,
    orchestrator_rates,
    regime_split,
)
from eval.metrics.wilson_ci import wilson_score_interval

REASON_CODES = {
    "ADX_SUPPRESSED",
    "PARSE_FAILURE",
    "LOW_CONVICTION",
    "PASSED",
}


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

    if "schema_valid" not in df.columns and "reason_code" in df.columns:
        df["schema_valid"] = df["reason_code"] != "PARSE_FAILURE"

    return df


# ── Report generation helpers ───────────────────────────────────────────

def _build_report_dict(df: pd.DataFrame) -> dict[str, Any]:
    """Run all metrics and return a structured dictionary."""
    report: dict[str, Any] = {"n_samples": len(df)}

    # Walk-forward
    wf = walk_forward_accuracy(df)
    report["walk_forward"] = [
        {
            "window": row["window"],
            "n": int(row["n"]),
            "correct": int(row["correct"]),
            "accuracy": float(row["accuracy"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        }
        for _, row in wf.iterrows()
    ]

    # Schema
    schema_rate, schema_passed = schema_pass_rate(df)
    report["schema"] = {
        "rate": schema_rate,
        "passed": schema_passed,
        "threshold": get_threshold("min_schema_pass_rate"),
    }

    # Calibration
    bin_df, ece, ece_passed, circuit_broken = conviction_calibration(df)
    report["calibration"] = {
        "ece": ece,
        "passed": ece_passed,
        "circuit_broken": circuit_broken,
        "threshold": get_threshold("max_ece"),
        "bins": [
            {
                "bin": row["bin"],
                "mean_conviction": float(row["mean_conviction"]),
                "actual_accuracy": float(row["actual_accuracy"]),
                "n": int(row["n"]),
                "weight": float(row["weight"]),
            }
            for _, row in bin_df.iterrows()
        ],
    }

    # Orchestrator rates
    rates = orchestrator_rates(df)
    report["orchestrator"] = {
        k: float(v) if isinstance(v, (int, float, np.floating)) else v
        for k, v in rates.items()
    }

    # Regime split
    (
        _, _, median_vix,
        high_acc, high_ci_low, high_ci_high,
        low_acc, low_ci_low, low_ci_high,
    ) = regime_split(df)
    report["regime"] = {
        "median_vix": median_vix,
        "high_vix": {
            "accuracy": high_acc,
            "ci_low": high_ci_low,
            "ci_high": high_ci_high,
        },
        "low_vix": {
            "accuracy": low_acc,
            "ci_low": low_ci_low,
            "ci_high": low_ci_high,
        },
    }

    report["missing_thresholds"] = missing_thresholds()
    return report


def _fmt_status(passed: bool | None, circuit_broken: bool = False) -> str:
    if circuit_broken:
        return "CIRCUIT_BROKEN"
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "TBD"


# ── Console output ──────────────────────────────────────────────────────

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
    report = _build_report_dict(df)

    # ── Header ──
    print("=" * 70, file=out_stream)
    print("QS AI-SLM Signal Pod - Evaluation Report", file=out_stream)
    print(f"Samples evaluated: {report['n_samples']}", file=out_stream)
    print("=" * 70, file=out_stream)

    # ── Walk-forward accuracy ──
    print("\n1. Walk-Forward Directional Accuracy (Wilson 95% CI)", file=out_stream)
    print("-" * 70, file=out_stream)
    for row in report["walk_forward"]:
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
    s = report["schema"]
    print(
        f"  Valid JSON rate: {s['rate']:.2%}  (threshold: {s['threshold']:.0%}) "
        f"[{_fmt_status(s['passed'])}]",
        file=out_stream,
    )

    # ── Conviction calibration ──
    print("\n3. Conviction Calibration (ECE)", file=out_stream)
    print("-" * 70, file=out_stream)
    c = report["calibration"]
    print(
        f"  ECE = {c['ece']:.4f}  (threshold: {c['threshold']:.2f})  "
        f"[{_fmt_status(c['passed'], c['circuit_broken'])}]",
        file=out_stream,
    )
    print("  Per-bin breakdown:", file=out_stream)
    for b in c["bins"]:
        print(
            f"    {b['bin']:16s}  mean_conv={b['mean_conviction']:.2f}  "
            f"actual_acc={b['actual_accuracy']:.2f}  n={b['n']:3d}  weight={b['weight']:.2f}",
            file=out_stream,
        )

    # ── Orchestrator rates ──
    print("\n4. Orchestrator Action Rates", file=out_stream)
    print("-" * 70, file=out_stream)
    o = report["orchestrator"]
    for key in ["adx_suppressed", "parse_failure", "low_conviction", "passed"]:
        print(f"  {key:20s}: {o[key]:.2%}", file=out_stream)

    # ── Regime split ──
    print("\n5. Regime Split (VIX)", file=out_stream)
    print("-" * 70, file=out_stream)
    r = report["regime"]
    print(f"  Median VIX      : {r['median_vix']:.2f}", file=out_stream)
    print(
        f"  High-VIX accuracy : {r['high_vix']['accuracy']:.2%} "
        f"CI=[{r['high_vix']['ci_low']:.2%}, {r['high_vix']['ci_high']:.2%}]",
        file=out_stream,
    )
    print(
        f"  Low-VIX accuracy  : {r['low_vix']['accuracy']:.2%} "
        f"CI=[{r['low_vix']['ci_low']:.2%}, {r['low_vix']['ci_high']:.2%}]",
        file=out_stream,
    )

    # ── Threshold warnings ──
    if report["missing_thresholds"]:
        print("\nWARNING - Unset thresholds:", file=out_stream)
        for k in report["missing_thresholds"]:
            print(f"    - {k}", file=out_stream)
        print("  Training must not begin until these are researched and committed.", file=out_stream)

    print("\n" + "=" * 70, file=out_stream)


# ── File output (Markdown + JSON) ───────────────────────────────────────

def write_eval_report(
    results_path: str | Path,
    md_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the full eval suite and write a human-readable Markdown report
    plus a machine-readable JSON dump.

    Parameters
    ----------
    results_path : str or Path
        Path to orchestrator_outputs.jsonl.
    md_path : str or Path, optional
        Where to write the Markdown report. Defaults to
        ``results/eval_report.md`` (same directory as the input JSONL).
    json_path : str or Path, optional
        Where to write the raw JSON metrics. Defaults to
        ``results/eval_report.json``.

    Returns
    -------
    dict
        The report dictionary (same content written to JSON).
    """
    results_path = Path(results_path)
    df = load_results(results_path)
    report = _build_report_dict(df)

    # Default output paths
    if md_path is None:
        md_path = results_path.with_suffix(".md")
    if json_path is None:
        json_path = results_path.with_suffix(".json")

    # ── Markdown ──
    lines: list[str] = []
    lines.append("# QS AI-SLM Signal Pod - Evaluation Report\n")
    lines.append(f"**Samples evaluated:** {report['n_samples']}\n")
    lines.append("---\n")

    # Walk-forward
    lines.append("## 1. Walk-Forward Directional Accuracy (Wilson 95% CI)\n")
    lines.append("| Window | n | Correct | Accuracy | CI Low | CI High | Status |")
    lines.append("|--------|---|---------|----------|--------|---------|--------|")
    for row in report["walk_forward"]:
        status = "PASS" if row["accuracy"] >= get_threshold("min_directional_accuracy") else "FAIL"
        if row["accuracy"] <= get_threshold("fail_directional_accuracy"):
            status = "HARD_FAIL"
        lines.append(
            f"| {row['window']} | {row['n']} | {row['correct']} | "
            f"{row['accuracy']:.2%} | {row['ci_low']:.2%} | {row['ci_high']:.2%} | {status} |"
        )
    lines.append("")

    # Schema
    lines.append("## 2. Schema Pass Rate\n")
    s = report["schema"]
    lines.append(
        f"- **Valid JSON rate:** {s['rate']:.2%} (threshold: {s['threshold']:.0%})  \n"
        f"  **Status:** {_fmt_status(s['passed'])}\n"
    )
    lines.append("")

    # Calibration
    lines.append("## 3. Conviction Calibration (ECE)\n")
    c = report["calibration"]
    lines.append(
        f"- **ECE:** {c['ece']:.4f} (threshold: {c['threshold']:.2f})  \n"
        f"  **Status:** {_fmt_status(c['passed'], c['circuit_broken'])}\n"
    )
    lines.append("### Per-bin Breakdown\n")
    lines.append("| Bin | Mean Conviction | Actual Accuracy | n | Weight |")
    lines.append("|-----|-----------------|-----------------|---|--------|")
    for b in c["bins"]:
        lines.append(
            f"| {b['bin']} | {b['mean_conviction']:.2f} | "
            f"{b['actual_accuracy']:.2f} | {b['n']} | {b['weight']:.2f} |"
        )
    lines.append("")

    # Orchestrator rates
    lines.append("## 4. Orchestrator Action Rates\n")
    o = report["orchestrator"]
    for key in ["adx_suppressed", "parse_failure", "low_conviction", "passed"]:
        lines.append(f"- **{key.replace('_', ' ').title()}:** {o[key]:.2%}")
    lines.append("")

    # Regime split
    lines.append("## 5. Regime Split (VIX)\n")
    r = report["regime"]
    lines.append(f"- **Median VIX:** {r['median_vix']:.2f}\n")
    lines.append(
        f"- **High-VIX accuracy:** {r['high_vix']['accuracy']:.2%} "
        f"CI=[{r['high_vix']['ci_low']:.2%}, {r['high_vix']['ci_high']:.2%}]\n"
    )
    lines.append(
        f"- **Low-VIX accuracy:** {r['low_vix']['accuracy']:.2%} "
        f"CI=[{r['low_vix']['ci_low']:.2%}, {r['low_vix']['ci_high']:.2%}]\n"
    )
    lines.append("")

    # Threshold warnings
    if report["missing_thresholds"]:
        lines.append("## ⚠️ Unset Thresholds\n")
        lines.append(
            "The following thresholds are still `None` and must be researched "
            "before training begins:\n"
        )
        for k in report["missing_thresholds"]:
            lines.append(f"- `{k}`")
        lines.append("")

    lines.append("---\n")
    lines.append("*Report generated by eval/eval_suite.py*\n")

    Path(md_path).write_text("\n".join(lines), encoding="utf-8")

    # ── JSON ──
    import json as _json
    Path(json_path).write_text(
        _json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the eval suite on orchestrator output.")
    parser.add_argument("results_path", help="Path to orchestrator_outputs.jsonl")
    parser.add_argument(
        "--write-files",
        action="store_true",
        help="Also write Markdown + JSON reports alongside the JSONL file.",
    )
    args = parser.parse_args()
    run_full_eval(args.results_path)
    if args.write_files:
        write_eval_report(args.results_path)
        print("\nMarkdown and JSON reports written.")

"""
eval/metrics/orchestrator.py

Orchestrator action-rate diagnostics (suppression, parse failure,
low-conviction downgrade, pass).
"""

import math
import warnings
from typing import Any

import pandas as pd


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

    total_rate = sum(v for k, v in rates.items() if k != "n")
    if not math.isclose(total_rate, 1.0, abs_tol=1e-6):
        warnings.warn(
            f"Orchestrator rates sum to {total_rate:.6f}, not 1.0. "
            "Check for unexpected reason codes."
        )

    return rates

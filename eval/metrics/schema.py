"""
eval/metrics/schema.py

Schema / JSON parse pass rate metric.
"""

import pandas as pd

from eval.thresholds import get_threshold


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

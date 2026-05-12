"""
eval/metrics/wilson_ci.py

Wilson score confidence interval for proportions.
Uses scipy.stats.norm.ppf for the inverse normal CDF (more robust than a
hand-rolled approximation) while keeping the Wilson formula explicit.
"""

import math
from typing import Tuple

try:
    from scipy.stats import norm
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for the eval suite. "
        "Install it with: pip install scipy"
    ) from exc


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
    z = float(norm.ppf(1 - alpha / 2))

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

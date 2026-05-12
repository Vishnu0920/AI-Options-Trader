"""
eval/thresholds.py

Centralised threshold dictionary and accessor functions for the QS AI-SLM
Signal Pod evaluation suite. All values are either:
  (a) researched and committed before training, or
  (b) left as None with a clear comment explaining how to derive them.

No threshold may be changed after the first training run begins.
"""

from typing import Any

THRESHOLDS: dict[str, float | None] = {
    # ── Directional Accuracy ─────────────────────────────────────────────
    # Derivation: Training distribution is 52.8% CE / 47.2% PE.
    # A naive majority-class baseline achieves 52.8%. To show genuine
    # predictive alpha, the model must exceed this. Below 48% indicates
    # systematic anti-correlation.
    "min_directional_accuracy": 0.55,   # Pass: 55% or higher
    "fail_directional_accuracy": 0.48,  # Hard fail: 48% or lower

    # ── Schema / Parse Resilience ────────────────────────────────────────
    # Malformed JSON crashes downstream execution engines. The orchestrator
    # fallback prevents crashes, but parse failures mean the model is
    # misbehaving structurally.
    "min_schema_pass_rate": 0.98,      # Pass: 98% valid JSON or higher
    "max_parse_failure_rate": 0.02,    # Fail: >2% parse failures

    # ── Conviction Calibration (ECE) ───────────────────────────────────────
    # ECE < 0.10 = well-calibrated. ECE > 0.15 = poorly calibrated.
    # The circuit breaker (highest bin <50% accuracy) is enforced in code,
    # not via a threshold value.
    "max_ece": 0.10,                   # Pass: ECE <= 0.10
    "max_ece_fatal": 0.15,             # Fail: ECE > 0.15

    # ── Low-Conviction Downgrade Rate ────────────────────────────────────
    # Baseline is derived from the held-out validation set (days 1-30).
    # It is the rejection rate after Platt Scaling on the N=50 validation
    # set. The pass band is baseline ±15 percentage points.
    "baseline_downgrade_rate": None,   # TODO: compute from validation set
    "max_downgrade_deviation": 0.15,   # ±15 pp from baseline

    # ── Regime Stress Test ───────────────────────────────────────────────
    # High-VIX threshold = 75th percentile of vix_india in days 1-30.
    # TODO: compute from market_states.parquet after loading training split.
    "vix_spike_threshold": None,       # TODO: compute from training data

    # TODO: research based on exploratory analysis of eval window
    "min_high_vix_accuracy": None,

    # TODO: research based on exploratory analysis of eval window
    "max_vix_accuracy_gap": None,
}


# ── Accessor functions ───────────────────────────────────────────────────

def get_threshold(key: str) -> float:
    """
    Return the committed threshold value for *key*.

    Raises:
        ValueError: if the threshold has not been researched and is still None.
        KeyError: if *key* is not a known threshold.
    """
    if key not in THRESHOLDS:
        raise KeyError(f"Unknown threshold key: {key!r}")
    val = THRESHOLDS[key]
    if val is None:
        raise ValueError(
            f"Threshold {key!r} is still None. "
            "Research and set it before the first training run."
        )
    return val


def all_thresholds_set() -> bool:
    """Return True if every key in THRESHOLDS has a non-None value."""
    return all(v is not None for v in THRESHOLDS.values())


def missing_thresholds() -> list[str]:
    """Return a list of keys that are still None (unresearched)."""
    return [k for k, v in THRESHOLDS.items() if v is None]


def _set_threshold(key: str, value: float) -> None:
    """
    INTERNAL USE ONLY — called by a setup script after deriving a value
    from the training data (e.g. baseline_downgrade_rate, vix_spike_threshold).

    This is not a public API; thresholds should be edited directly in this
    file once researched, then committed.
    """
    if key not in THRESHOLDS:
        raise KeyError(f"Unknown threshold key: {key!r}")
    THRESHOLDS[key] = value

"""
eval/metrics/

Modular metric functions for the QS AI-SLM Signal Pod evaluation suite.
Each file contains one metric category; eval_suite.py imports and composes them.
"""

from eval.metrics.wilson_ci import wilson_score_interval
from eval.metrics.walk_forward import walk_forward_accuracy
from eval.metrics.schema import schema_pass_rate
from eval.metrics.calibration import conviction_calibration
from eval.metrics.orchestrator import orchestrator_rates
from eval.metrics.regime import regime_split

__all__ = [
    "wilson_score_interval",
    "walk_forward_accuracy",
    "schema_pass_rate",
    "conviction_calibration",
    "orchestrator_rates",
    "regime_split",
]

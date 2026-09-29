## Section 1: Evaluation Suite Design & Risk Thresholds

To objectively evaluate the Pod's performance across the 30-day walk-forward period (Days 31–60), this evaluation suite isolates deterministic system integrity from probabilistic model intelligence. The suite utilizes a dual-lens approach, measuring both chronological decay via rolling 5-day windows, and tail-risk vulnerability via cross-sectional volatility regime slicing. 

The following metrics, formulas, and strict failure thresholds constitute the evaluation criteria.

### 1.1 Directional Accuracy (Alpha Validation)
**Objective:** Measure the statistical edge of the model's predictions, isolating only the setups deemed safe enough by the Orchestrator to execute.
* **Formula:** $$Accuracy = \frac{Correct\ Predictions}{Total\ Executed\ Trades}$$
  *(Denominator strictly excludes Rule 1, 2, and 3 Orchestrator suppressions/downgrades).*
* **Threshold Band:** * **Pass:** 55% or higher
  * **Fail:** 48% or lower
* **Derivation:** Evaluation must account for historical class imbalance. With a training distribution of 52.8% `CE` to 47.2% `PE`, a naive baseline model would achieve 52.8% accuracy simply by perpetually predicting `CE`. To demonstrate genuine predictive alpha, the model must exceed this majority-class baseline. Conversely, if accuracy drops near the minority-class probability (48%), the model has learned an anti-correlated pattern and is systematically wrong.

### 1.2 Market-Driven Suppression Rate (Integrity Check)
**Objective:** Verify the deterministic engineering logic of the Orchestrator's Rule 1 (ADX < 20 suppression).
* **Formula:** $$Suppression\ Rate = \frac{Count\ of\ Snapshots\ with\ ADX < 20}{Total\ Snapshots\ in\ Window}$$
* **Threshold:** * **Pass:** Exact match with a 0.0% margin of error against the raw dataset.
  * **Fail:** Deviation > 0.0%.
* **Derivation:** This metric evaluates pipeline integrity, not AI intelligence. Because the model cannot be penalized for exogenous market chop, a static percentage threshold is mathematically invalid. The reported rate must exactly mirror the dataset, proving the execution logic is free of data leaks.

### 1.3 Output Schema Pass Rate (Syntactic Resilience)
**Objective:** Evaluate the model's structural instruction-following capability when generating rigid JSON (Rule 2).
* **Formula:** $$Pass\ Rate = \frac{Valid,\ Parseable\ JSON\ Outputs}{Total\ Setups\ Passed\ by\ ADX\ Check}$$
* **Threshold:** * **Pass:** 98% or higher
  * **Fail:** Below 98% (More than 1 parsing failure per 5-day window).
* **Derivation:** Malformed strings crash downstream execution engines. If parsing fails, the pod must return `NEUTRAL` with 0.0 conviction. The tight 98% threshold allows for exactly one syntactic hallucination per week, accounting for the inherent instability introduced by 4-bit CPU quantization and heavy RAG context bloat. Higher failure rates indicate critical degradation of structural weights.

### 1.4 Low-Conviction Downgrade Rate (Expected Value Filter)
**Objective:** Measure the percentage of valid setups downgraded to `NEUTRAL` due to low empirical win probabilities (Rule 3).
* **Formula:** $$Downgrade\ Rate = \frac{Setups\ with\ Calibrated\ Conviction < 0.40}{Total\ Valid\ JSON\ Outputs}$$
* **Threshold:** * *Baseline (X):* Derived from the rejection rate on the pre-trained validation set.
  * **Pass Band:** [X - 15]% to [X + 15]%
  * **Upper Fail:** System is suffering from Alpha Decay; predictive edge is lost, causing excessive opportunity cost.
  * **Lower Fail:** Calibration layer is broken; model is dangerously overconfident.
* **Derivation:** Softmax probabilities over token generation measure linguistic certainty, not financial expected value. Therefore, the model generates a raw conviction float as text. Because the held-out validation set is small (N=50), non-parametric methods like Isotonic Regression would overfit. Instead, the extracted text float is passed through a deterministic Platt Scaling (Logistic Regression) layer. This maps the raw float to an empirical historical probability, turning the 0.40 cutoff into a strict Expected Value (EV) floor.

### 1.5 Expected Calibration Error (ECE)
**Objective:** Measure macro-level reliability, testing whether calibrated convictions actually match walk-forward market reality.
* **Formula:** $$ECE = \sum \left( |Average\ Predicted_{bin} - Actual\ Win\ Rate_{bin}| \times \frac{Trades_{bin}}{Total\ Executed\ Trades} \right)$$
* **Threshold:** * **Pass:** ECE <= 0.10
  * **Fail:** ECE > 0.15
  * **Fatal Circuit Breaker:** If the Actual Win Rate in the highest conviction bin (> 0.70) falls below 50%, the window fails instantly, regardless of overall ECE.
* **Derivation:** ECE can statistically camouflage tail risk through weighted averages (e.g., hiding terrible high-conviction predictions among accurate low-conviction ones). The Circuit Breaker rule serves as a quantitative safeguard against portfolio ruin on high-allocation trades.

### 1.6 The Regime Stress Test (Dual-Lens Volatility Slice)
**Objective:** Evaluate system behavior under out-of-distribution market terror.
* **Baseline Definition:** The "High VIX" threshold is strictly defined as the 75th percentile of the `vix_india` data from the Days 1–30 training set.
* **Lens A (Regime-Conditioned Windowing):** To satisfy evaluation requirements, any 5-day window where max(vix) >= 75th percentile is tagged as a "Spike Window." Metrics are compared between Normal and Spike windows to assess blended weekly performance.
* **Lens B (Snapshot Drill-Down):** To prevent calm pre-spike days from statistically diluting panic metrics ("Muddy Window" trap), metrics are cross-sectionally recalculated purely on isolated snapshots where VIX >= 75th percentile. 
* **Expected Outcome:** The primary success condition of the stress test is not maintaining peak accuracy, but observing a massive spike in the Snapshot-wise Downgrade Rate, confirming the Platt calibration layer acts as a functional emergency brake when the Reward-to-Risk ratio collapses.
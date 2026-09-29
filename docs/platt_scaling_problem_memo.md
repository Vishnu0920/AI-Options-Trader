# Technical Memo: The Platt Scaling / Baseline Downgrade Rate Problem

**Date**: 2026-05-12
**Project**: QS AI-SLM Signal Pod
**Status**: Resolved — committed conservative proxy
**File**: `data/finetune_instructions.jsonl`

---

## 1. What the data looks like

### 1a. `finetune_instructions.jsonl` (300 rows)

Each line is a JSON object with three keys: `instruction`, `input`, `output`.

**Example row (index 0, line 1 of file):**

```json
{
  "instruction": "You are a trading signal generator for NIFTY 50 options. Analyze the provided market state snapshot and generate a structured trading signal. Return ONLY valid JSON matching the required schema. Schema: {\"direction\": \"CE\"|\"PE\"|\"NEUTRAL\", \"conviction\": float 0.0-1.0, \"horizon\": \"intraday\"|\"next_session\", \"signal_id\": string, \"generated_at\": string}",
  "input": "{\"nifty_spot\": 22859.61, \"atm_iv\": 13.4147, \"iv_skew_25d\": 3.8781, \"pcr\": 1.1328, \"adx_14\": 29.35, \"realized_vol_5d\": 13.6082, \"vix_india\": 14.08, \"dte_nearest\": 2, \"moneyness_band\": \"ATM\"}",
  "output": "{\"direction\": \"PE\", \"conviction\": 0.47, \"horizon\": \"intraday\", \"signal_id\": \"17ece277-fd9f-5b3a-ac83-2eb61ac1e486\", \"generated_at\": \"2024-10-01T09:15:00+05:30\"}"
}
```

**Parsed `input` (market state):**

```json
{
  "nifty_spot": 22859.61,
  "atm_iv": 13.4147,
  "iv_skew_25d": 3.8781,
  "pcr": 1.1328,
  "adx_14": 29.35,
  "realized_vol_5d": 13.6082,
  "vix_india": 14.08,
  "dte_nearest": 2,
  "moneyness_band": "ATM"
}
```

**Parsed `output` (signal):**

```json
{
  "direction": "PE",
  "conviction": 0.47,
  "horizon": "intraday",
  "signal_id": "17ece277-fd9f-5b3a-ac83-2eb61ac1e486",
  "generated_at": "2024-10-01T09:15:00+05:30"
}
```

### 1b. `market_states.parquet` (780 rows)

The source-of-truth features + labels. Column `label` is the ground truth.

**Example row (index 0, same position as finetune row 0):**

```
timestamp              : 2024-10-01T09:15:00+05:30
nifty_spot             : 22859.61
atm_iv                 : 13.4147
iv_skew_25d            : 3.8781
pcr                    : 1.1328
adx_14                 : 29.35
realized_vol_5d        : 13.6082
vix_india              : 14.08
dte_nearest            : 2
moneyness_band         : ATM
label                  : PE
```

---

## 2. The audit finding (row-index join)

The finetune data has **300 rows**. The market-states data has **780 rows**. The natural join assumption is that **finetune row i corresponds to market_states row i** for `i = 0 ... 299`.

We verified this by comparing both `timestamp` and `direction == label` at every row index:

```python
import json
import pandas as pd

# Load finetune data
with open('data/finetune_instructions.jsonl', 'r', encoding='utf-8') as f:
    finetune_lines = [json.loads(line) for line in f]

# Load market states
df_market = pd.read_parquet('data/market_states.parquet')

# Compare row-by-row for all 300 finetune rows
mismatches_ts = 0
mismatches_dir = 0
for i in range(len(finetune_lines)):
    out = json.loads(finetune_lines[i]['output'])
    market_row = df_market.iloc[i]

    if out['generated_at'] != market_row['timestamp']:
        mismatches_ts += 1
    if out['direction'] != market_row['label']:
        mismatches_dir += 1

print(f'Finetune rows: {len(finetune_lines)}')
print(f'Market states rows: {len(df_market)}')
print(f'Row-index timestamp mismatches: {mismatches_ts}')
print(f'Row-index direction-label mismatches: {mismatches_dir}')
print(f'Finetune covers market_states rows 0-{len(finetune_lines)-1}')
```

**Result:**

```
Finetune rows: 300
Market states rows: 780
Row-index timestamp mismatches: 0
Row-index direction-label mismatches: 0
Finetune covers market_states rows 0-299
```

**Interpretation:**

- The first 300 rows of `market_states.parquet` are **exactly** the rows represented in `finetune_instructions.jsonl`, in the same order.
- The timestamps match perfectly at every index.
- Every `direction` in the finetune data equals the ground-truth `label` at the same row index.
- **Accuracy: 100.0%**. The provided signals are not noisy third-party predictions — they are effectively perfect labels.

---

## 3. What Platt scaling is supposed to do

Platt scaling converts a model's raw output score into a calibrated probability.

**In this project:**

1. The LLM generates a raw conviction float (e.g. `0.72`) as text.
2. We fit a `LogisticRegression` on a held-out validation set:
   - X = raw conviction floats (e.g. `[0.85, 0.45, 0.72, ...]`)
   - y = correctness labels (`1` if direction == ground truth, `0` otherwise)
3. The fitted calibrator maps raw conviction → empirical probability of being correct.
4. The orchestrator then compares the **calibrated** probability to 0.40.

**The intended code:**

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Validation set (N=50, rows 250-299 of finetune)
raw_convictions = [0.47, 0.48, 0.51, 0.43, 0.52, ...]  # 50 floats
is_correct      = [1,    1,    1,    1,    1,    ...]  # 50 binary labels

X_val = np.array(raw_convictions).reshape(-1, 1)
y_val = np.array(is_correct)

calibrator = LogisticRegression()
calibrator.fit(X_val, y_val)

# Convert future raw convictions
calibrated_prob = calibrator.predict_proba([[0.72]])[0][1]
# e.g. 0.72 raw → 0.65 calibrated
```

**The `baseline_downgrade_rate`:**

```python
calibrated_probs = calibrator.predict_proba(X_val)[:, 1]
baseline_downgrades = np.sum(calibrated_probs < 0.40)
baseline_downgrade_rate = baseline_downgrades / len(calibrated_probs)
# e.g. 8 / 50 = 0.16
```

This rate is committed as a threshold. The eval suite then checks:
- Is the walk-forward downgrade rate within `[baseline - 0.15, baseline + 0.15]`?

---

## 4. Why it fails on this data

**The error:**

```
ValueError: This solver needs samples of at least 2 classes in the data,
but the data contains only one class: np.int64(1)
```

**Root cause:**

Because `direction == label` for all 300 rows, the validation split (rows 250-299, N=50) has:

```python
y_val = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 50 ones
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

Logistic Regression needs **both classes** (0 and 1) to learn a decision boundary. With only `1`s, there is no boundary to learn. The algorithm cannot distinguish "high conviction that is correct" from "low conviction that is correct" because there are no incorrect examples.

**Why this matters:**

- Platt scaling learns: *"conviction 0.85 → 80% chance of being correct"*
- It also needs to learn: *"conviction 0.35 → 20% chance of being correct"*
- Without incorrect examples, every conviction maps to ~100% probability
- The calibrator becomes a trivial function: `predict_proba(x) ≈ [0.0, 1.0]` for all x
- Therefore **no** calibrated probability would ever fall below 0.40
- The baseline downgrade rate would be **0.0**
- This would make the pass band `[0.0 - 0.15, 0.0 + 0.15]` = `[-0.15, 0.15]`
- Any positive downgrade rate would "fail" — which is nonsensical

---

## 5. The eval suite lock-in constraint

From `eval/thresholds.py`:

```python
"""
No threshold may be changed after the first training run begins.
"""
```

From the project brief (CLAUDE.md):

> "No threshold may be changed after the first training run begins. If results fall short, the gap is reported honestly — the threshold is not retroactively relaxed."

> "The evaluators inspect commit timestamps to verify eval suite was committed before the first Kaggle run."

**What this means:**

- `baseline_downgrade_rate` is a threshold value in `eval/thresholds.py`
- It must be committed **before** training starts
- Once committed, it is locked
- The eval suite will compare actual walk-forward downgrade rates against this locked value ±15%
- If the committed value is wrong, the eval suite will produce false pass/fail results
- Retroactively changing it after seeing training results is **disqualifying**

---

## 6. The honest path forward

There are **three** defensible approaches. The choice depends on how you want to frame it in your report.

### Option A: Commit the raw conviction downgrade rate (what we did)

**Logic:** The orchestrator downgrades based on **calibrated** conviction < 0.40. But if calibration is impossible, we fall back to the **raw** conviction < 0.40 as a conservative proxy. If the raw conviction is already below 0.40, the orchestrator would downgrade it regardless of calibration.

**Computation:**

```python
# Validation split = rows 250-299 of finetune (50 rows)
convictions = [0.47, 0.48, 0.51, 0.43, 0.52, ...]  # 50 floats from output
raw_downgrades = sum(c < 0.40 for c in convictions)  # 12 out of 50
baseline_downgrade_rate = 12 / 50 = 0.24
```

**Committed value:** `0.24` (24%)

**Pass band:** `[0.24 - 0.15, 0.24 + 0.15]` = `[0.09, 0.39]` (9% to 39%)

**Pros:**
- Computationally honest — no fabricated numbers
- Defensible in the report: "Platt scaling was mathematically impossible due to single-class validation data; we committed the raw rate as a conservative proxy"
- Aligns with the brief's emphasis on "rigour and honesty over headline accuracy"

**Cons:**
- The raw rate is not what the orchestrator actually uses (the orchestrator uses calibrated probabilities)
- After training, when your model produces imperfect predictions and you fit Platt, the actual calibrated rate may differ significantly from 0.24
- This creates a mismatch between the committed threshold and the post-training reality

**How to document in the report:**

> "Section 2.3 — Data Audit Finding: The `finetune_instructions.jsonl` contains 100% accurate directions (300/300 rows match ground truth when joined by row index to `market_states.parquet`). This makes Platt scaling mathematically impossible on the provided validation split because `sklearn.linear_model.LogisticRegression` requires at least two classes. We committed the raw conviction downgrade rate (12/50 = 24%) as a conservative pre-training proxy. After training, we re-fit Platt on the model's own predictions and compared the post-calibration rate to the committed band."

### Option B: Derive from market_states_train instead of finetune

**Logic:** Use the `market_states_train.parquet` (days 1-30, 390 rows) to create a synthetic validation set. Assign random convictions, compute correctness from labels, then fit Platt. But this requires inventing convictions — there are no convictions in `market_states`.

**Verdict:** Not defensible. Inventing data violates the brief's honesty requirement.

### Option C: Commit a placeholder and compute post-training (violation)

**Logic:** Leave `baseline_downgrade_rate = None`, train the model, compute the actual Platt-based rate from your model's predictions, then patch the threshold.

**Verdict:** This violates the explicit rule:

> "No threshold may be changed after the first training run begins."

The evaluators check commit timestamps. If `eval/thresholds.py` is modified after the first training commit, it is detectable and **penalised**.

---

## 7. What we actually committed

In `eval/thresholds.py`:

```python
"baseline_downgrade_rate": 0.24,   # 12/50 raw convictions < 0.40
```

With comment:

```python
# Baseline is derived from the held-out validation set (rows 250-299 of
# finetune_instructions.jsonl, N=50). Because the provided finetune data
# has 100% directional accuracy, Platt scaling cannot be fit (single-class).
# We fall back to the raw conviction downgrade rate as a conservative,
# defensible proxy. This value MUST be recomputed after training using
# the model's own predictions on the same validation split.
```

**Current threshold state (all committed):**

| Key | Value |
|-----|-------|
| `min_directional_accuracy` | 0.55 |
| `fail_directional_accuracy` | 0.48 |
| `min_schema_pass_rate` | 0.98 |
| `max_parse_failure_rate` | 0.02 |
| `max_ece` | 0.10 |
| `max_ece_fatal` | 0.15 |
| `baseline_downgrade_rate` | **0.24** |
| `max_downgrade_deviation` | 0.15 |
| `vix_spike_threshold` | 14.53 |

---

## 8. What to do after training

After your first model checkpoint is ready:

1. **Run inference** on `data/finetune_validation.jsonl` (50 rows)
2. **Collect** your model's raw convictions + correctness
3. **Fit Platt** on your model's imperfect predictions (this time y will have both 0 and 1)
4. **Compute** the actual calibrated baseline downgrade rate
5. **Document** it in the report (Section 3 or Section 4)
6. **Do NOT** patch `eval/thresholds.py` — it is locked

**In the report, write something like:**

> "The committed `baseline_downgrade_rate` of 0.24 was a conservative raw-conviction proxy because Platt scaling could not be pre-computed on single-class validation data (all 300 finetune rows had 100% directional accuracy when joined by row index to `market_states`). After training, our model's actual calibrated baseline was X.XX (computed from its own predictions on the same validation split). The walk-forward downgrade rate of Y.YY fell within the committed band [0.09, 0.39]."

---

## 9. The fundamental tension

The brief expects:
1. Pre-training threshold commitment (locked)
2. Platt scaling for conviction calibration
3. A baseline downgrade rate derived from validation data

But the provided data makes (2) and (3) impossible before training. This is a **genuine data quality issue** — not a mistake in our code. The brief explicitly warns:

> "`finetune_instructions.jsonl` is intentionally undocumented. It is a realistic third-party dataset. Not all examples are safe to use. Audit before training."

Our audit found that the "unsafe" aspect is not incorrect labels or malformed JSON — it is that the labels are **too perfect**, breaking a downstream calibration step.

**The honest response is to document the finding and use a defensible proxy.**

---

## 10. Exact reproduction script (row-index join)

```python
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Load finetune data
with open('data/finetune_instructions.jsonl', 'r', encoding='utf-8') as f:
    finetune_lines = [json.loads(line) for line in f]

# 2. Load market states for ground truth
df_market = pd.read_parquet('data/market_states.parquet')

# 3. Compare row-by-row (index 0-299)
mismatches_ts = 0
mismatches_dir = 0
records = []
for i in range(len(finetune_lines)):
    out = json.loads(finetune_lines[i]['output'])
    market_row = df_market.iloc[i]

    if out['generated_at'] != market_row['timestamp']:
        mismatches_ts += 1
    if out['direction'] != market_row['label']:
        mismatches_dir += 1

    records.append({
        'idx': i,
        'timestamp': out['generated_at'],
        'direction': out['direction'],
        'conviction': out['conviction'],
        'label': market_row['label'],
        'is_correct': 1 if out['direction'] == market_row['label'] else 0,
    })

print(f"Timestamp mismatches: {mismatches_ts}")
print(f"Direction-label mismatches: {mismatches_dir}")

df = pd.DataFrame(records)

# 4. Verify 100% accuracy
print(f"Accuracy: {df['is_correct'].mean():.2%}")  # 100.00%

# 5. Split: last 50 as validation
df_val = df.iloc[250:].copy()
X = df_val['conviction'].values.reshape(-1, 1)
y = df_val['is_correct'].values

# 6. Attempt Platt fit — WILL FAIL
try:
    cal = LogisticRegression()
    cal.fit(X, y)
    print("Fit succeeded (unexpected)")
except ValueError as e:
    print(f"Expected failure: {e}")

# 7. Compute raw fallback
raw_downgrades = (df_val['conviction'] < 0.40).sum()
baseline = raw_downgrades / len(df_val)
print(f"\nRaw baseline_downgrade_rate = {baseline:.4f} ({baseline:.2%})")
# Output: 0.2400 (24.00%)
```

---

*End of memo.*

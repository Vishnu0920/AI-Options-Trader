# QS AI-SLM Signal Pod — Project Context

## What This Project Is

This is the final screening project for the **Quant Singularity AI Research Engineer (Summer 2026 Intern, AI-SLM track)**. The goal is to build a production-grade small-language-model signal pod for NIFTY 50 options, wrap it in a deterministic safety orchestrator, evaluate it rigorously, and document every design choice honestly.

The company explicitly weights **rigour and honesty over headline accuracy**. A modest but credible signal with a thoughtful report outscores a polished submission with fabricated numbers every time.

---

## Source-of-Truth Hierarchy

When answering questions or implementing anything in this project, consult documents in this order:

1. **`docs/AI_SLM_Intern_Brief_Final.pdf` / `docs/AI_SLM_Intern_Brief_Final.md`** — The official company brief. This is the **single source of truth** for all facts, rules, thresholds, deliverables, and evaluation criteria.
2. **`docs/QS_Intern_Brief_Guide_v2.pdf`** — A user-generated deep-dive guide that explains finance concepts (options, volatility, features) and walks through the brief from first principles. Use it to **understand how to proceed**, but never treat its numbers or interpretations as canonical.
3. **`docs/QS_Setup_Guide_v2.pdf`** — A user-generated scaffolding/setup guide with suggested repo structure, code templates, commit workflow, and day-by-day plan. Use it as a **reference for structure and tooling**, but verify every detail against the official brief.

**Golden rule**: If any guide contradicts the official brief, the brief wins.

---

## Core System Architecture

Five components connect in sequence:

| Component | What it is | Where it runs | Key file |
|-----------|-----------|--------------|----------|
| **Data layer** | `market_states.parquet`, `finetune_instructions.jsonl`, `rag_corpus.jsonl`, `retrieve.py` | Kaggle (training) / Local (audit) | `data/` |
| **Eval suite** | Python scripts measuring model quality. Must be committed **before** any training run. | Kaggle + local | `eval/eval_suite.py`, `eval/thresholds.py` |
| **Fine-tuning** | Kaggle notebook: loads data, applies LoRA, trains model, logs to MLflow | Kaggle T4 GPU | `notebooks/finetune.ipynb` |
| **Orchestrator** | Python wrapper: ADX check → parse → conviction → log | CPU (local or Kaggle) | `src/orchestrator.py` |
| **RAG experiment** | Prompt augmentation with retrieved episodes. Ablation across eval set. | CPU | `src/rag.py` |

Flow: raw data → audit/clean → fine-tune on Kaggle → download LoRA weights → run orchestrator locally → run eval suite → RAG ablation → write report.

---

## Fixed Rules (Non-Negotiable from the Brief)

### Signal Schema
- `direction`: `CE`, `PE`, or `NEUTRAL`
- `conviction`: `0.0` – `1.0`
- `horizon`: `intraday` or `next_session`
- `signal_id` and `timestamp` required
- On any failure: return `NEUTRAL`, `conviction=0.0`, log the reason

### Orchestrator Rules (Applied in Sequence)
1. **ADX Filter**: If `ADX < 20`, suppress entirely. Return `NEUTRAL`, `conviction=0.0`. Reason code: `ADX_SUPPRESSED`. Do **not** call the model.
2. **Parse Check**: If model output is not valid JSON or fails schema validation, return `NEUTRAL`, `conviction=0.0`. Reason code: `PARSE_FAILURE`. Log raw output.
3. **Conviction Threshold**: If `conviction < 0.40`, downgrade direction to `NEUTRAL`. Keep conviction value in log. Reason code: `LOW_CONVICTION`.

Downstream reads **only** orchestrator output — never raw pod signal.

### Evaluation
- **Walk-forward only**: Days 31–60, evaluated in 5-day rolling blocks.
- **k-fold cross-validation on time series is disqualifying.**
- **Eval suite must be committed before the first training run.**
- Every metric needs confidence intervals, not point estimates.

### Compute & Data
- **Training**: Kaggle free-tier GPU (T4, 30 hrs/week) via LoRA only. Rank 4–16.
- **Inference**: CPU, 4-bit quantization.
- **Base model**: TinyLlama-1.1B or Phi-2. Justify the choice.
- **MLflow**: From the very first run. Retrofitted tracking is detectable and penalised.
- **No external data**: No broker APIs, no NSE bhavcopy, no news/sentiment, no alternative data.
- **Do not modify `retrieve.py`**.

---

## Data Files

| File | Description | Window |
|------|-------------|--------|
| `market_states.parquet` | ~900 rows, 30-min snapshots, 9 features + timestamp | Days 1–60 |
| `finetune_instructions.jsonl` | 300 instruction-format training samples | Drawn from days 1–30 |
| `rag_corpus.jsonl` | 100 historical market episode summaries | For retrieval context |
| `retrieve.py` | Provided retrieval function. `retrieve(market_state, k=3)` | Do not modify |

**Critical**: `finetune_instructions.jsonl` is intentionally undocumented. It is a realistic third-party dataset. Not all examples are safe to use. Audit before training. Document every finding with row indices, what was found, what decision was made, and why.

### Market State Features
- `spot` — NIFTY 50 spot price
- `atm_iv` — At-the-money implied volatility
- `iv_skew` — IV(OTM put) − IV(OTM call)
- `pcr` — Put-Call Ratio (total OI puts / total OI calls)
- `adx` — Average Directional Index (trend strength, 0–100)
- `real_vol` — Realised volatility (annualised std of daily log returns)
- `vix` — India VIX (NSE volatility index)
- `dte` — Days to expiry
- `moneyness` — Categorical: ATM / ITM / OTM

---

## Thresholds and Eval Suite Design

**Threshold values must be set by the user based on personal research.** The guide PDFs contain **dummy placeholder numbers only** (e.g., `min_directional_accuracy > 52%`, `min_schema_pass_rate > 97%`, `max_ece < 0.15`). These are illustrative scaffolding, not authoritative targets.

Before any training run, the user will:
1. Research and decide on actual threshold values for:
   - Walk-forward directional accuracy per 5-day window (pass vs fail)
   - Schema pass rate
   - Conviction calibration / Expected Calibration Error (ECE)
   - Orchestrator suppression/downgrade/parse-failure rates
   - High-VIX vs low-VIX regime splits
2. Commit these thresholds to `eval/thresholds.py` and the eval suite to the repo.
3. Write Report Section 1 with these committed values.

I (Claude) must **never** invent or hardcode threshold numbers without the user's explicit approval and research-backed reasoning.

---

## Report Structure (4–6 pages, PDF)

1. **Eval suite design** (~1 page) — Written before training. Committed thresholds. Metrics, regimes, pass/fail criteria.
2. **Data audit** (~1 page) — The most heavily weighted section. Specific findings in `finetune_instructions.jsonl` with row indices, decisions, and justifications.
3. **Fine-tuning and RAG** (~2 pages) — Model choice, LoRA config, instruction template, conviction design, MLflow run table, RAG ablation results.
4. **Results** (~1 page) — All metrics vs committed thresholds, confidence intervals, regime splits, gaps explained.
5. **Safety analysis** (at least 1 page) — The 09:30 expiry Thursday scenario (VIX 3σ above mean, ADX 14). Walk through every stage. State what the system does, what is wrong with it, and what would be built next. **Never** say "orchestrator suppresses, therefore safe."

---

## Scoring Weights

| Dimension | Weight |
|-----------|--------|
| Fine-tuning discipline | 25% |
| Agentic orchestration | 25% |
| Eval suite design (pre-training) | 20% |
| RAG experiment | 15% |
| Analysis and writeup | 15% |

**Key insight**: Fine-tuning discipline + orchestration = 50%. Data audit is the most important section of the report. Honesty beats polish.

---

## Commit Discipline (Timestamps Matter)

The evaluators inspect commit timestamps to verify eval suite was committed before the first Kaggle run.

Critical sequence:
1. `chore: initial repo setup` — Day 1 morning (`.gitignore`, `requirements.txt`, `README.md`)
2. `eval: pre-training eval suite committed` — Day 1 afternoon (**before any training**)
3. `data: audit findings` — Day 2 (specific issues found)
4. `exp: first training run` — Day 3+ (after eval suite is in repo)

Never `git add .` blindly. Always stage specific files.

---

## How I Should Help

When assisting with this project:

1. **Fact-check against the official brief first.** If the brief is silent on a detail, note the ambiguity and propose a defensible choice rather than guessing.
2. **Never use threshold numbers from the guide PDFs as canonical.** Always ask the user what their researched thresholds are, or explicitly flag that a value is a placeholder pending their decision.
3. **Prioritise safe failure modes.** The orchestrator's rules are deterministic and must be implemented exactly. Any uncertainty in model behaviour should be caught and logged.
4. **Encourage pre-training commits.** The eval suite must exist in git before training begins. Remind the user of this sequencing constraint when relevant.
5. **Support honest evaluation.** If results are weak, help frame them honestly with concrete next steps rather than cosmetic fixes.
6. **For code implementation**: scaffold the structure first (eval suite, orchestrator, data audit), then move to training notebook and RAG.
7. **For the report**: ask the user to write their own analysis in Section 5. I can scaffold the structure, but the safety reasoning must be the user's own critical thinking.

---

## Key Contacts & References

- **Submission email**: surya@quantsingularity.in
- **Subject line**: `AI-SLM Screening — [Your Name]`
- **Deadline**: As stated in invitation email, IST. Late submissions not reviewed.
- **Question contact**: surya@quantsingularity.in

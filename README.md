# QS AI-SLM Signal Pod

## Overview

NIFTY 50 options signal pod with deterministic safety orchestrator. Fine-tuned small language model (TinyLlama-1.1B or Phi-2) with LoRA adapters. Built for the Quant Singularity AI Research Engineer intern screening project.

## Kaggle Notebook

https://www.kaggle.com/YOUR_USERNAME/qs-finetune-notebook <-- PUT URL HERE AFTER FIRST RUN

## Setup

```bash
# Clone (after you make the repo public or share the link)
git clone https://github.com/YOUR_USERNAME/qs-slm-project.git
cd qs-slm-project

# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Mac/Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the orchestrator on the eval set

```bash
# Without RAG
python src/run_eval.py --data data/market_states.parquet --days 31-60 --no-rag

# With RAG
python src/run_eval.py --data data/market_states.parquet --days 31-60 --rag
```

## Run evaluation

```bash
python eval/eval_suite.py --results results/orchestrator_outputs.jsonl
```

## LoRA Adapter Weights

Download from: [Google Drive / HuggingFace Hub link here]

Place in: `models/lora_adapter/`

## MLflow Runs

See `mlflow_artifacts/runs_summary.csv` for all experiment details.

To view the MLflow UI locally:

```bash
mlflow ui
# Open http://localhost:5000
```

## Project Structure

```
qs-slm-project/
├── data/                  # Provided data files (gitignored)
│   ├── market_states.parquet
│   ├── finetune_instructions.jsonl
│   ├── rag_corpus.jsonl
│   └── retrieve.py
├── eval/                  # Evaluation suite (committed before training)
│   ├── eval_suite.py
│   └── thresholds.py
├── src/                   # Core implementation
│   ├── orchestrator.py    # Safety wrapper (3 rules)
│   ├── data_audit.py      # Training data inspection
│   ├── rag.py             # RAG prompt builder + ablation
│   ├── inference.py       # CPU inference with 4-bit quant
│   └── run_eval.py        # End-to-end eval pipeline
├── notebooks/             # Kaggle training notebooks
│   └── finetune.ipynb
├── mlflow_artifacts/      # Exported run summaries
├── report/                # Final PDF report
├── results/               # Orchestrator outputs + logs
└── models/                # LoRA adapter weights (gitignored)
```

## Commit Discipline

The eval suite must be committed **before** the first Kaggle training run. Evaluators check timestamps.

## Contact

- Brief: surya@quantsingularity.in
- Submission: `AI-SLM Screening — [Your Name]`

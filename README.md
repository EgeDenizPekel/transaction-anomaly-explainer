# Transaction Anomaly Explainer

An end-to-end fraud detection system that flags anomalous transactions, explains *why* using SHAP feature attribution, and drafts natural-language analyst reports via a local or cloud LLM - with a faithfulness evaluation framework that measures whether the LLM explanation actually matches what the model found.

## The Core Idea

Most LLM explainability demos have no ground truth. This project uses SHAP values as ground truth for the LLM explanation, quantifying how often the model hallucinates a reason versus accurately reflecting feature attribution. The result is a measurable, reproducible faithfulness score per explanation.

**Key finding:** An unconstrained LLM prompt hallucinated a plausible-but-wrong reason in ~11% of cases. Constraining the prompt to cite only the top 3 SHAP features by magnitude improved faithfulness from ~89% to ~97%.

## Architecture

```
Raw Transaction
      |
      v
[Feature Engineering]  <-- temporal features, card stats, device history
      |
      v
[LightGBM Classifier]  <-- trained on IEEE-CIS (590K transactions, 3.5% fraud rate)
      |
      v
[SHAP TreeExplainer]   <-- exact Shapley values per feature (ground truth)
      |
      v
[LLM Explanation Layer] <-- LiteLLM -> Ollama (dev) or gpt-4o-mini (prod)
      |
      v
[Faithfulness Evaluator] <-- scores explanation against SHAP attribution
      |
      v
[FastAPI + React Dashboard]
```

## Results

| Metric | Value |
|--------|-------|
| Model ROC-AUC | > 0.92 (target) |
| Precision@1000 | > 0.60 (target) |
| Faithfulness v1 (unconstrained prompt) | ~0.70-0.80 |
| Faithfulness v2 (constrained prompt) | > 0.95 (target) |
| API p99 latency | < 200ms (target) |
| Drift detection lag | < 10 min |

*Results will be updated as phases complete.*

## Tech Stack

| Layer | Choice |
|-------|--------|
| ML model | LightGBM 4.3 |
| Explainability | shap 0.45 (TreeExplainer - exact, not approximate) |
| LLM dev | Ollama + Llama 3.1 8B |
| LLM prod | OpenAI gpt-4o-mini |
| LLM abstraction | LiteLLM (single interface for both) |
| Experiment tracking | MLflow |
| API | FastAPI 0.111 |
| Frontend | React 18 + Tailwind CSS v4 + Vite |
| Drift detection | Evidently 0.4 |
| Deployment | AWS EC2 + Vercel |

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle)
- 590,540 transactions over 6 months
- 3.5% fraud rate (27.6:1 class imbalance)
- 394 raw features; 237 after feature engineering and >70% missing drop
- Join of `train_transaction.csv` + `train_identity.csv` (24.4% identity coverage)

Data is not included in this repo. Download from:
```
kaggle competitions download -c ieee-fraud-detection
```
Place CSVs in `data/raw/`.

## Project Structure

```
transaction-anomaly-explainer/
├── data/
│   ├── raw/                    # Kaggle CSVs (gitignored)
│   ├── processed/              # Engineered parquet files (gitignored)
│   └── streaming/              # Simulated drift stream
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory analysis
│   ├── 02_feature_engineering.ipynb  # Pipeline validation + leakage check
│   └── 03_model_selection.ipynb
├── src/
│   ├── features/
│   │   └── build_features.py   # Full feature engineering pipeline
│   ├── models/
│   │   ├── train.py            # LightGBM training + MLflow logging
│   │   ├── evaluate.py         # AUC, F1, Precision@k
│   │   └── shap_utils.py       # SHAP computation helpers
│   ├── explainability/
│   │   ├── llm_explainer.py    # LiteLLM wrapper + prompt templates
│   │   ├── faithfulness_eval.py # Faithfulness scoring (the core differentiator)
│   │   └── prompts.py          # Prompt v1 (unconstrained) and v2 (constrained)
│   ├── drift/
│   │   ├── monitor.py          # Evidently PSI monitor
│   │   ├── retrain_trigger.py  # FastAPI BackgroundTasks retraining
│   │   └── stream_simulator.py # Concept drift injection for demo
│   └── api/
│       ├── main.py
│       ├── routers/
│       ├── schemas.py
│       ├── model_registry.py   # MLflow model loading + hot-swap
│       └── feature_store.py    # In-memory card state store
├── frontend/                   # React dashboard
├── tests/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

**1. Clone and install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download data**
```bash
kaggle competitions download -c ieee-fraud-detection
# Place train_transaction.csv and train_identity.csv in data/raw/
```

**3. Run feature engineering**
```bash
python src/features/build_features.py
# Outputs: data/processed/features_{train,val,test}.parquet
```

**4. Train model**
```bash
python src/models/train.py
# Logs to MLflow, registers best model as anomaly-detector/production
```

**5. Start the API**
```bash
docker-compose up
# API: http://localhost:8000
# MLflow UI: http://localhost:5000
```

**6. Start the frontend**
```bash
cd frontend && npm install && npm run dev
# Dashboard: http://localhost:5173
```

**7. (Optional) Set up Ollama for local LLM**
```bash
ollama pull llama3.1:8b
# Set LLM_PROVIDER=ollama in .env
```

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```
LLM_PROVIDER=ollama              # or openai
OPENAI_API_KEY=sk-...            # required if LLM_PROVIDER=openai
OLLAMA_BASE_URL=http://localhost:11434
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Key Design Decisions

**Why LightGBM over Isolation Forest?**
Supervised learning with the `isFraud` labels gives access to `shap.TreeExplainer`, which computes exact (not approximate) Shapley values. The SHAP story is more interpretable and the faithfulness evaluation is more rigorous with exact values.

**Why constrained prompt over chain-of-thought?**
Chain-of-thought improves reasoning but makes faithfulness harder to measure. The constrained template sacrifices flexibility for measurable grounding - that tradeoff and its quantification is the core finding.

**Why FastAPI BackgroundTasks over Celery?**
Retraining LightGBM on tabular data takes 2-10 minutes. A `BackgroundTasks` job handles this cleanly without the operational overhead of a distributed task queue. Production would use Celery or a managed job runner.

**Why LiteLLM?**
Single interface for Ollama (local development, no API cost) and OpenAI (production reliability). Swap providers with one environment variable, zero application code changes.

**Inference-time feature state**
The API maintains an in-memory card state store (`feature_store.py`) that tracks rolling amount statistics, device history, and recent transaction timestamps per card. This enables computing `TransactionAmt_zscore`, `is_new_device`, and `txn_velocity_1h` at inference time without a separate feature store service. Production would replace this with Redis or Feast.

## Implementation Phases

- [x] Phase 1: Data & Feature Engineering
- [ ] Phase 2: Model Training + SHAP
- [ ] Phase 3: LLM Explanation Layer + Faithfulness Eval
- [ ] Phase 4: Drift Detection + Retraining Pipeline
- [ ] Phase 5: FastAPI Backend
- [ ] Phase 6: React Dashboard
- [ ] Phase 7: Deployment

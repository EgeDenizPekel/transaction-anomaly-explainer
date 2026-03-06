# Transaction Anomaly Explainer

An end-to-end fraud detection system that flags anomalous transactions, summarizes which model attribution signals drove the risk score using SHAP, and drafts natural-language analyst summaries via a local or cloud LLM - with a faithfulness evaluation framework measuring whether LLM explanations faithfully reflect the SHAP attribution, compared against a deterministic template baseline.

## The Core Idea

Most LLM explainability demos have no reference signal. This project uses SHAP (Shapley) values as the reference attribution - not as causal ground truth (SHAP has assumptions under feature correlation), but as the most principled available attribution for tree models and the signal explicitly provided to the LLM.

The evaluation framework measures faithfulness across three explanation strategies: a deterministic template baseline (no LLM, zero hallucination by construction), an unconstrained LLM prompt (v1), and a constrained LLM prompt that must cite only the top-3 SHAP features (v2). The comparison isolates what prompt constraints actually buy in faithfulness terms.

**Key finding:** An unconstrained prompt hallucinated a feature domain not in the SHAP top-3 in 16% of cases, and got the risk direction wrong 12% of the time. Constraining the prompt reduced hallucination to 4% and improved direction accuracy from 87.9% to 99.1% - better than the template baseline on direction accuracy while matching it on hallucination control.

## Architecture

```
Raw Transaction
      |
      v
[Feature Engineering]     <-- temporal features, card stats, device history
      |
      v
[LightGBM Classifier]     <-- trained on IEEE-CIS (590K transactions, 3.5% fraud rate)
      |
      v
[SHAP TreeExplainer]      <-- exact Shapley values (reference attribution signal)
      |          |
      v          v
[Counterfactuals]    [Attribution Stability]  <-- operational + robustness signals
      |
      v
[LLM Explanation Layer]   <-- v1/v2/v3 prompts, LiteLLM -> Ollama or gpt-4o-mini
      |
      v
[Faithfulness Evaluator]  <-- template baseline vs v1 vs v2: mention, direction,
      |                       rank fidelity, hallucination, failure taxonomy
      v
[FastAPI + React Dashboard + Drift + Calibration]
```

## Results

### Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Model ROC-AUC (val / test) | 0.909 / 0.869 | Val-test gap reflects temporal drift |
| Val Brier score | 0.0361 | Naive baseline (class prior) = 0.0332 |
| Precision@1000 (test) | 0.838 | 83.8% of top-1000 scored are actual fraud |
| F1 at operating threshold (test) | 0.497 | Threshold 0.757, tuned on val set |
| Drift detection lag | 1 batch (1,000 tx) | Simulated stream - see note below |

### Faithfulness Evaluation (50 flagged test transactions)

| Metric | Template (no LLM) | v1 - unconstrained | v2 - constrained |
|--------|:-----------------:|:------------------:|:----------------:|
| mention_rate | 1.000 | 0.879 | ~1.000 |
| direction_accuracy | - | 0.879 | 0.991 |
| hallucination_rate | 0.000 | 0.160 | 0.040 |
| value_accuracy | - | 0.740 | 0.993 |

Template baseline has hallucination_rate=0 and mention_rate=1 by construction - it provides the minimum faithfulness bar the LLM must clear. v2's direction accuracy (99.1%) and hallucination rate (4%) both beat v1 and approach template faithfulness while adding natural-language fluency.

**Note on drift metrics**: The drift demo uses a synthetic concept drift stream generated from test-set data. In the post-drift segment, `txn_velocity_1h` is scaled 4x, `hour_of_day` is biased toward 0-5 AM, and fraud labels are reassigned so that high-velocity + odd-hour transactions are fraud with 80% probability (replacing the original high-amount + new-device pattern). This exaggeration is intentional: 6 batches of 1,000 transactions each produce a clear F1 degradation and recovery chart. Real concept drift is gradual and requires much larger observation windows to detect.

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
│   ├── 01_eda.ipynb                    # Exploratory analysis
│   ├── 02_feature_engineering.ipynb    # Pipeline validation + leakage check
│   ├── 03_llm_faithfulness_eval.ipynb  # LLM explanation + faithfulness experiment
│   └── 04_drift_analysis.ipynb         # Drift detection + retrain + F1 recovery chart
├── src/
│   ├── features/
│   │   └── build_features.py           # Full feature engineering pipeline
│   ├── models/
│   │   ├── train.py                    # LightGBM training + MLflow logging
│   │   ├── evaluate.py                 # AUC, F1, Precision@k
│   │   ├── shap_utils.py               # SHAP computation + attribution_stability()
│   │   └── calibration.py              # Brier score + reliability diagram data
│   ├── explainability/
│   │   ├── llm_explainer.py            # LiteLLM wrapper, v1/v2/v3 prompts
│   │   ├── faithfulness_eval.py        # mention, direction, rank_fidelity, failure taxonomy
│   │   ├── prompts.py                  # v1 unconstrained, v2 constrained, v3 structured JSON
│   │   ├── template_explainer.py       # Deterministic non-LLM baseline
│   │   ├── counterfactuals.py          # Single-feature perturbation counterfactuals
│   │   └── run_faithfulness_eval.py    # Template vs v1 vs v2 comparison runner
│   ├── drift/
│   │   ├── monitor.py          # Evidently PSI monitor
│   │   ├── retrain_trigger.py  # FastAPI BackgroundTasks retraining
│   │   └── stream_simulator.py # Concept drift injection for demo
│   └── api/
│       ├── main.py             # Lifespan: model load, calibration, drift monitor, seeder
│       ├── routers/
│       │   ├── transactions.py # POST /score (+ counterfactuals, stability), GET /transactions, POST /explain
│       │   ├── metrics.py      # GET /metrics, /drift-status, /batch-metrics, /drift-history, /calibration, /explanation-drift
│       │   └── admin.py        # POST /retrain, GET /retrain/status
│       ├── schemas.py
│       ├── model_registry.py   # MLflow model loading + hot-swap
│       ├── feature_store.py    # In-memory card state store
│       └── stream_seeder.py    # Background daemon: replays stream, computes SHAP + counterfactuals
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
# Local dev (no Docker)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# API: http://localhost:8000

# Or with Docker
docker-compose up
# API: http://localhost:8000 | MLflow UI: http://localhost:5000
```

**6. Start the frontend**
```bash
cd frontend && npm install && npm run dev
# Dashboard: http://localhost:5173
```

**7. (Optional) Set up Ollama for local LLM**
```bash
ollama pull llama3.1:8b
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
# Set LLM_PROVIDER=ollama in .env (default)
```

**8. (Optional) Run faithfulness evaluation**
```bash
# Compares template baseline vs v1 (unconstrained) vs v2 (constrained LLM)
python src/explainability/run_faithfulness_eval.py
# Results: data/processed/faithfulness_results.json
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

**SHAP as reference attribution signal, not ground truth**
SHAP values have known limitations under feature correlation and depend on the background distribution. They are used here as the reference signal - the most principled available attribution for tree models - not as a causal claim about reality. The language throughout the system says "model attribution signal" and "risk score driver", not "why the transaction is fraudulent". This distinction matters: the model provides a score; the LLM summarizes what drove that score; neither constitutes a determination of fraud.

**Why a template baseline?**
The template baseline (deterministic, no LLM, zero hallucination by construction) provides the minimum bar LLM explanations must clear on faithfulness metrics. Without it, there is no way to isolate what the LLM actually adds. A system that scores similarly to a template on faithfulness while adding hallucination risk has not earned the LLM.

**Prompt engineering: three-version progression with quantified evaluation**
The LLM explanation layer was developed in three iterations, each with a specific design hypothesis tested against the faithfulness evaluation framework:

- **v1 (unconstrained):** Provides the anomaly score and asks for a 2-3 sentence explanation. No constraints on what features the model can reference. Baseline: measures what an off-the-shelf prompt does. Result: 16% hallucination rate, 87.9% direction accuracy.
- **v2 (constrained):** Explicitly lists the top-3 SHAP features by name, value, and signed impact. Instructs the model to cite only those features and nothing else. Hypothesis: forcing grounding eliminates hallucination and direction errors. Result: hallucination dropped to 4%, direction accuracy rose to 99.1%, value accuracy rose from 74.0% to 99.3%.
- **v3 (structured JSON):** Requests a JSON object with `primary_drivers` (structured list) and `summary` (prose) in a single call. The application validates the JSON structure before accepting the prose - if parsing fails, it falls back to v2. Hypothesis: separating grounded structure from narrative generation prevents structure-level hallucination while keeping fluency. Trade-off: adds one validation layer and a fallback path in exchange for programmatic verifiability.

The constrained-to-unconstrained comparison is what makes this more than a demo: it isolates exactly what prompt constraints buy in faithfulness terms, with a template baseline (deterministic, zero hallucination by construction) as the minimum bar. Chain-of-thought was deliberately excluded because intermediate steps can introduce concepts not grounded in SHAP, making faithfulness harder to measure - that tradeoff is documented, not assumed.

**Why single-feature counterfactuals instead of DiCE?**
Single-feature perturbation answers a concrete operational question: "if only X changed, what would it need to be?" This is interpretable and fast (~30ms per flagged transaction). Joint optimization (DiCE) finds more realistic counterfactuals but requires significantly more compute and adds a dependency. The limitation (infeasibility when multiple features jointly push a score over threshold) is documented explicitly.

**Why FastAPI BackgroundTasks over Celery?**
Retraining LightGBM on tabular data takes 2-10 minutes. A `BackgroundTasks` job handles this cleanly without the operational overhead of a distributed task queue. Production would use Celery or a managed job runner.

**Why LiteLLM?**
Single interface for Ollama (local development, no API cost) and OpenAI (production reliability). Swap providers with one environment variable, zero application code changes.

**Inference-time feature state**
The API maintains an in-memory card state store (`feature_store.py`) that tracks rolling amount statistics, device history, and recent transaction timestamps per card. This enables computing `TransactionAmt_zscore`, `is_new_device`, and `txn_velocity_1h` at inference time without a separate feature store service. Production would replace this with Redis or Feast.

**Brier score and calibration**
The model's Brier score (0.0361) is slightly above the naive baseline (0.0332), which is expected for LightGBM with `class_weight='balanced'` - the class weighting inflates raw probability estimates. The model is useful (high AUC, high Precision@1000) but its probability estimates should be treated as ranking scores, not calibrated probabilities. A reliability diagram is exposed at `/calibration` for inspection.

## Implementation Phases

- [x] Phase 1: Data & Feature Engineering
- [x] Phase 2: Model Training + SHAP
- [x] Phase 3: LLM Explanation Layer + Faithfulness Eval
- [x] Phase 4: Drift Detection + Retraining Pipeline
- [x] Phase 5: FastAPI Backend
- [x] Phase 6: React Dashboard
- [x] Quality improvements: template baseline, rank fidelity, failure taxonomy, calibration, counterfactuals, attribution stability, explanation drift monitoring, structured v3 prompt, causality language

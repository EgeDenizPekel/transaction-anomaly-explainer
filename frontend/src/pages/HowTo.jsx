import { useState, useEffect, useRef, useCallback } from 'react'

const SECTIONS = [
  { id: 'dataset',       title: 'The Dataset' },
  { id: 'model',         title: 'The Model' },
  { id: 'shap',          title: 'SHAP Explanations' },
  { id: 'faithfulness',  title: 'Faithfulness Evaluation' },
  { id: 'drift',         title: 'Concept Drift & PSI' },
  { id: 'llm',           title: 'LLM Layer' },
  { id: 'dashboard',     title: 'Dashboard Guide' },
]

function Section({ id, title, children }) {
  return (
    <section id={id} className="mb-12 scroll-mt-6">
      <h2 className="text-xl font-bold text-white mb-4 pb-2 border-b border-gray-700">
        {title}
      </h2>
      <div className="space-y-4 text-gray-300 text-sm leading-relaxed">
        {children}
      </div>
    </section>
  )
}

function Term({ name, children }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-blue-400 font-semibold mb-1">{name}</div>
      <div className="text-gray-300">{children}</div>
    </div>
  )
}

function Metric({ label, value, note }) {
  return (
    <div className="bg-gray-800 rounded-lg p-3 text-center">
      <div className="text-2xl font-bold text-white">{value}</div>
      <div className="text-xs font-semibold text-gray-400 mt-0.5">{label}</div>
      {note && <div className="text-xs text-gray-600 mt-1">{note}</div>}
    </div>
  )
}

function Table({ headers, rows }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-gray-700">
      <table className="w-full text-sm">
        <thead className="bg-gray-800 text-gray-400">
          <tr>
            {headers.map(h => (
              <th key={h} className="px-4 py-2 text-left font-semibold">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-gray-900' : 'bg-gray-850'}>
              {row.map((cell, j) => (
                <td key={j} className="px-4 py-2 border-t border-gray-800 text-gray-300">{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Callout({ type = 'info', children }) {
  const styles = {
    info:    'border-blue-600 bg-blue-950/40 text-blue-200',
    warn:    'border-yellow-600 bg-yellow-950/40 text-yellow-200',
    result:  'border-green-600 bg-green-950/40 text-green-200',
  }
  const labels = { info: 'Note', warn: 'Important', result: 'Key Finding' }
  return (
    <div className={`border-l-4 rounded-r-lg px-4 py-3 text-sm ${styles[type]}`}>
      <span className="font-bold">{labels[type]}: </span>{children}
    </div>
  )
}

export default function HowTo({ isOpen, onClose }) {
  const [activeSection, setActiveSection] = useState('dataset')
  const contentRef = useRef(null)

  const handleClose = useCallback(() => onClose(), [onClose])

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') handleClose() }
    if (isOpen) document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [isOpen, handleClose])

  useEffect(() => {
    const el = contentRef.current
    if (!el) return
    const handler = () => {
      for (const s of [...SECTIONS].reverse()) {
        const target = document.getElementById(s.id)
        if (target && target.getBoundingClientRect().top <= 120) {
          setActiveSection(s.id)
          return
        }
      }
      setActiveSection(SECTIONS[0].id)
    }
    el.addEventListener('scroll', handler)
    return () => el.removeEventListener('scroll', handler)
  }, [])

  function scrollTo(id) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    setActiveSection(id)
  }

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={handleClose}
        className={`fixed inset-0 z-40 bg-black/50 backdrop-blur-sm transition-opacity duration-300 ${
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
      />

      {/* Sliding panel */}
      <div
        className={`fixed top-0 right-0 z-50 h-full w-[58%] bg-gray-950 text-gray-100 flex flex-col shadow-2xl border-l border-gray-800 transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
      {/* Top bar */}
      <header className="shrink-0 px-6 py-3 border-b border-gray-800 flex items-center gap-4">
        <h1 className="text-lg font-bold text-white flex-1">How It Works</h1>
        <span className="text-xs text-gray-500">Transaction Anomaly Explainer - Concept Guide</span>
        <button
          onClick={handleClose}
          className="ml-2 text-gray-400 hover:text-white transition-colors text-xl leading-none"
          aria-label="Close"
        >
          &#x2715;
        </button>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <nav className="w-44 shrink-0 border-r border-gray-800 overflow-y-auto py-6 px-3">
          <p className="text-xs font-semibold text-gray-600 uppercase tracking-wider mb-3">Sections</p>
          <ul className="space-y-1">
            {SECTIONS.map(s => (
              <li key={s.id}>
                <button
                  onClick={() => scrollTo(s.id)}
                  className={`w-full text-left text-sm px-3 py-1.5 rounded transition-colors ${
                    activeSection === s.id
                      ? 'bg-blue-900/50 text-blue-300 font-medium'
                      : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                  }`}
                >
                  {s.title}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        {/* Content */}
        <main ref={contentRef} className="flex-1 overflow-y-auto px-8 py-8">

          {/* ------------------------------------------------------------------ */}
          <Section id="dataset" title="The Dataset">
            <p>
              This project uses the <strong className="text-white">IEEE-CIS Fraud Detection</strong> dataset
              from Kaggle - 590,540 real financial transactions spanning 6 months, with a 3.5% fraud
              rate (27.6:1 class imbalance). Transactions are joined from two tables: transaction
              records and identity records (only 24.4% of transactions have identity data).
            </p>

            <div className="grid grid-cols-3 gap-3">
              <Metric label="Total Transactions" value="590K" />
              <Metric label="Fraud Rate" value="3.5%" note="27.6:1 imbalance" />
              <Metric label="Features (after engineering)" value="234" note="from 394 raw" />
            </div>

            <p>
              The dataset has <strong className="text-white">four groups of anonymized features</strong> - Vesta
              Corporation (the payment processor) deliberately obfuscated their proprietary fraud
              signals before publishing. This means ~90% of features have no human-readable names.
            </p>

            <Table
              headers={['Group', 'Count', 'Known Meaning', 'Example']}
              rows={[
                ['V1-V339', '180', 'Vesta proprietary signals - undisclosed', 'V45, V87, V258'],
                ['C1-C14', '14', 'Count-type features (addresses, devices per card)', 'C1, C13'],
                ['D1-D15', '8', 'Timedelta features (days since last transaction)', 'D1, D15'],
                ['M1-M9', '9', 'Match flags (name on card vs billing name)', 'M4, M6'],
                ['Named', '23', 'Transaction amount, card info, email domain, + 8 engineered', 'txn_velocity_1h'],
              ]}
            />

            <Callout type="info">
              Despite most features being opaque, the model achieves ROC-AUC 0.909. The SHAP
              explanation layer becomes more important, not less, in this context - it surfaces
              which anonymous features are driving each flag so an analyst can at least reason
              about the magnitude and direction even without a label.
            </Callout>

            <p>
              The 8 engineered features added to the dataset are computed at both training time
              (from the sorted transaction history) and inference time (from an in-memory card state store):
            </p>

            <Table
              headers={['Feature', 'What it measures']}
              rows={[
                ['TransactionAmt_log', 'Log-scaled transaction amount (reduces skew)'],
                ['TransactionAmt_zscore', 'How unusual this amount is for this card (z-score vs card history)'],
                ['card_amt_mean', 'Rolling mean of past transaction amounts for this card'],
                ['card_amt_std', 'Rolling std of past transaction amounts for this card'],
                ['amt_to_mean_ratio', 'Current amount / card mean (catches sudden large purchases)'],
                ['time_since_last_txn', 'Seconds since this card last transacted'],
                ['txn_velocity_1h', 'Number of transactions from this card in the past hour'],
                ['is_new_device', 'Whether this device has been seen before for this card'],
              ]}
            />

            <Callout type="warn">
              Leakage prevention is critical. All rolling features use
              expanding().shift(1) - the current transaction is excluded from its own statistics.
              Features are computed on the sorted-by-time dataset before the train/val/test split.
              A leakage check validates 10 random val-set rows against manual recomputation from
              training rows only.
            </Callout>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="model" title="The Model">
            <p>
              The classifier is <strong className="text-white">LightGBM</strong> - a gradient boosted
              decision tree framework optimized for speed and memory efficiency. It was chosen
              specifically because it supports <strong className="text-white">SHAP TreeExplainer</strong>,
              which computes exact (not approximate) Shapley values in polynomial time.
            </p>

            <Term name="Why not Isolation Forest or Autoencoder?">
              Unsupervised anomaly detection methods (Isolation Forest, autoencoders) don't use the
              isFraud label - they detect statistical outliers, not fraud specifically. The IEEE-CIS
              dataset has ground truth labels, so supervised learning is strictly better. More
              importantly, TreeExplainer only works with tree-based models. An autoencoder's
              "reconstruction error" explanation is far less interpretable than exact Shapley values.
            </Term>

            <Term name="Class imbalance handling">
              With a 3.5% fraud rate, a naive classifier predicting all-legitimate gets 96.5%
              accuracy. LightGBM's class_weight='balanced' parameter upweights fraud examples
              by 27.6x during training. The operating threshold (0.757) is tuned on the validation
              set to maximize F1, not accuracy.
            </Term>

            <div className="grid grid-cols-2 gap-3">
              <Metric label="Val ROC-AUC" value="0.909" note="Area under ROC curve" />
              <Metric label="Test ROC-AUC" value="0.869" note="0.04 gap = temporal drift" />
              <Metric label="Precision@1000" value="83.8%" note="Of top 1000 scored, 838 are fraud" />
              <Metric label="F1 at threshold" value="0.497" note="Threshold 0.757, tuned on val" />
            </div>

            <Callout type="warn">
              The val/test AUC gap (0.909 vs 0.869) is real temporal drift, not overfitting. The
              model was trained on earlier months and tested on later months - fraud patterns shift
              over time. This gap motivates the drift detection pipeline in Phase 4.
            </Callout>

            <p>
              The model is registered in MLflow as <code className="bg-gray-800 px-1 rounded text-blue-300">anomaly-detector v1</code> and
              loaded at API startup. Hot-swap is supported via a threading.Lock - a retrained model
              can replace the in-memory model without restarting the API.
            </p>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="shap" title="SHAP Explanations">
            <p>
              <strong className="text-white">SHAP (SHapley Additive exPlanations)</strong> is a
              game-theoretic framework for attributing a model's prediction to individual input
              features. The core idea comes from cooperative game theory: if features are "players"
              collaborating to produce a prediction, the Shapley value for each feature is its
              fair share of the credit (or blame).
            </p>

            <Term name="How Shapley values are calculated">
              For a given prediction, SHAP computes: for each feature, what is the average
              marginal contribution of that feature across all possible orderings of features?
              In practice this means comparing the model's output with and without each feature,
              across all possible subsets - exponential in theory, but LightGBM's TreeExplainer
              computes exact values in polynomial time by exploiting the tree structure.
            </Term>

            <Term name="How to read the SHAP bars in the dashboard">
              Each bar represents one of the top-3 features by absolute SHAP magnitude for that
              transaction. Bar width = relative importance (widest bar = strongest influence).
              Red = pushes the fraud score up (increases risk). Green = pushes it down (decreases risk).
              The number is the signed SHAP value in log-odds units - +1.0 means this feature
              alone shifted the fraud probability by roughly +27 percentage points from the base rate.
            </Term>

            <Callout type="info">
              The "processed value" shown below each bar is the feature's value after the
              full preprocessing pipeline. Features like card1 are frequency-encoded (their value
              is the proportion of training transactions with that card ID), so 0.034 means that
              card appeared in 3.4% of training data - not the raw card number.
            </Callout>

            <Term name="Base value vs SHAP values">
              Every SHAP explanation starts from the base value - the model's average prediction
              across the training set (approximately the log-odds of the 3.5% base fraud rate).
              Each feature's SHAP value is an additive shift from that base. The sum of all SHAP
              values + base value = the model's final log-odds output, which maps to the anomaly
              score via sigmoid.
            </Term>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="faithfulness" title="Faithfulness Evaluation">
            <p>
              This is the <strong className="text-white">core differentiator</strong> of the project.
              Most LLM explainability demos have no ground truth. Here, SHAP values are treated as
              ground truth for the LLM explanation - and we measure how often the LLM accurately
              reflects what the model actually found vs. hallucinating different features or
              getting the risk direction wrong.
            </p>

            <Term name="Why faithfulness matters">
              An LLM can generate a fluent, confident-sounding fraud explanation that has nothing
              to do with the model's actual decision. Without measurement, you can't tell.
              Faithfulness evaluation quantifies this gap using SHAP as an objective reference.
            </Term>

            <Term name="What is measured">
              Three metrics per explanation, evaluated on 50 flagged test-set transactions:
              Direction accuracy - does the LLM correctly identify whether each SHAP feature
              increases or decreases fraud risk? Hallucination rate - does the LLM mention a
              feature concept not in the SHAP top-3? Value accuracy - does the LLM correctly
              characterize the magnitude (high/low/unusual)?
            </Term>

            <Table
              headers={['Metric', 'v1 Unconstrained', 'v2 Constrained', 'Improvement']}
              rows={[
                ['Direction Accuracy', '87.9%', '99.1%', '+11.2pp'],
                ['Hallucination Rate', '16.0%', '4.0%', '-12pp'],
                ['Value Accuracy', '74.0%', '99.3%', '+25.3pp'],
              ]}
            />

            <Callout type="result">
              The unconstrained prompt hallucinated a feature concept not in the SHAP top-3 in
              1 in 6 explanations (16%), and got the risk direction wrong 12% of the time. The
              constrained v2 prompt - which explicitly instructs the LLM to cite only the
              top-3 SHAP features - reduced hallucination to 4% and direction errors to under 1%.
            </Callout>

            <Term name="v1 vs v2 prompt design">
              v1 (unconstrained) gives the LLM the anomaly score and asks for a general explanation.
              v2 (constrained) passes the top-3 SHAP features explicitly and instructs the model
              to ground every claim in those features only. The tradeoff: v2 sacrifices explanation
              flexibility for measurable factual accuracy. The Generate button in the dashboard
              uses v2.
            </Term>

            <Callout type="warn">
              The 4% residual hallucination rate in v2 comes from the LLM occasionally paraphrasing
              a feature name in a way that doesn't match any synonym set (e.g. calling card1's
              frequency-encoded value a "card type" instead of "card usage frequency"). Direction
              errors in v2 are almost entirely from the M-series match features where the LLM
              misinterprets "no match" vs "match" semantics.
            </Callout>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="drift" title="Concept Drift & PSI">
            <p>
              <strong className="text-white">Concept drift</strong> occurs when the statistical
              relationship between features and the fraud label changes over time. A model trained
              on January's fraud patterns may degrade by June if fraudsters have changed tactics.
              The val/test AUC gap (0.909 vs 0.869) in this project is a real example of this.
            </p>

            <Term name="PSI - Population Stability Index">
              PSI measures how much a feature's distribution has shifted between a reference period
              (the validation set) and the current period (recent scored transactions). It is
              computed as the sum over bins of (current% - reference%) * ln(current% / reference%).
              PSI {"<"} 0.1 = stable, 0.1-0.2 = monitor, {">"}  0.2 = significant drift.
            </Term>

            <Term name="Synthetic drift in the demo">
              The stream seeder replays a synthetic concept drift scenario across 6 batches of
              1,000 transactions each. Batches 0-2 use real test-set data (pre-drift, ~3.5% fraud).
              Batches 3-5 apply synthetic drift: transaction velocity is scaled 4x,
              hour_of_day is biased toward 0-5 AM (70% probability), and fraud labels are
              reassigned so that high-velocity + odd-hour transactions are fraud with 80% probability.
              This replaces the original high-amount + new-device fraud pattern.
            </Term>

            <div className="grid grid-cols-3 gap-3">
              <Metric label="Pre-drift F1" value="0.449" note="Batches 0-2, current model" />
              <Metric label="Post-drift F1" value="0.170" note="Batches 3-5, current model" />
              <Metric label="Post-drift F1" value="0.908" note="Batches 3-5, retrained model" />
            </div>

            <Callout type="result">
              Drift was detected on batch 3 (hour_of_day PSI = 1.51, txn_velocity_1h PSI = 0.76),
              one batch after drift injection - a 1-batch detection lag. All other features stayed
              below 0.05, confirming only the two manipulated features triggered the alert. The
              retrained model recovered F1 from 0.170 to 0.908.
            </Callout>

            <Callout type="warn">
              The drift scenario is intentionally exaggerated for demo clarity. Real concept drift
              is gradual and requires much larger observation windows. The retrained model also
              scored slightly lower on the original validation set (AUC -0.019) because it
              partially forgot the original fraud pattern - this is why the promotion threshold
              is strict (-0.005 delta). In this demo the retrained model is available in-memory
              but not auto-promoted to production.
            </Callout>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="llm" title="LLM Layer">
            <p>
              The LLM explanation layer sits after SHAP - it takes the top-3 SHAP features and
              the anomaly score and generates a natural-language analyst report. It is the last
              mile between a numeric model output and something an analyst can act on.
            </p>

            <Term name="LiteLLM abstraction">
              LiteLLM provides a single OpenAI-compatible interface for both Ollama (local) and
              OpenAI (production). Switching providers requires changing one environment variable
              (LLM_PROVIDER=ollama or openai) with zero application code changes. In local dev,
              Llama 3.1 8B runs on-device via Ollama with MLX acceleration. In production,
              gpt-4o-mini is used.
            </Term>

            <Term name="Why constrained prompt over chain-of-thought">
              Chain-of-thought improves reasoning quality but makes faithfulness harder to measure
              - intermediate steps can introduce concepts not grounded in SHAP. The constrained
              template explicitly passes the top-3 features in a structured format and instructs
              the model to cite only those features. This sacrifices output flexibility for
              measurable factual grounding. The tradeoff is quantified by the faithfulness
              evaluation.
            </Term>

            <Table
              headers={['Setting', 'Local Dev', 'Production']}
              rows={[
                ['Provider', 'Ollama', 'OpenAI'],
                ['Model', 'llama3.1:8b', 'gpt-4o-mini'],
                ['Latency', '~1-3s (MLX)', '~300ms'],
                ['Cost', 'Free (local compute)', '~$0.0002/explanation'],
                ['Start command', 'OLLAMA_FLASH_ATTENTION=1 ollama serve', 'Set OPENAI_API_KEY'],
              ]}
            />

            <Callout type="info">
              The SHAP explainer (TreeExplainer) is built lazily on the first flagged transaction
              after API startup and cached. Subsequent SHAP computations take ~40ms. The LLM call
              adds ~180ms (prod) or ~1-3s (local Llama). For the dashboard, explanations are
              generated on demand via the Generate button rather than automatically, to avoid
              LLM calls for every flagged transaction in the stream.
            </Callout>
          </Section>

          {/* ------------------------------------------------------------------ */}
          <Section id="dashboard" title="Dashboard Guide">
            <p>
              The dashboard has four panels, each polling the FastAPI backend at different intervals.
            </p>

            <Table
              headers={['Panel', 'Polls', 'Data source', 'What to look for']}
              rows={[
                ['Live Transaction Feed', 'Every 2s', 'GET /transactions', 'HIGH alerts appear in red - click to inspect. Filter by alert level. Pause to freeze the list.'],
                ['Alert Detail', 'On click', 'Transaction record + POST /explain', 'SHAP bars show which features drove the flag. Generate calls the LLM for a natural-language explanation.'],
                ['Model Metrics', 'Every 5s', 'GET /metrics + /batch-metrics', 'F1 drops sharply at batch 3 when drift hits. Fraud rate (orange dashed) spikes. The reference line marks drift injection.'],
                ['Drift Monitor', 'Every 10s', 'GET /drift-status + /drift-history', 'hour_of_day and txn_velocity_1h PSI go red (>0.2). All other features stay near zero - only the injected features drifted.'],
              ]}
            />

            <Term name="Stream seeder">
              The dashboard is populated by a background daemon thread that replays
              data/streaming/simulated_stream.parquet at 10 tx/s (configurable via SEEDER_TX_INTERVAL
              env var). It loops indefinitely. SHAP is computed for flagged transactions only
              (~3% of stream). After each batch of 1,000 transactions, F1 is computed and a
              drift check runs - results appear in Model Metrics and Drift Monitor.
            </Term>

            <Term name="Starting the full stack">
              <code className="block bg-gray-900 rounded p-2 mt-2 text-blue-300 text-xs whitespace-pre">{
`# Terminal 1 - API (from project root, venv active)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend && npm run dev

# Terminal 3 - Ollama (for LLM explanations, optional)
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`
              }</code>
            </Term>

            <Callout type="info">
              The first HIGH alert auto-selects on load. If the SHAP bars are empty on a flagged
              transaction, it was scored before the SHAP TreeExplainer was built (first flagged
              transaction after startup triggers a ~5s build). Subsequent flagged transactions
              will have SHAP bars.
            </Callout>
          </Section>

        </main>
      </div>
      </div>
    </>
  )
}

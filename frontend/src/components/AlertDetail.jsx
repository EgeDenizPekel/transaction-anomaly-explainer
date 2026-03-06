import { useState } from 'react'
import Tooltip from './Tooltip'

const DIRECTION_COLOR = {
  increases_risk: '#ef4444',
  decreases_risk: '#22c55e',
}

function formatFeatureValue(feature, value) {
  if (value === -999) return 'missing'
  switch (feature) {
    case 'time_since_last_txn':
      if (value < 0) return 'N/A'
      if (value < 60) return `${value.toFixed(0)}s`
      if (value < 3600) return `${(value / 60).toFixed(1)} min`
      if (value < 86400) return `${(value / 3600).toFixed(1)} hrs`
      return `${(value / 86400).toFixed(1)} days`
    case 'txn_velocity_1h':
      return `${value.toFixed(0)} txns/hr`
    case 'TransactionAmt':
    case 'card_amt_mean':
    case 'card_amt_std':
      return `$${value.toFixed(2)}`
    case 'TransactionAmt_zscore':
      return `${value.toFixed(2)}σ`
    case 'hour_of_day':
      return `${value.toFixed(0)}:00`
    case 'is_new_device':
      return value >= 0.5 ? 'yes' : 'no'
    default:
      return value.toFixed(4)
  }
}

const ALERT_BADGE = {
  HIGH: 'bg-red-500 text-white',
  MEDIUM: 'bg-yellow-500 text-gray-900',
  LOW: 'bg-gray-600 text-gray-200',
}

export default function AlertDetail({ transaction }) {
  const [explanation, setExplanation] = useState(null)
  const [loadingExplain, setLoadingExplain] = useState(false)
  const [explainError, setExplainError] = useState(null)

  // Reset explanation state when a different transaction is selected
  const txId = transaction?.transaction_id
  const [lastTxId, setLastTxId] = useState(null)
  if (txId !== lastTxId) {
    setLastTxId(txId)
    setExplanation(null)
    setExplainError(null)
  }

  if (!transaction) {
    return (
      <div className="bg-gray-900 rounded-lg p-4 flex items-center justify-center">
        <p className="text-gray-500 text-sm">Select a transaction to view details</p>
      </div>
    )
  }

  const topFeatures = transaction.top_features ?? []
  const maxAbs = Math.max(...topFeatures.map(f => Math.abs(f.shap_value)), 0.001)
  const displayExplanation = explanation ?? transaction.explanation
  const counterfactuals = transaction.counterfactuals ?? []
  const stabilityScore = transaction.stability_score ?? null

  async function handleExplain() {
    if (!topFeatures.length) return
    setLoadingExplain(true)
    setExplainError(null)
    try {
      const r = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transaction_id: transaction.transaction_id,
          anomaly_score: transaction.anomaly_score,
          top_features: topFeatures,
        }),
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const data = await r.json()
      setExplanation(data.explanation)
    } catch (e) {
      setExplainError('LLM unavailable. Start Ollama or set OPENAI_API_KEY.')
    } finally {
      setLoadingExplain(false)
    }
  }

  return (
    <div className="bg-gray-900 rounded-lg p-4 overflow-y-auto">
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-base font-semibold text-white">Alert Detail</h2>
        <Tooltip text="SHAP (Shapley) values show which model attribution signals drove this risk score. Bar width = relative magnitude. Red = increases risk score, green = decreases it. SHAP is the reference attribution signal (not causal ground truth). Click Generate to call the LLM for a natural-language summary grounded in these attribution values." />
      </div>

      {/* Summary row */}
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-mono text-gray-400 truncate">{transaction.transaction_id}</span>
        <span className={`text-xs font-bold px-2 py-0.5 rounded shrink-0 ml-2 ${ALERT_BADGE[transaction.alert_level] ?? 'bg-gray-600 text-gray-200'}`}>
          {transaction.alert_level}
        </span>
      </div>
      <div className="flex gap-4 text-sm text-gray-400 mb-4">
        <span>Score: <strong className="text-white">{(transaction.anomaly_score * 100).toFixed(1)}%</strong></span>
        <span>Amount: <strong className="text-white">${transaction.transaction_amt != null ? transaction.transaction_amt.toFixed(2) : '—'}</strong></span>
      </div>

      {/* SHAP top-3 horizontal bars */}
      {topFeatures.length > 0 && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-2">SHAP Attribution (Top 3)</h3>
          <div className="space-y-3">
            {topFeatures.map(f => {
              const pct = (Math.abs(f.shap_value) / maxAbs) * 100
              const color = DIRECTION_COLOR[f.direction] ?? '#6b7280'
              const label = f.direction === 'increases_risk' ? 'increases risk' : 'decreases risk'
              return (
                <div key={f.feature}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="font-mono text-gray-300">{f.feature}</span>
                    <span style={{ color }} className="font-semibold">
                      {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(4)}{' '}
                      <span className="font-normal text-gray-500">({label})</span>
                    </span>
                  </div>
                  <div className="bg-gray-800 rounded h-4 overflow-hidden">
                    <div
                      className="h-4 rounded transition-all duration-300"
                      style={{ width: `${pct}%`, backgroundColor: color }}
                    />
                  </div>
                  <div className="text-xs text-gray-600 mt-0.5">
                    value: {formatFeatureValue(f.feature, f.feature_value)}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {topFeatures.length === 0 && transaction.is_flagged && (
        <p className="text-sm text-gray-500 mb-4 italic">
          SHAP not computed - transaction scored before SHAP explainer was built (first flagged tx after startup).
        </p>
      )}

      {!transaction.is_flagged && (
        <p className="text-sm text-gray-500 mb-4 italic">
          Transaction not flagged - no SHAP attribution computed.
        </p>
      )}

      {/* Counterfactuals */}
      {counterfactuals.length > 0 && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold text-gray-300">What Would Change It?</h3>
            <Tooltip text="Single-feature counterfactuals: the minimum change to each risk-increasing feature that would drop the model score below the decision threshold (if achievable). Single-feature only - real fraud patterns may require multiple features to jointly change." />
          </div>
          <div className="space-y-1.5">
            {counterfactuals.map(cf => (
              <div key={cf.feature} className="bg-gray-800 rounded px-3 py-2 text-xs">
                <div className="flex justify-between items-center">
                  <span className="font-mono text-gray-300">{cf.feature}</span>
                  <span className="text-orange-400 font-semibold">{cf.pct_change.toFixed(1)}%</span>
                </div>
                <div className="text-gray-500 mt-0.5">
                  {formatFeatureValue(cf.feature, cf.current_value)} &rarr; {formatFeatureValue(cf.feature, cf.counterfactual_value)}
                  <span className="ml-2 text-gray-600">(score: {(cf.score_after * 100).toFixed(1)}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Attribution stability */}
      {stabilityScore !== null && (
        <div className="mb-4">
          <div className="flex items-center gap-2 mb-1">
            <h3 className="text-sm font-semibold text-gray-300">Attribution Stability</h3>
            <Tooltip text="Spearman rank correlation of SHAP feature importances under small input perturbations (5 runs, 5% noise). High (>= 0.85) means the top-3 ranking is robust. Low (<= 0.6) means the explanation is sensitive to minor input variations." />
          </div>
          <div className={`inline-flex items-center gap-2 text-xs rounded px-2 py-1 ${
            stabilityScore >= 0.85 ? 'bg-green-900/40 text-green-300'
            : stabilityScore >= 0.6 ? 'bg-yellow-900/40 text-yellow-300'
            : 'bg-red-900/40 text-red-300'
          }`}>
            <span>{stabilityScore.toFixed(3)}</span>
            <span className="text-gray-500">
              {stabilityScore >= 0.85 ? '(stable)' : stabilityScore >= 0.6 ? '(moderate)' : '(unstable)'}
            </span>
          </div>
        </div>
      )}

      {/* LLM Explanation */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-300">LLM Explanation</h3>
          {transaction.is_flagged && topFeatures.length > 0 && !displayExplanation && (
            <button
              onClick={handleExplain}
              disabled={loadingExplain}
              className="text-xs px-2 py-1 rounded font-medium bg-blue-700 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
            >
              {loadingExplain ? 'Generating...' : 'Generate'}
            </button>
          )}
        </div>

        {loadingExplain && (
          <p className="text-sm text-gray-400 italic animate-pulse">Calling LLM (v2 constrained prompt - attribution-grounded)...</p>
        )}

        {explainError && (
          <p className="text-sm text-red-400 italic">{explainError}</p>
        )}

        {displayExplanation && !loadingExplain && (
          <p className="text-sm text-gray-300 leading-relaxed bg-gray-800 rounded p-3">
            {displayExplanation}
          </p>
        )}

        {!displayExplanation && !loadingExplain && !explainError && transaction.is_flagged && topFeatures.length > 0 && (
          <p className="text-sm text-gray-600 italic">
            Click Generate to call the LLM (requires Ollama or OpenAI key).
          </p>
        )}

        {!displayExplanation && !loadingExplain && !explainError && transaction.is_flagged && topFeatures.length === 0 && (
          <p className="text-sm text-gray-600 italic">
            No SHAP features available - cannot generate explanation.
          </p>
        )}
      </div>
    </div>
  )
}

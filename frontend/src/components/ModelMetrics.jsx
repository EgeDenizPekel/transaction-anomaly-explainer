import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, Legend, ResponsiveContainer,
} from 'recharts'
import CardTooltip from './Tooltip'

function StatCard({ label, value }) {
  return (
    <div className="bg-gray-800 rounded p-3 text-center">
      <div className="text-xl font-bold text-white">{value}</div>
      <div className="text-xs text-gray-400 mt-0.5">{label}</div>
    </div>
  )
}

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null)
  const [batchMetrics, setBatchMetrics] = useState([])
  const [explanationDrift, setExplanationDrift] = useState([])
  const [retrainStatus, setRetrainStatus] = useState({ running: false })
  const [retraining, setRetraining] = useState(false)

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [mr, br, rs, ed] = await Promise.all([
          fetch('/api/metrics').then(r => r.json()),
          fetch('/api/batch-metrics').then(r => r.json()),
          fetch('/api/retrain/status').then(r => r.json()),
          fetch('/api/explanation-drift').then(r => r.json()),
        ])
        setMetrics(mr)
        setBatchMetrics(br)
        setRetrainStatus(rs)
        setExplanationDrift(ed)
        if (!rs.running) setRetraining(false)
      } catch {
        // API may not be up yet
      }
    }

    fetchAll()
    const interval = setInterval(fetchAll, 5000)
    return () => clearInterval(interval)
  }, [])

  async function handleRetrain() {
    if (retraining) return
    setRetraining(true)
    try {
      await fetch('/api/retrain', { method: 'POST' })
    } catch {
      setRetraining(false)
    }
  }

  const chartData = batchMetrics.map(b => ({
    name: `B${b.batch_id}`,
    f1: b.f1,
    fraud_rate: parseFloat((b.fraud_rate * 100).toFixed(1)),
    is_post_drift: b.is_post_drift,
  }))

  // Find first post-drift batch for reference line
  const firstPostDrift = batchMetrics.find(b => b.is_post_drift)
  const driftBatchName = firstPostDrift ? `B${firstPostDrift.batch_id}` : null

  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-col min-h-0 overflow-hidden">
      <div className="flex items-center gap-2 mb-3 shrink-0">
        <h2 className="text-base font-semibold text-white">Model Metrics</h2>
        <CardTooltip placement="top" text="Live scoring stats and per-batch F1 score from the stream seeder. The chart shows model degradation when synthetic concept drift is injected at batch 3 (transaction velocity 4x, hour_of_day biased toward 0-5 AM). Val ROC-AUC and Brier score are from the original training run on the validation set. Brier score measures calibration: lower is better, with the naive baseline (always predict class prior) at ~0.034 for 3.5% fraud rate." />
      </div>

      {metrics && (
        <div className="grid grid-cols-4 gap-2 mb-4 shrink-0">
          <StatCard label="Scored" value={metrics.n_transactions_scored.toLocaleString()} />
          <StatCard label="Flag Rate" value={`${(metrics.flag_rate * 100).toFixed(1)}%`} />
          <StatCard label="Val ROC-AUC" value={metrics.val_roc_auc?.toFixed(3) ?? '—'} />
          <StatCard label="Val Brier" value={metrics.val_brier_score?.toFixed(4) ?? '—'} />
        </div>
      )}

      <div className="shrink-0">
        <h3 className="text-sm font-semibold text-gray-300 mb-2">
          F1 per Batch (seeder stream)
        </h3>

        {chartData.length === 0 ? (
          <p className="text-sm text-gray-500">
            Waiting for first batch to complete (~8 min at 2 tx/s or set SEEDER_TX_INTERVAL=0.1)...
          </p>
        ) : (
          <>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#6b7280" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    fontSize: 12,
                    color: '#f3f4f6',
                  }}
                  formatter={(v, name) => [
                    name === 'f1' ? v.toFixed(3) : `${v}%`,
                    name === 'f1' ? 'F1' : 'Fraud Rate',
                  ]}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, color: '#9ca3af' }}
                  formatter={name => name === 'f1' ? 'F1 Score' : 'Fraud Rate %'}
                />
                {driftBatchName && (
                  <ReferenceLine
                    x={driftBatchName}
                    stroke="#f59e0b"
                    strokeDasharray="4 4"
                    label={{ value: 'drift', fill: '#f59e0b', fontSize: 10, position: 'top' }}
                  />
                )}
                <Line
                  type="monotone"
                  dataKey="f1"
                  stroke="#60a5fa"
                  strokeWidth={2}
                  dot={{ fill: '#60a5fa', r: 3 }}
                  activeDot={{ r: 5 }}
                />
                <Line
                  type="monotone"
                  dataKey="fraud_rate"
                  stroke="#f97316"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-600 mt-1">
              Batches 0-2: pre-drift (real test data). Batches 3-5: post-drift (synthetic).
            </p>
          </>
        )}
      </div>

      {/* Explanation drift: top SHAP features per batch */}
      {explanationDrift.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-700 shrink-0">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-sm font-semibold text-gray-300">Top SHAP Drivers per Batch</h3>
            <CardTooltip placement="top" text="Which features dominated the model's top-3 SHAP attributions in each batch. A shift in dominant features post-drift (e.g. txn_velocity_1h and hour_of_day displacing card/amount features) indicates the explanation layer is tracking a different causal structure - explanation drift." />
          </div>
          <div className="space-y-1">
            {explanationDrift.map(batch => {
              const topFeats = Object.entries(batch.top_feature_counts)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 3)
              if (topFeats.length === 0) return null
              return (
                <div key={batch.batch_id} className={`flex items-start gap-2 text-xs rounded px-2 py-1 ${batch.is_post_drift ? 'bg-orange-950/30 text-orange-300' : 'bg-gray-800 text-gray-400'}`}>
                  <span className="shrink-0 font-mono font-semibold">B{batch.batch_id}</span>
                  <span className="text-right leading-relaxed">
                    {topFeats.map(([feat, count]) => `${feat} (${count}x)`).join(', ')}
                  </span>
                </div>
              )
            })}
          </div>
          <p className="text-xs text-gray-600 mt-1">
            Post-drift batches (orange) should show txn_velocity_1h and hour_of_day displacing other drivers.
          </p>
        </div>
      )}

      {/* Retrain section */}
      <div className="mt-4 pt-3 border-t border-gray-700 shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-gray-300">Model Retraining</h3>
            <CardTooltip placement="top" text="Triggers a background retrain using the post-drift stream data (3K rows) + a 50K subsample of the original training set. The retrained model is promoted and hot-swapped into the API only if its val ROC-AUC does not degrade by more than 0.005 vs the current model. No API restart required." />
          </div>
          <button
            onClick={handleRetrain}
            disabled={retraining || retrainStatus.running}
            className="text-xs px-2 py-1 rounded font-medium bg-orange-700 hover:bg-orange-600 disabled:opacity-40 disabled:cursor-not-allowed text-white transition-colors"
          >
            {retraining || retrainStatus.running ? (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-orange-300 animate-pulse inline-block" />
                Retraining...
              </span>
            ) : 'Trigger Retrain'}
          </button>
        </div>

        {!retrainStatus.running && retrainStatus.auc_before != null && (
          <div className={`text-xs rounded px-3 py-2 ${retrainStatus.promoted ? 'bg-green-900/40 text-green-300' : 'bg-gray-800 text-gray-400'}`}>
            {retrainStatus.promoted
              ? `Promoted to v${retrainStatus.new_version}: AUC ${retrainStatus.auc_before?.toFixed(3)} → ${retrainStatus.auc_after?.toFixed(3)} (delta ${retrainStatus.auc_delta?.toFixed(3)})`
              : `Not promoted: AUC ${retrainStatus.auc_before?.toFixed(3)} → ${retrainStatus.auc_after?.toFixed(3)} (delta ${retrainStatus.auc_delta?.toFixed(3)}, threshold −0.005)`
            }
          </div>
        )}
        {retrainStatus.error && (
          <div className="text-xs rounded px-3 py-2 bg-red-900/40 text-red-300">
            Error: {retrainStatus.error}
          </div>
        )}
        {!retrainStatus.auc_before && !retrainStatus.running && !retrainStatus.error && (
          <p className="text-xs text-gray-600">
            Uses post-drift stream data + 50K training sample. Promotes if AUC delta ≥ −0.005.
          </p>
        )}
      </div>
    </div>
  )
}

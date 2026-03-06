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

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [mr, br] = await Promise.all([
          fetch('/api/metrics').then(r => r.json()),
          fetch('/api/batch-metrics').then(r => r.json()),
        ])
        setMetrics(mr)
        setBatchMetrics(br)
      } catch {
        // API may not be up yet
      }
    }

    fetchAll()
    const interval = setInterval(fetchAll, 5000)
    return () => clearInterval(interval)
  }, [])

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
        <CardTooltip text="Live scoring stats and per-batch F1 score from the stream seeder. The chart shows model degradation when synthetic concept drift is injected at batch 3 (transaction velocity 4×, hour_of_day biased toward 0-5 AM). Val ROC-AUC is fixed from the original training run." />
      </div>

      {metrics && (
        <div className="grid grid-cols-3 gap-2 mb-4 shrink-0">
          <StatCard label="Scored" value={metrics.n_transactions_scored.toLocaleString()} />
          <StatCard label="Flag Rate" value={`${(metrics.flag_rate * 100).toFixed(1)}%`} />
          <StatCard label="Val ROC-AUC" value={metrics.val_roc_auc?.toFixed(3) ?? '—'} />
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
    </div>
  )
}

import { useState, useEffect } from 'react'
import Tooltip from './Tooltip'

const PSI_THRESHOLD = 0.2

function PsiBar({ feature, psi, maxPsi }) {
  const drifted = psi > PSI_THRESHOLD
  const pct = Math.min((psi / Math.max(maxPsi, PSI_THRESHOLD * 1.5)) * 100, 100)
  return (
    <div>
      <div className="flex justify-between text-xs mb-0.5">
        <span className="font-mono text-gray-300">{feature}</span>
        <span className={drifted ? 'text-red-400 font-semibold' : 'text-gray-400'}>
          {psi.toFixed(4)}{drifted ? ' *' : ''}
        </span>
      </div>
      <div className="bg-gray-800 rounded h-3 overflow-hidden">
        <div
          className="h-3 rounded transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: drifted ? '#ef4444' : '#3b82f6' }}
        />
      </div>
    </div>
  )
}

export default function DriftMonitor() {
  const [driftStatus, setDriftStatus] = useState(null)
  const [driftHistory, setDriftHistory] = useState([])

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [ds, dh] = await Promise.all([
          fetch('/api/drift-status').then(r => r.json()),
          fetch('/api/drift-history').then(r => r.json()),
        ])
        setDriftStatus(ds)
        setDriftHistory(dh)
      } catch {
        // API may not be up yet
      }
    }

    fetchAll()
    const interval = setInterval(fetchAll, 10000)
    return () => clearInterval(interval)
  }, [])

  const psiEntries = driftStatus
    ? Object.entries(driftStatus.psi_scores).sort(([, a], [, b]) => b - a)
    : []
  const maxPsi = psiEntries.reduce((m, [, v]) => Math.max(m, v), PSI_THRESHOLD)

  return (
    <div className="bg-gray-900 rounded-lg p-4 overflow-y-auto">
      <div className="flex items-center gap-2 mb-3">
        <h2 className="text-base font-semibold text-white">Drift Monitor</h2>
        <Tooltip placement="top" text="PSI (Population Stability Index) measures how much the current feature distribution has shifted vs. the validation set reference. PSI > 0.2 triggers a drift alert. Only the two synthetically modified features (hour_of_day, txn_velocity_1h) should drift — all others should stay near zero." />
      </div>

      {!driftStatus ? (
        <p className="text-sm text-gray-500">Loading drift data...</p>
      ) : (
        <>
          {/* Status badge */}
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-semibold mb-4 ${driftStatus.drift_detected ? 'bg-red-900 text-red-200' : 'bg-green-900 text-green-200'}`}
          >
            <span
              className={`w-2 h-2 rounded-full ${driftStatus.drift_detected ? 'bg-red-400 animate-pulse' : 'bg-green-400'}`}
            />
            {driftStatus.drift_detected ? 'Drift Detected' : 'No Drift'}
          </div>

          {/* PSI bars */}
          {psiEntries.length > 0 ? (
            <div className="mb-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-2">
                PSI Scores{' '}
                <span className="text-gray-500 font-normal">(threshold = {PSI_THRESHOLD})</span>
              </h3>
              <div className="space-y-2">
                {psiEntries.map(([feat, psi]) => (
                  <PsiBar key={feat} feature={feat} psi={psi} maxPsi={maxPsi} />
                ))}
              </div>
              {driftStatus.last_checked && (
                <p className="text-xs text-gray-600 mt-2">
                  Last checked: {new Date(driftStatus.last_checked).toLocaleTimeString()}
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500 mb-4">
              PSI scores will appear after each seeder batch completes or after 1,000 scored
              transactions via the /score endpoint.
            </p>
          )}

          {/* Batch drift history */}
          {driftHistory.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-300 mb-2">Batch Drift History</h3>
              <div className="space-y-1">
                {driftHistory.map(event => (
                  <div
                    key={`${event.batch_id}-${event.checked_at}`}
                    className={`flex items-start justify-between text-xs rounded px-2 py-1.5 gap-2 ${event.drift_detected ? 'bg-red-900/40 text-red-300' : 'bg-gray-800 text-gray-400'}`}
                  >
                    <span className="shrink-0 font-mono">
                      B{event.batch_id}{' '}
                      <span className="text-gray-500">
                        ({event.is_post_drift ? 'post' : 'pre'}-drift)
                      </span>
                    </span>
                    <span className="text-right">
                      {event.drift_detected
                        ? `DRIFT: ${event.drifted_features.join(', ')}`
                        : 'OK'}
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-600 mt-2">
                Synthetic concept drift: batches 3-5 have 4x velocity and biased hour_of_day.
              </p>
            </div>
          )}
        </>
      )}
    </div>
  )
}

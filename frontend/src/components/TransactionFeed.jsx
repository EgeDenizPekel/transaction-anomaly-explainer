import { useState, useEffect, useRef } from 'react'
import Tooltip from './Tooltip'

const ALERT_BG = {
  HIGH: 'bg-red-950 border-red-700 hover:border-red-400',
  MEDIUM: 'bg-yellow-950 border-yellow-700 hover:border-yellow-400',
  LOW: 'bg-gray-800 border-gray-700 hover:border-gray-500',
}

const BADGE = {
  HIGH: 'bg-red-500 text-white',
  MEDIUM: 'bg-yellow-500 text-gray-900',
  LOW: 'bg-gray-600 text-gray-200',
}

const FILTER_OPTIONS = ['ALL', 'HIGH', 'MEDIUM', 'LOW']

function formatTxId(id) {
  // "stream_b1_429" -> "B1 #429"
  const m = id.match(/stream_b(\d+)_(\d+)/)
  if (m) return `B${m[1]} #${m[2]}`
  return id
}

export default function TransactionFeed({ onSelect, selectedId }) {
  const [transactions, setTransactions] = useState([])
  const [loading, setLoading] = useState(true)
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('ALL')
  const autoSelectedRef = useRef(false)

  useEffect(() => {
    if (paused) return

    const fetchData = async () => {
      try {
        const r = await fetch('/api/transactions?limit=40')
        if (r.ok) {
          const data = await r.json()
          setTransactions(data)

          // Auto-select first flagged transaction on initial load
          if (!autoSelectedRef.current && data.length > 0) {
            const firstFlagged = data.find(tx => tx.is_flagged)
            if (firstFlagged) {
              autoSelectedRef.current = true
              onSelect(firstFlagged)
            }
          }
        }
      } catch {
        // silently ignore - API may not be up yet
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 2000)
    return () => clearInterval(interval)
  }, [paused])

  const filtered = filter === 'ALL'
    ? transactions
    : transactions.filter(tx => tx.alert_level === filter)

  const counts = transactions.reduce((acc, tx) => {
    acc[tx.alert_level] = (acc[tx.alert_level] || 0) + 1
    return acc
  }, {})

  return (
    <div className="bg-gray-900 rounded-lg p-4 flex flex-col min-h-0 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-2 shrink-0">
        <div className="flex items-center gap-2">
          <h2 className="text-base font-semibold text-white">Live Transaction Feed</h2>
          <Tooltip text="Transactions scored by the model in real-time from the simulated stream. RED = HIGH risk (score ≥ 75%), YELLOW = MEDIUM (≥ 40%), GREY = LOW. Filter by alert level or pause to inspect a transaction." />
        </div>
        <div className="flex items-center gap-2">
          {paused && <span className="text-xs text-yellow-400">Paused</span>}
          <button
            onClick={() => setPaused(p => !p)}
            className={`text-xs px-2 py-1 rounded font-medium transition-colors ${paused ? 'bg-green-700 hover:bg-green-600 text-white' : 'bg-gray-700 hover:bg-gray-600 text-gray-200'}`}
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
        </div>
      </div>

      {/* Filter bar */}
      <div className="flex gap-1 mb-3 shrink-0">
        {FILTER_OPTIONS.map(opt => {
          const count = opt === 'ALL' ? transactions.length : (counts[opt] || 0)
          const active = filter === opt
          return (
            <button
              key={opt}
              onClick={() => setFilter(opt)}
              className={`text-xs px-2 py-0.5 rounded font-medium transition-colors ${
                active
                  ? opt === 'HIGH' ? 'bg-red-600 text-white'
                    : opt === 'MEDIUM' ? 'bg-yellow-600 text-gray-900'
                    : opt === 'LOW' ? 'bg-gray-500 text-white'
                    : 'bg-blue-700 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {opt} {count > 0 && <span className="opacity-75">({count})</span>}
            </button>
          )
        })}
      </div>

      {/* Transaction list */}
      <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
        {loading && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <span className="w-2 h-2 rounded-full bg-gray-500 animate-pulse inline-block" />
            Connecting to API...
          </div>
        )}
        {!loading && transactions.length === 0 && (
          <div className="flex items-center gap-2 text-gray-500 text-sm">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse inline-block" />
            Seeder starting up...
          </div>
        )}
        {!loading && transactions.length > 0 && filtered.length === 0 && (
          <p className="text-gray-500 text-sm">No {filter} transactions in current window.</p>
        )}
        {filtered.map(tx => (
          <button
            key={tx.transaction_id}
            className={`w-full text-left border rounded px-3 py-2 transition-colors ${ALERT_BG[tx.alert_level] ?? 'bg-gray-800 border-gray-700'} ${selectedId === tx.transaction_id ? 'ring-2 ring-blue-400' : ''}`}
            onClick={() => onSelect(tx)}
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-mono text-gray-300 truncate flex-1">
                {formatTxId(tx.transaction_id)}
              </span>
              <span className={`text-xs font-bold px-1.5 py-0.5 rounded shrink-0 ${BADGE[tx.alert_level] ?? 'bg-gray-600 text-gray-200'}`}>
                {tx.alert_level}
              </span>
            </div>
            <div className="flex justify-between mt-0.5 text-xs text-gray-400">
              <span>${tx.transaction_amt != null ? tx.transaction_amt.toFixed(2) : '—'}</span>
              <span>Score: {(tx.anomaly_score * 100).toFixed(1)}%</span>
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

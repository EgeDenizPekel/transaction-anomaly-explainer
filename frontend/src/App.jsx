import { useState } from 'react'
import TransactionFeed from './components/TransactionFeed'
import AlertDetail from './components/AlertDetail'
import HowTo from './pages/HowTo'
import Metrics from './pages/Metrics'

export default function App() {
  const [selectedTx, setSelectedTx] = useState(null)
  const [showMetrics, setShowMetrics] = useState(false)
  const [showHowTo, setShowHowTo] = useState(false)

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4">
      <header className="mb-4 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Transaction Anomaly Explainer
          </h1>
          <p className="text-gray-400 text-sm mt-0.5">
            Real-time fraud detection with SHAP-grounded LLM explanations
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => setShowMetrics(true)}
            className="text-sm px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors border border-gray-700"
          >
            Model Metrics
          </button>
          <button
            onClick={() => setShowHowTo(true)}
            className="text-sm px-3 py-1.5 rounded bg-gray-800 hover:bg-gray-700 text-gray-300 hover:text-white transition-colors border border-gray-700"
          >
            How It Works
          </button>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-4" style={{ height: 'calc(100vh - 96px)' }}>
        <TransactionFeed
          onSelect={setSelectedTx}
          selectedId={selectedTx?.transaction_id}
        />
        <AlertDetail transaction={selectedTx} />
      </div>

      <Metrics isOpen={showMetrics} onClose={() => setShowMetrics(false)} />
      <HowTo isOpen={showHowTo} onClose={() => setShowHowTo(false)} />
    </div>
  )
}

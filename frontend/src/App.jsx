import { useState } from 'react'
import TransactionFeed from './components/TransactionFeed'
import AlertDetail from './components/AlertDetail'
import ModelMetrics from './components/ModelMetrics'
import DriftMonitor from './components/DriftMonitor'

export default function App() {
  const [selectedTx, setSelectedTx] = useState(null)

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-4">
      <header className="mb-4">
        <h1 className="text-2xl font-bold text-white tracking-tight">
          Transaction Anomaly Explainer
        </h1>
        <p className="text-gray-400 text-sm mt-0.5">
          Real-time fraud detection with SHAP-grounded LLM explanations
        </p>
      </header>

      <div className="grid grid-cols-2 gap-4" style={{ height: 'calc(100vh - 96px)' }}>
        <TransactionFeed
          onSelect={setSelectedTx}
          selectedId={selectedTx?.transaction_id}
        />
        <AlertDetail transaction={selectedTx} />
        <ModelMetrics />
        <DriftMonitor />
      </div>
    </div>
  )
}

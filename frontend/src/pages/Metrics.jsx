import { useEffect, useCallback } from 'react'
import ModelMetrics from '../components/ModelMetrics'
import DriftMonitor from '../components/DriftMonitor'

export default function Metrics({ isOpen, onClose }) {
  const handleClose = useCallback(() => onClose(), [onClose])

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') handleClose() }
    if (isOpen) document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [isOpen, handleClose])

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
        className={`fixed top-0 right-0 z-50 h-full w-[78%] bg-gray-950 text-gray-100 flex flex-col shadow-2xl border-l border-gray-800 transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        <header className="shrink-0 px-6 py-3 border-b border-gray-800 flex items-center gap-4">
          <h1 className="text-lg font-bold text-white flex-1">Model Metrics &amp; Drift</h1>
          <span className="text-xs text-gray-500">
            F1 per batch, PSI drift, calibration, explanation drift
          </span>
          <button
            onClick={handleClose}
            className="ml-2 text-gray-400 hover:text-white transition-colors text-xl leading-none"
            aria-label="Close"
          >
            &#x2715;
          </button>
        </header>

        <div className="flex-1 overflow-hidden grid grid-cols-2 gap-4 p-4">
          <div className="overflow-y-auto">
            <ModelMetrics />
          </div>
          <div className="overflow-y-auto">
            <DriftMonitor />
          </div>
        </div>
      </div>
    </>
  )
}

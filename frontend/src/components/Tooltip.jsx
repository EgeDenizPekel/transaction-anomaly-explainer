import { useState, useRef, useEffect } from 'react'
import { createPortal } from 'react-dom'

export default function Tooltip({ text, placement = 'bottom' }) {
  const [visible, setVisible] = useState(false)
  const [coords, setCoords] = useState({ top: 0, left: 0 })
  const iconRef = useRef(null)

  function show() {
    const rect = iconRef.current.getBoundingClientRect()
    if (placement === 'top') {
      setCoords({ top: rect.top - 8, left: rect.left })
    } else {
      setCoords({ top: rect.bottom + 8, left: rect.left })
    }
    setVisible(true)
  }

  function hide() {
    setVisible(false)
  }

  const popup = visible
    ? createPortal(
        <div
          style={{
            position: 'fixed',
            top: placement === 'top' ? coords.top : coords.top,
            left: coords.left,
            transform: placement === 'top' ? 'translateY(-100%)' : 'none',
            zIndex: 9999,
            width: '18rem',
          }}
          className="rounded bg-gray-800 border border-gray-700 px-3 py-2 text-xs text-gray-300 leading-relaxed shadow-xl pointer-events-none"
        >
          {text}
        </div>,
        document.body
      )
    : null

  return (
    <>
      <span
        ref={iconRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        className="w-4 h-4 rounded-full border border-gray-600 text-gray-500 hover:text-gray-300 hover:border-gray-400 flex items-center justify-center text-[10px] font-bold cursor-default transition-colors select-none leading-none shrink-0"
      >
        i
      </span>
      {popup}
    </>
  )
}

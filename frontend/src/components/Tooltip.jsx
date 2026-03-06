export default function Tooltip({ text }) {
  return (
    <span className="relative group inline-flex items-center shrink-0">
      <span className="w-4 h-4 rounded-full border border-gray-600 text-gray-500 hover:text-gray-300 hover:border-gray-400 flex items-center justify-center text-[10px] font-bold cursor-default transition-colors select-none leading-none">
        i
      </span>
      <span className="
        pointer-events-none absolute left-0 top-full mt-2 z-50
        w-72 rounded bg-gray-800 border border-gray-700
        px-3 py-2 text-xs text-gray-300 leading-relaxed shadow-xl
        opacity-0 group-hover:opacity-100
        transition-opacity duration-150
      ">
        {text}
      </span>
    </span>
  )
}

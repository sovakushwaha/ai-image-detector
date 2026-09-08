type MarkProps = { className?: string }

export function SovaMark({ className = 'h-8 w-8' }: MarkProps) {
  return (
    <svg
      className={className}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <rect width="64" height="64" rx="14" fill="#0B0D12" />
      <path d="M18 40 L32 14 L46 40 Z" stroke="#7DD3FC" strokeWidth="2.2" />
      <circle cx="32" cy="36" r="6" stroke="#818CF8" strokeWidth="2" />
      <path d="M22 46 H42" stroke="#A5B4FC" strokeWidth="2" strokeLinecap="round" />
    </svg>
  )
}

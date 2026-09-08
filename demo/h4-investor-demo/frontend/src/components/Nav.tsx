import { motion, useReducedMotion } from 'framer-motion'
import { brand } from '../brand'
import { SovaMark } from './SovaMark'

const links = [
  { href: '#analyser', label: 'Analyse' },
  { href: '#technology', label: 'Technology' },
  { href: '#research', label: 'Research' },
  { href: '#limitations', label: 'Limitations' },
  { href: '#vision', label: 'About' },
]

export function Nav() {
  const reduce = useReducedMotion()
  return (
    <motion.header
      initial={reduce ? false : { opacity: 0, y: -12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="sticky top-0 z-50 border-b border-[var(--color-line)] bg-[rgba(7,8,11,0.78)] backdrop-blur-xl"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <a href="#top" className="flex items-center gap-3 no-underline">
          <SovaMark className="h-9 w-9" />
          <div>
            <div className="font-[family-name:var(--font-display)] text-sm font-semibold tracking-[0.14em] text-[var(--color-text)]">
              {brand.name}
            </div>
            <div className="text-[11px] text-[var(--color-muted)]">{brand.subtitle}</div>
          </div>
        </a>
        <nav aria-label="Primary" className="hidden items-center gap-6 md:flex">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-[var(--color-muted)] transition-colors hover:text-[var(--color-text)]"
            >
              {l.label}
            </a>
          ))}
        </nav>
        <a
          href="#analyser"
          className="rounded-full border border-[var(--color-line)] bg-[rgba(129,140,248,0.12)] px-3 py-1.5 text-xs font-medium text-[var(--color-cyan)] transition hover:bg-[rgba(129,140,248,0.2)]"
        >
          Analyse an image
        </a>
      </div>
    </motion.header>
  )
}

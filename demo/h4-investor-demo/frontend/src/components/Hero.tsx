import { motion, useReducedMotion } from 'framer-motion'
import { brand } from '../brand'

export function Hero() {
  const reduce = useReducedMotion()
  return (
    <section className="relative overflow-hidden px-4 pt-14 pb-4 sm:px-6 sm:pt-20">
      <div className="sova-grid pointer-events-none absolute inset-0 opacity-70" aria-hidden />
      <div
        className="pointer-events-none absolute -top-24 left-1/2 h-72 w-[36rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(129,140,248,0.22),transparent_65%)]"
        aria-hidden
      />
      <div className="relative mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.65, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-3xl"
        >
          <p className="mb-4 text-xs tracking-[0.22em] text-[var(--color-cyan)] uppercase">
            {brand.prototypeBadge} · {brand.subtitle}
          </p>
          <h1 className="font-[family-name:var(--font-display)] text-4xl leading-[1.08] font-semibold tracking-tight text-[var(--color-text)] sm:text-6xl">
            {brand.tagline}
          </h1>
          <p className="mt-5 max-w-xl text-base leading-relaxed text-[var(--color-muted)] sm:text-lg">
            {brand.heroSupport}
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="#analyser"
              className="rounded-full bg-gradient-to-r from-[rgba(129,140,248,0.95)] to-[rgba(125,211,252,0.85)] px-5 py-2.5 text-sm font-semibold text-[#07080b]"
            >
              Analyse an image
            </a>
            <a
              href="#research"
              className="rounded-full border border-[var(--color-line)] px-5 py-2.5 text-sm text-[var(--color-text)] hover:bg-white/5"
            >
              View the research
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

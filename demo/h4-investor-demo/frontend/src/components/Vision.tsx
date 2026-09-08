import { motion, useReducedMotion } from 'framer-motion'

const directions = [
  'API integration for media pipelines',
  'Content-platform screening assist',
  'Publisher / newsroom workflow',
  'Forensic triage prioritisation',
  'Provenance signal integration',
  'Continuous model update operations',
]

export function Vision() {
  const reduce = useReducedMotion()
  return (
    <section id="vision" className="scroll-mt-24 px-4 py-16 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
        >
          <p className="text-xs tracking-[0.18em] text-[var(--color-cyan)] uppercase">About</p>
          <h2 className="mt-2 max-w-3xl font-[family-name:var(--font-display)] text-3xl font-semibold">
            From research prototype to media integrity infrastructure.
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
            Today’s demo exposes a frozen scientific ensemble for local inspection. The items below
            are product direction — not shipped capabilities of this build.
          </p>
        </motion.div>
        <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {directions.map((d, i) => (
            <motion.div
              key={d}
              initial={reduce ? false : { opacity: 0, y: 10 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.04 * i }}
              className="rounded-2xl border border-[var(--color-line)] px-4 py-4"
            >
              <span className="text-[10px] tracking-[0.16em] text-[var(--color-indigo)] uppercase">
                Product direction
              </span>
              <p className="mt-2 text-sm">{d}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

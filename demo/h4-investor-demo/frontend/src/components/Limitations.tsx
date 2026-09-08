import { motion, useReducedMotion } from 'framer-motion'
import { brand } from '../brand'

const points = [
  'This is a research prototype, not an authenticity oracle.',
  'Sealed NTIRE external evaluation showed weak AI recall (~0.130) despite high Real specificity (~0.956).',
  'A low AI Detection Score does not prove an image is authentic.',
  'Generator and domain shift remain difficult; development gains do not guarantee broad transfer.',
  'Outputs should support — not replace — forensic judgement.',
]

export function Limitations() {
  const reduce = useReducedMotion()
  return (
    <section id="limitations" className="scroll-mt-24 px-4 py-16 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
          className="glass rounded-3xl p-6 sm:p-10"
        >
          <p className="text-xs tracking-[0.18em] text-[var(--color-cyan)] uppercase">
            Responsible use
          </p>
          <h2 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold">
            Limitations, stated clearly
          </h2>
          <p className="mt-3 max-w-2xl text-sm text-[var(--color-muted)]">
            Transparency is part of the product surface — not an afterthought.
          </p>
          <ul className="mt-8 space-y-3">
            {points.map((p) => (
              <li
                key={p}
                className="rounded-xl border border-[var(--color-line)] bg-[rgba(255,255,255,0.02)] px-4 py-3 text-sm leading-relaxed"
              >
                {p}
              </li>
            ))}
          </ul>
          <p className="mt-6 text-sm text-[var(--color-cyan)]">{brand.authenticityWarning}</p>
        </motion.div>
      </div>
    </section>
  )
}

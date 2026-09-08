import { motion, useReducedMotion } from 'framer-motion'

const stages = [
  { title: 'Image', body: 'Local JPEG / PNG / WEBP upload. EXIF-aware RGB preprocessing.' },
  {
    title: 'Frozen LoRA representation',
    body: 'open_clip ViT-B-16-quickgelu + fold LoRA → R1 512-d L2-normalized.',
  },
  {
    title: 'Four H4 heads',
    body: 'Fold-specific frozen Logistic Regression heads score p(AI-like signal).',
  },
  {
    title: 'Equal-weight ensemble',
    body: 'AI Detection Score = mean(p1…p4). Classification at threshold 0.5.',
  },
]

export function HowItWorks() {
  const reduce = useReducedMotion()
  return (
    <section id="technology" className="scroll-mt-24 px-4 py-16 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.3 }}
        >
          <p className="text-xs tracking-[0.18em] text-[var(--color-cyan)] uppercase">Technology</p>
          <h2 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold">
            How the frozen ensemble works
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
            Exact H4 inference stack used in sealed external evaluation. The model does not identify
            a generator and does not produce explainability heatmaps.
          </p>
        </motion.div>
        <div className="mt-10 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {stages.map((s, i) => (
            <motion.article
              key={s.title}
              initial={reduce ? false : { opacity: 0, y: 14 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ delay: 0.05 * i }}
              className="rounded-2xl border border-[var(--color-line)] bg-[rgba(16,19,26,0.7)] p-5"
            >
              <div className="mb-3 text-xs tracking-[0.16em] text-[var(--color-indigo)] uppercase">
                0{i + 1}
              </div>
              <h3 className="font-[family-name:var(--font-display)] text-lg font-medium">
                {s.title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-[var(--color-muted)]">{s.body}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  )
}

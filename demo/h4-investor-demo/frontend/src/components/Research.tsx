import { motion, useReducedMotion } from 'framer-motion'
import { researchMetrics } from '../brand'

function MetricCard({
  title,
  dataset,
  rows,
}: {
  title: string
  dataset: string
  rows: { label: string; value: string }[]
}) {
  return (
    <article className="rounded-2xl border border-[var(--color-line)] bg-[rgba(16,19,26,0.75)] p-5">
      <h3 className="font-[family-name:var(--font-display)] text-lg font-medium">{title}</h3>
      <p className="mt-1 text-xs text-[var(--color-muted)]">{dataset}</p>
      <dl className="mt-4 space-y-2">
        {rows.map((r) => (
          <div key={r.label} className="flex items-baseline justify-between gap-3 text-sm">
            <dt className="text-[var(--color-muted)]">{r.label}</dt>
            <dd className="tabular-nums font-medium">{r.value}</dd>
          </div>
        ))}
      </dl>
    </article>
  )
}

export function Research() {
  const reduce = useReducedMotion()
  const { developmentClean, strongRobust, ntire, comparabilityNote } = researchMetrics
  return (
    <section id="research" className="scroll-mt-24 px-4 py-16 sm:px-6">
      <div className="mx-auto max-w-6xl">
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, amount: 0.25 }}
        >
          <p className="text-xs tracking-[0.18em] text-[var(--color-cyan)] uppercase">Research</p>
          <h2 className="mt-2 font-[family-name:var(--font-display)] text-3xl font-semibold">
            Evidence, not marketing spin
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-relaxed text-[var(--color-muted)]">
            Authoritative frozen metrics from the completed V2 programme. Values are shown as
            reported in the scientific freeze and sealed external evaluation.
          </p>
        </motion.div>

        <div className="mt-10 grid gap-4 lg:grid-cols-3">
          <MetricCard
            title={developmentClean.label}
            dataset={developmentClean.dataset}
            rows={[
              { label: 'AUC', value: `~${developmentClean.auc.toFixed(3)}` },
              { label: 'AI recall @0.5', value: `~${developmentClean.aiRecall.toFixed(3)}` },
              { label: 'Real specificity', value: `~${developmentClean.realSpec.toFixed(3)}` },
            ]}
          />
          <MetricCard
            title={strongRobust.label}
            dataset={strongRobust.dataset}
            rows={[
              { label: 'Mean transformed AUC', value: `~${strongRobust.auc.toFixed(3)}` },
              {
                label: 'Mean transformed RealSpec',
                value: `~${strongRobust.realSpec.toFixed(3)}`,
              },
            ]}
          />
          <MetricCard
            title={ntire.label}
            dataset={ntire.dataset}
            rows={[
              { label: 'AUC', value: ntire.auc.toFixed(3) },
              { label: 'AI recall @0.5', value: ntire.aiRecall.toFixed(3) },
              { label: 'Real specificity', value: ntire.realSpec.toFixed(3) },
              { label: 'Balanced accuracy', value: ntire.balAcc.toFixed(3) },
            ]}
          />
        </div>

        <p className="mt-6 rounded-xl border border-[var(--color-line)] bg-[rgba(129,140,248,0.06)] px-4 py-3 text-sm text-[var(--color-muted)]">
          {comparabilityNote}
        </p>
      </div>
    </section>
  )
}

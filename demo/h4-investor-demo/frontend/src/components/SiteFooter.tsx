import { brand } from '../brand'
import { SovaMark } from './SovaMark'

export function SiteFooter() {
  return (
    <footer className="border-t border-[var(--color-line)] px-4 py-10 sm:px-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex items-start gap-3">
          <SovaMark className="h-8 w-8" />
          <div>
            <p className="font-[family-name:var(--font-display)] text-sm tracking-[0.14em]">
              {brand.name}
            </p>
            <p className="mt-1 text-xs text-[var(--color-muted)]">{brand.subtitle}</p>
            <p className="mt-3 max-w-md text-xs leading-relaxed text-[var(--color-muted)]">
              Local analysis. Uploaded images stay on this machine for the request only — no
              external API, no tracking, no analytics, no permanent retention.
            </p>
          </div>
        </div>
        <p className="text-xs text-[var(--color-muted)]">
          FINAL_RESEARCH_MODEL_V2 = H4 · threshold 0.5 · uncalibrated
        </p>
      </div>
    </footer>
  )
}

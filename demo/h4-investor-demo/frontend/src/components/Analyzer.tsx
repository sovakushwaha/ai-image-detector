import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { analyzeImage } from '../api'
import { brand } from '../brand'
import type { AnalyzeResult } from '../types'

const SCAN_STEPS = [
  'Preparing image',
  'Running frozen ensemble',
  'Aggregating four model signals',
  'Preparing result',
] as const

function formatBytes(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}

function ScoreMeter({ score, reduce }: { score: number; reduce: boolean | null }) {
  const pct = Math.round(score * 1000) / 10
  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between gap-3">
        <div>
          <p className="text-xs tracking-[0.18em] text-[var(--color-muted)] uppercase">
            {brand.scoreLabel}
          </p>
          <motion.p
            key={score}
            initial={reduce ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-1 font-[family-name:var(--font-display)] text-4xl font-semibold tabular-nums sm:text-5xl"
          >
            {pct.toFixed(1)}
            <span className="ml-1 text-lg text-[var(--color-muted)]">/ 100</span>
          </motion.p>
          <p className="mt-1 text-sm text-[var(--color-muted)]">
            Uncalibrated ensemble score · threshold 50
          </p>
        </div>
      </div>
      <div className="relative h-3 overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
        <motion.div
          className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-[var(--color-indigo)] via-[var(--color-violet)] to-[var(--color-cyan)]"
          initial={reduce ? false : { width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        />
        <div
          className="absolute inset-y-0 w-px bg-white/70"
          style={{ left: '50%' }}
          aria-hidden
        />
      </div>
      <div className="flex justify-between text-[11px] text-[var(--color-muted)]">
        <span>0 Real-like side</span>
        <span>0.50 threshold</span>
        <span>1.0 AI-like side</span>
      </div>
    </div>
  )
}

export function Analyzer() {
  const reduce = useReducedMotion()
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [stepIdx, setStepIdx] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      setDims(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    const img = new Image()
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight })
    img.src = url
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!busy) return
    setStepIdx(0)
    const id = window.setInterval(() => {
      setStepIdx((s) => (s + 1) % SCAN_STEPS.length)
    }, 1600)
    return () => window.clearInterval(id)
  }, [busy])

  const acceptFile = useCallback((f: File | null) => {
    setError(null)
    setResult(null)
    if (!f) {
      setFile(null)
      return
    }
    const okType =
      ['image/jpeg', 'image/png', 'image/webp'].includes(f.type) ||
      /\.(jpe?g|png|webp)$/i.test(f.name)
    if (!okType) {
      setError('Unsupported file. Use JPEG, PNG, or WEBP.')
      setFile(null)
      return
    }
    if (f.size > 20 * 1024 * 1024) {
      setError('File too large. Maximum upload is 20 MB.')
      setFile(null)
      return
    }
    setFile(f)
  }, [])

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) acceptFile(f)
  }

  const runAnalyze = async () => {
    if (!file || busy) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const res = await analyzeImage(file)
      setResult(res)
      setDetailsOpen(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  const foldEntries = useMemo(() => {
    if (!result) return []
    return [
      ['Fold 1', result.fold_scores.fold_1],
      ['Fold 2', result.fold_scores.fold_2],
      ['Fold 3', result.fold_scores.fold_3],
      ['Fold 4', result.fold_scores.fold_4],
    ] as const
  }, [result])

  return (
    <section id="analyser" className="scroll-mt-24 px-4 py-8 sm:px-6 sm:py-12">
      <div className="mx-auto max-w-6xl">
        <div className="glass overflow-hidden rounded-3xl">
          <div className="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="border-b border-[var(--color-line)] p-5 sm:p-8 lg:border-r lg:border-b-0">
              <div className="mb-5 flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-[var(--color-line)] px-2.5 py-1 text-[11px] tracking-[0.14em] text-[var(--color-cyan)] uppercase">
                  {brand.prototypeBadge}
                </span>
                <span className="rounded-full border border-[var(--color-line)] px-2.5 py-1 text-[11px] text-[var(--color-muted)]">
                  Local analysis · no external API
                </span>
              </div>

              <div
                role="button"
                tabIndex={0}
                aria-label="Upload image area. Drop a file or press Enter to choose."
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    inputRef.current?.click()
                  }
                }}
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragOver(true)
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={`relative cursor-pointer overflow-hidden rounded-2xl border border-dashed p-4 transition ${
                  dragOver
                    ? 'border-[var(--color-cyan)] bg-[rgba(125,211,252,0.06)]'
                    : 'border-[var(--color-line)] bg-[rgba(255,255,255,0.02)]'
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
                  className="sr-only"
                  onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
                />

                {preview ? (
                  <div className="relative">
                    <img
                      src={preview}
                      alt="Selected upload preview"
                      className="mx-auto max-h-80 w-full rounded-xl object-contain"
                    />
                    {busy && !reduce && (
                      <motion.div
                        className="pointer-events-none absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-[var(--color-cyan)] to-transparent"
                        animate={{ y: [0, 280, 0] }}
                        transition={{ duration: 2.4, repeat: Infinity, ease: 'linear' }}
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex min-h-56 flex-col items-center justify-center gap-3 text-center">
                    <div className="rounded-full border border-[var(--color-line)] px-3 py-1 text-xs text-[var(--color-muted)]">
                      JPEG · PNG · WEBP · max 20 MB
                    </div>
                    <p className="font-[family-name:var(--font-display)] text-lg font-medium">
                      Drop an image here
                    </p>
                    <p className="max-w-sm text-sm text-[var(--color-muted)]">
                      Or click to choose a file from your Mac. Analysis runs locally against the
                      frozen H4 ensemble.
                    </p>
                  </div>
                )}
              </div>

              {file && (
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm">
                  <div className="min-w-0">
                    <p className="truncate font-medium">{file.name}</p>
                    <p className="text-[var(--color-muted)]">
                      {formatBytes(file.size)}
                      {dims ? ` · ${dims.w}×${dims.h}` : ''}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className="rounded-full border border-[var(--color-line)] px-3 py-1.5 text-xs text-[var(--color-muted)] hover:text-[var(--color-text)]"
                      onClick={(e) => {
                        e.stopPropagation()
                        acceptFile(null)
                      }}
                    >
                      Remove
                    </button>
                    <button
                      type="button"
                      className="rounded-full border border-[var(--color-line)] px-3 py-1.5 text-xs text-[var(--color-muted)] hover:text-[var(--color-text)]"
                      onClick={(e) => {
                        e.stopPropagation()
                        inputRef.current?.click()
                      }}
                    >
                      Replace
                    </button>
                  </div>
                </div>
              )}

              <div className="mt-6 flex flex-wrap items-center gap-3">
                <motion.button
                  type="button"
                  whileTap={reduce ? undefined : { scale: 0.98 }}
                  disabled={!file || busy}
                  onClick={runAnalyze}
                  className="rounded-full bg-gradient-to-r from-[rgba(129,140,248,0.95)] to-[rgba(125,211,252,0.85)] px-5 py-2.5 text-sm font-semibold text-[#07080b] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {busy ? 'Analysing…' : 'Analyse image'}
                </motion.button>
                {busy && (
                  <p className="text-sm text-[var(--color-muted)]" aria-live="polite">
                    {SCAN_STEPS[stepIdx]}
                  </p>
                )}
              </div>

              {error && (
                <div
                  role="alert"
                  className="mt-4 rounded-xl border border-[rgba(248,113,113,0.35)] bg-[rgba(127,29,29,0.25)] px-4 py-3 text-sm text-[#fecaca]"
                >
                  {error}
                </div>
              )}
            </div>

            <div className="p-5 sm:p-8">
              <AnimatePresence mode="wait">
                {!result && !busy && (
                  <motion.div
                    key="empty"
                    initial={reduce ? false : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex h-full min-h-72 flex-col justify-center gap-4 text-[var(--color-muted)]"
                  >
                    <p className="font-[family-name:var(--font-display)] text-xl text-[var(--color-text)]">
                      Result panel
                    </p>
                    <p className="text-sm leading-relaxed">
                      After analysis you will see an AI Detection Score, a Model Classification
                      (AI-like / Real-like), and the four fold signals that form the equal-weight
                      ensemble.
                    </p>
                    <p className="text-sm leading-relaxed text-[var(--color-cyan)]/90">
                      {brand.authenticityWarning}
                    </p>
                  </motion.div>
                )}

                {busy && (
                  <motion.div
                    key="busy"
                    initial={reduce ? false : { opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex min-h-72 flex-col justify-center gap-5"
                    aria-busy="true"
                    aria-live="polite"
                  >
                    <div className="h-3 w-40 animate-pulse rounded bg-white/10" />
                    <div className="h-10 w-56 animate-pulse rounded bg-white/10" />
                    <div className="h-3 w-full animate-pulse rounded bg-white/10" />
                    <ul className="space-y-2 text-sm text-[var(--color-muted)]">
                      {SCAN_STEPS.map((s, i) => (
                        <li
                          key={s}
                          className={i === stepIdx ? 'text-[var(--color-cyan)]' : ''}
                        >
                          {i === stepIdx ? '▸ ' : '· '}
                          {s}
                        </li>
                      ))}
                    </ul>
                  </motion.div>
                )}

                {result && !busy && (
                  <motion.div
                    key="result"
                    initial={reduce ? false : { opacity: 0, y: 16 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    className="space-y-6"
                  >
                    <ScoreMeter score={result.ai_detection_score} reduce={reduce} />

                    <div className="rounded-2xl border border-[var(--color-line)] bg-[rgba(255,255,255,0.02)] p-4">
                      <p className="text-xs tracking-[0.16em] text-[var(--color-muted)] uppercase">
                        Model Classification
                      </p>
                      <p className="mt-1 font-[family-name:var(--font-display)] text-2xl font-semibold">
                        {result.classification}
                      </p>
                      <p className="mt-3 text-sm text-[var(--color-cyan)]">
                        {brand.authenticityWarning}
                      </p>
                      <p className="mt-2 text-xs text-[var(--color-muted)]">{result.score_note}</p>
                    </div>

                    <div>
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <h3 className="font-[family-name:var(--font-display)] text-sm tracking-[0.14em] uppercase">
                          Ensemble signal
                        </h3>
                        <span className="text-xs text-[var(--color-muted)]">
                          {result.ensemble_agreement}
                        </span>
                      </div>
                      <div className="space-y-3">
                        {foldEntries.map(([label, score], i) => (
                          <motion.div
                            key={label}
                            initial={reduce ? false : { opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.05 * i }}
                          >
                            <div className="mb-1 flex justify-between text-xs text-[var(--color-muted)]">
                              <span>{label}</span>
                              <span className="tabular-nums">{score.toFixed(4)}</span>
                            </div>
                            <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                              <div
                                className="h-full rounded-full bg-[var(--color-indigo)]/80"
                                style={{ width: `${score * 100}%` }}
                              />
                            </div>
                          </motion.div>
                        ))}
                      </div>
                      <p className="mt-3 text-xs text-[var(--color-muted)]">
                        Fold mean {result.fold_mean.toFixed(4)} · Fold dispersion (σ){' '}
                        {result.fold_dispersion.toFixed(4)} · {result.processing_ms} ms ·{' '}
                        {result.device}
                      </p>
                    </div>

                    <div>
                      <button
                        type="button"
                        aria-expanded={detailsOpen}
                        onClick={() => setDetailsOpen((v) => !v)}
                        className="flex w-full items-center justify-between rounded-xl border border-[var(--color-line)] px-4 py-3 text-left text-sm"
                      >
                        <span>Technical details</span>
                        <span className="text-[var(--color-muted)]">{detailsOpen ? '−' : '+'}</span>
                      </button>
                      <AnimatePresence initial={false}>
                        {detailsOpen && (
                          <motion.div
                            initial={reduce ? false : { height: 0, opacity: 0 }}
                            animate={{ height: 'auto', opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            className="overflow-hidden"
                          >
                            <dl className="mt-3 grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
                              {[
                                ['Model', result.model_version],
                                ['Representation', 'LoRA R1 512-d L2'],
                                ['Ensemble', '4 folds, equal mean'],
                                ['Threshold', String(result.threshold)],
                                ['Calibration', result.calibration],
                                ['Device', result.device],
                                ['Processing', `${result.processing_ms} ms`],
                                [
                                  'Model integrity',
                                  result.integrity_verified ? 'verified' : result.integrity_status,
                                ],
                              ].map(([k, v]) => (
                                <div
                                  key={k}
                                  className="rounded-lg border border-[var(--color-line)] px-3 py-2"
                                >
                                  <dt className="text-[11px] text-[var(--color-muted)]">{k}</dt>
                                  <dd className="mt-0.5">{v}</dd>
                                </div>
                              ))}
                            </dl>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

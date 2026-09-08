import { useEffect, useState } from 'react'
import { fetchHealth } from './api'
import { Analyzer } from './components/Analyzer'
import { Hero } from './components/Hero'
import { HowItWorks } from './components/HowItWorks'
import { Limitations } from './components/Limitations'
import { Nav } from './components/Nav'
import { Research } from './components/Research'
import { SiteFooter } from './components/SiteFooter'
import { Vision } from './components/Vision'

export default function App() {
  const [backendNote, setBackendNote] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((h) => {
        if (cancelled) return
        if (!h.integrity_verified) {
          setBackendNote(
            `Model integrity ${h.integrity_status}. Inference will be refused until freeze hashes verify.`,
          )
        } else if (!h.ready) {
          setBackendNote('Backend is reachable but not ready for inference.')
        } else {
          setBackendNote(null)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBackendNote('Backend unavailable on :8000. Start the local API with run_local.sh.')
        }
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div id="top" className="min-h-screen bg-[var(--color-ink)]">
      <Nav />
      {backendNote && (
        <div
          role="status"
          className="border-b border-[rgba(251,191,36,0.25)] bg-[rgba(120,53,15,0.35)] px-4 py-2 text-center text-sm text-[#fde68a]"
        >
          {backendNote}
        </div>
      )}
      <main>
        <Hero />
        <Analyzer />
        <HowItWorks />
        <Research />
        <Limitations />
        <Vision />
      </main>
      <SiteFooter />
    </div>
  )
}

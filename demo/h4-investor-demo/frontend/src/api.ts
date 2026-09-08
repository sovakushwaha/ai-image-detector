import type { AnalyzeResult, ApiErrorBody, HealthResponse, ModelResponse } from './types'

async function parseError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ApiErrorBody
    if (typeof body.detail === 'string') return body.detail
    if (body.detail && typeof body.detail === 'object') {
      return body.detail.message || body.detail.code || res.statusText
    }
  } catch {
    /* ignore */
  }
  if (res.status === 0) return 'Backend unavailable. Is the local API running?'
  return `Request failed (${res.status})`
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch('/api/health')
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function fetchModel(): Promise<ModelResponse> {
  const res = await fetch('/api/model')
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function analyzeImage(file: File): Promise<AnalyzeResult> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch('/api/analyze', { method: 'POST', body: form })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

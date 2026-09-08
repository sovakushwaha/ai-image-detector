export type AnalyzeResult = {
  ok: boolean
  image: {
    filename: string
    content_type: string
    size_bytes: number
    width: number
    height: number
    mode: string
  }
  ai_detection_score: number
  classification: 'AI-like' | 'Real-like'
  threshold: number
  fold_scores: {
    fold_1: number
    fold_2: number
    fold_3: number
    fold_4: number
  }
  fold_mean: number
  fold_dispersion: number
  ensemble_agreement: string
  processing_ms: number
  model_version: string
  model_name: string
  integrity_status: string
  integrity_verified: boolean
  device: string
  calibration: string
  aggregation: string
  research_prototype: boolean
  warning: string
  score_note: string
}

export type HealthResponse = {
  status: string
  ready: boolean
  integrity_verified: boolean
  integrity_status: string
  device: string
  local_only: boolean
}

export type ModelResponse = {
  model_name: string
  version: string
  freeze_status: string
  integrity_verified: boolean
  integrity_status: string
  threshold: number
  n_folds: number
  device: string
  ready: boolean
  calibration: string
}

export type ApiErrorBody = {
  detail?:
    | string
    | {
        code?: string
        message?: string
      }
}

# SOVA VERIFY — H4 Investor Demo

Local product/demo layer for **FINAL_RESEARCH_MODEL_V2 = H4**.

This is **not** a research experiment. It does not train, calibrate, retune, or modify frozen scientific artifacts.

## Architecture

```
demo/h4-investor-demo/
  backend/          FastAPI inference API
  frontend/         React + TypeScript + Vite + Tailwind + Framer Motion
  run_local.sh      One-command launcher
```

- **Backend** loads frozen LoRA + H4 fold artifacts, verifies SHA256 against `models/v2/final_v2_h4_freeze_manifest_v1.json`, and runs equal-weight ensemble inference matching the V2-11 protocol.
- **Frontend** is an investor/dissertation-facing UI (`SOVA VERIFY`) with transparent research metrics and responsible-use language.
- Scientific code under `src/`, `results/`, `models/`, and `paper/` is **read-only** for this demo.

## Requirements

- Project `.venv` with torch / open_clip / joblib / pillow / scikit-learn (existing research environment)
- Node.js 20+ (npm)
- FastAPI + uvicorn (installed into `.venv` by the launcher if missing)

## Installation

From project root:

```bash
chmod +x demo/h4-investor-demo/run_local.sh
# frontend deps install automatically on first run
```

## Run

```bash
./demo/h4-investor-demo/run_local.sh
```

Then open:

- Frontend: http://127.0.0.1:5173
- Backend health: http://127.0.0.1:8000/api/health
- OpenAPI: http://127.0.0.1:8000/docs

Ctrl+C stops both processes.

### Manual (optional)

```bash
# backend
cd demo/h4-investor-demo/backend
PYTHONPATH=.:../../../src ../../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

# frontend
cd demo/h4-investor-demo/frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

## Model provenance

| Component | Artifact |
|-----------|----------|
| Freeze manifest | `models/v2/final_v2_h4_freeze_manifest_v1.json` |
| LoRA folds 1–4 | `models/v2/clip_lora_fold{1-4}_best_v1.pt` |
| H4 heads 1–4 | `models/v2/v2_10h_h4_clean_anchored_robust_logreg_fold{1-4}_v1.joblib` |
| Representation | LoRA R1 512-d L2-normalized |
| Aggregation | `AI_SCORE = mean(p1,p2,p3,p4)` |
| Threshold | exactly `0.5` |
| Calibration | none |

If SHA256 verification fails, `/api/analyze` refuses inference.

## Privacy behaviour

- Local-only analysis
- No external model API
- No tracking / analytics
- Upload bytes are not written to disk and are discarded after the request

## Product language

- Score label: **AI Detection Score** (uncalibrated)
- Classification: **AI-like** / **Real-like**
- Warning on every result: *A low AI score does not prove that an image is authentic.*
- Badge: **Research Prototype**

## Limitations (shown in UI)

- Weak sealed NTIRE AI recall (~0.130)
- Development ≠ NTIRE (not paired)
- Not an authenticity verifier
- Output supports, does not replace, forensic judgement

## API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness + integrity summary |
| GET | `/api/model` | Safe model metadata |
| POST | `/api/analyze` | Multipart image upload → ensemble result |

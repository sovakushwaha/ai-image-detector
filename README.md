# AI Image Detector

Machine learning project on detecting AI-generated images.

The aim is a lightweight model that can tell real photographs from AI-generated images, including images from generators the model has not seen, after typical social-media transformations. The work also looks at whether the model’s confidence scores are reliable.

The first stage is a classical baseline: handcrafted image features with logistic regression. Later stages add deep learning, robustness checks, and calibration.

## Labels

- `0` = real photograph
- `1` = AI-generated image

Generator identity is stored as metadata for splits and analysis. It is not used as a feature.

## Setup

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/environment_test.py
```

## Layout

```text
data/        # local images (not in git)
metadata/    # tables for images, audits, and splits
src/         # Python scripts
results/     # experiment outputs
models/      # saved models
figures/     # plots
```

Raw images stay under `data/raw/` and are never modified in place.

## Current status

Stage 24A–24D evaluated frozen F1/F2 against F0 with paired bootstrap uncertainty. **Stage 25 (RQ5)** completed calibration audit, scalar temperature scaling, validation-derived selective prediction (Real / AI / Uncertain), frozen test evaluation, and bootstrap analysis for frozen C0 (RQ3 A2) and C1 (RQ4 F2). **Stage 27A V2 (public datasets)** is the active external-evaluation protocol; fal.ai Stage 27A v1.1 is **SUPERSEDED** (38 fal images preserved but excluded; no external detector inference on fal data). External detector inference: **NOT STARTED**.

### Stage 27A V2 (active)

```bash
source .venv/bin/activate
PYTHONPATH=src python src/acquire_external_v2_public.py
PYTHONPATH=src python src/evaluate_external_v2.py
```

Do **not** run `src/generate_external_ai_fal_v1.py` (superseded fal workflow).

To rerun the RQ5 pipeline:

```bash
source .venv/bin/activate
PYTHONPATH=src python src/run_rq5_v1.py
```

To rerun the RQ4 pipeline:

```bash
python src/rq4_frequency_transform_v1.py
python src/train_rq4_f1_v1.py
python src/train_rq4_f2_v1.py
python src/evaluate_rq4_frozen_test_v1.py
python src/rq4_bootstrap_uncertainty_v1.py
```

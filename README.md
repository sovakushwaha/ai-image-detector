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

Stage 23: RQ3 transformation-aware MobileNet regimes (A1–A3) were trained on validation-only robustness suites, then clean-validation Youden thresholds were selected. Test splits remain locked.

To regenerate RQ3 development:

```bash
python src/generate_rq3_validation_v1.py
python src/evaluate_rq3_baseline_validation_v1.py
python src/train_rq3_mobilenet_v1.py
python src/select_rq3_thresholds_v1.py
```

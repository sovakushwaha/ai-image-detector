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

Stage 22: the RQ2 robustness suite (`robustness_v1`) was generated from locked test images, and all four frozen RQ1 models were evaluated on the transformed conditions.

To regenerate and evaluate robustness:

```bash
python src/generate_robustness_v1.py
python src/evaluate_rq2_robustness_v1.py
```

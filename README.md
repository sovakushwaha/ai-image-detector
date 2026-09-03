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
app/        # local research UI (Streamlit)
data/        # local images (not in git)
metadata/    # tables for images, audits, and splits
src/         # Python scripts
results/     # experiment outputs
models/      # saved models
figures/     # plots
```

Raw images stay under `data/raw/` and are never modified in place.

## Current status

Stage 27A V2 public external validation is **COMPLETE**. Stage 28A–28C local inference engine, Streamlit UI, and usability polish are **COMPLETE**. FINAL_RESEARCH_MODEL_V1 = C0 remains unchanged.

### Local experimental image inference

Research prototype only. Stage 27A V2 independent external validation showed substantial degradation on modern generators; local predictions are **not** proof of image authenticity.

```bash
source .venv/bin/activate
PYTHONPATH=src python src/predict_image_v1.py "/path/to/image.jpg"
PYTHONPATH=src python src/predict_image_v1.py "/path/to/image.jpg" --device cpu
PYTHONPATH=src python src/predict_image_v1.py "/path/to/image.jpg" --json
```

Uses frozen FINAL_RESEARCH_MODEL_V1 (C0), frozen temperature, and frozen Real / AI-GENERATED / Uncertain selective policy. Runs fully locally (no API upload). Predictions are not saved unless `--output` is explicitly provided.

### Local graphical detector

```bash
source .venv/bin/activate
PYTHONPATH=src streamlit run app/local_detector_ui_v1.py
```

Open the local Streamlit URL, then:

1. upload one image (JPG/PNG/WEBP/BMP/TIFF);
2. click **Analyse image**;
3. view the plain-language result (Likely Real / Uncertain lean / Likely AI-generated);
4. inspect the **AI likelihood score** and scale;
5. open Technical details / How to read for scientific wording;
6. read the research warnings (external ROC-AUC ≈ 0.516; blur failure mode).

The UI reuses `FinalImageDetectorV1` from Stage 28A. No external API is used; uploads are not automatically saved.

**Example (UI wording only; underlying scientific decision may be UNCERTAIN):**

```text
Decision:
Uncertain — leaning AI-generated

AI likelihood score:
70.4%

Meaning:
The detector leans toward AI-generated but does not have enough evidence to
make a final AI-generated classification.
```

Scores below 26.4% → Likely Real; above 73.6% → Likely AI-generated; between → Uncertain (with optional lean labels). Technical details remain available in the UI expander.

### Stage 27A V2 (complete)

```bash
source .venv/bin/activate
PYTHONPATH=src python src/acquire_external_v2_public.py
PYTHONPATH=src python src/evaluate_external_v2.py
```

Do **not** run `src/generate_external_ai_fal_v1.py` or `src/acquire_external_v1.py` — superseded fal workflows, **permanently hard-disabled** via `src/fal_guard_v1.py`.

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

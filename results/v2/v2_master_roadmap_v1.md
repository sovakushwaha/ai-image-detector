# V2 Master Roadmap v1

**Locked at:** Stage V2-0 (2026-08-27)  
**V1 status:** PERMANENTLY FROZEN  

---

## Stages

### V2-0 — Freeze V1 + lock V2 protocol
**Status:** COMPLETE (this document set)

- Freeze FINAL_RESEARCH_MODEL_V1 assets and metrics
- Preserve Stage27 external failure as historical baseline
- Lock hypothesis, RQs, model ladder, compute policy, dataset-role registry (proposed)
- Create V2 namespaces without modifying V1 artifacts

### V2-1 — Select and audit V2 development datasets + reserve untouched external test
**Status:** COMPLETE (2026-08-27)

- Finalize dataset roles → `metadata/v2_dataset_role_registry_locked_v1.csv`
- Reserve NEW untouched V2 external benchmark → NTIRE val (revision locked)
- Smartphone-real acquisition plan (laion-mobile; downloads deferred)
- AI/Real inventories + holdout design + scale estimate
- Contamination guard for NTIRE until V2-11

### V2-2 — Failure analysis
**Status:** COMPLETE (2026-08-27)

- Real-camera false positives (Tiny / MLLM / COCO)
- Modern-generator misses (GPT Image 2 / Nano Banana / Qwen)
- Blur operating-point collapse + diagnostic feature associations
- Design implications → `results/v2/v2_design_implications_v1.md`

### V2-3 — Build balanced/diverse V2 development protocol
**Status:** COMPLETE (2026-08-27)

- Smartphone reconstruction 3000 (laion-mobile)
- Generator registry + 4 holdout folds + split assignments
- Duplicate audit; dataset card; CLIP dry-run plan (no download)
- `V2_NATIVE_PIXEL_PRIMARY=YES`; train cap 300/generator/fold

### V2-4 — Frozen CLIP embedding pipeline dry run
**Status:** COMPLETE (2026-08-27)

- open_clip ViT-B-16-quickgelu / openai on MPS verified
- Best batch 32; LOCAL_MPS_RECOMMENDED for full 11,377 extraction
- No classifier training; full extraction not started

### V2-5 — Frozen CLIP embeddings + Logistic Regression baseline
**Status:** NOT STARTED

- Fit lightweight linear head on development splits only
- Compare to V2 MODEL 0 (V1 C0) on development holdouts

### V2-6 — Multi-generator holdout + smartphone-real evaluation
**Status:** NOT STARTED

- Generator-family holdouts
- Smartphone-real specificity / FPR
- Decision gate inputs

### DECISION GATE
**Status:** PENDING

If frozen CLIP does not materially improve generator-holdout performance and/or independent real specificity:

- STOP expensive fine-tuning
- Return for human review

Else continue.

### V2-7 — If justified: small MLP / partial CLIP fine-tuning / LoRA
**Status:** NOT STARTED

- **Remote GPU required** (Kaggle preferred)
- Do not launch heavy training on the local laptop silently

### V2-8 — Evidence-based robustness-aware augmentation
**Status:** NOT STARTED

### V2-9 — Freeze best V2 candidate
**Status:** NOT STARTED

### V2-10 — Calibration + selective prediction
**Status:** NOT STARTED

- Validation-only temperature / abstention policies
- No final-test tuning

### V2-11 — New untouched external evaluation
**Status:** NOT STARTED

### V2-12 — V1 vs V2 statistical/resource comparison
**Status:** NOT STARTED

### V2-13 — Conference paper finalisation
**Status:** NOT STARTED

---

## Integrity snapshot (V2-0)

| Check | Status |
|-------|--------|
| V1 modified | NO |
| V1 results overwritten | NO |
| V1 external failure removed | NO |
| V2 training | NO |
| V2 model inference | NO |
| V2 dataset roles finalized | NO |
| Stage27 reused for training yet | NO |
| New V2 external benchmark selected yet | NO |
| Heavy GPU training started | NO |
| Kaggle required now | NO |

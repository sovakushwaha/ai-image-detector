# V2 Development Dataset Card (v1)

**Stage:** V2-3  
**Date:** 2026-08-27  
**Status:** Development pool + multi-generator holdout protocol LOCKED  
**Training / CLIP / NTIRE:** none performed

## Purpose

Construct a balanced V2 development dataset after V2-2 failure analysis showed:

- Real-domain FPR shift (Tiny 0.117 → MLLM 0.194 → COCO 0.280)
- Modern-generator misses (GPT Image 2 / Nano Banana 2 / Qwen)
- Low-level Real-source and legacy/modern separability

## Sources

| Source | Role | Count (usable) |
|--------|------|----------------|
| Tiny-GenImage Real | Real development | 2000 |
| MLLM Real | Real development (Stage27-reclassified) | 726 |
| COCO Real (selected 400) | Real development (Stage27-reclassified) | 400 |
| laionmobile/laion-mobile Smartphone | Real development + internal holdout | 3000 |
| Tiny-GenImage AI | Legacy AI development | 2000 |
| MLLM AI | Modern AI development | 1451 |
| Qwen-Image-Bench Stage27 subset | Modern AI development | 1800 |

**Total development pool:** 11377  
**Excluded:** fal historical (38); exact SHA256 extras (1); NTIRE final test (untouched)

## Smartphone acquisition

- Dataset: `laionmobile/laion-mobile`
- Revision: `0c60f598e67cffe8475f54edd71307648cd03465`
- Method: metadata-only URL reconstruction; seed=42 candidate pool then sequential download
- Target 3000; achieved 3000
- Splits: train 2000 / validation 500 / internal_holdout 500
- Manufacturers: {'Samsung': 834, 'Apple': 1578, 'Vivo': 72, 'Oppo': 74, 'Huawei': 360, 'Xiaomi': 82}
- Device models: 1416; largest share 3.2% (['iPhone X', 96])
- Quality gate: readable image, min(side)≥256; blurred/dark/noisy kept
- Licensing: HF metadata CC-BY-4.0; images retain original web licences — **do not redistribute reconstructed files**
- Personal user phone photos: **not used**

## Licenses / provenance

- Tiny-GenImage: existing pilot provenance
- MLLM: Apache-2.0 (HF card); revision recorded in Stage27 metadata
- Qwen-Image-Bench: Apache-2.0; prompt subset locked Stage27
- COCO val2017 selected stress set: COCO terms; local native copies
- Smartphone: see above

## Generator identity

See `metadata/v2_generator_registry_v1.csv`.

- Original generator names preserved
- Canonical IDs namespaced by source (`tiny::`, `mllm::`, `qwen::`)
- No architectural family merges from name similarity alone
- Product-string aliases noted but kept as distinct source names

## Prompt grouping

Qwen `prompt_id` is a **group ID**. Within each fold, a prompt must not appear in both TRAIN and HOLDOUT_VALIDATION. Train-generator images sharing a prompt with any holdout-generator image are marked `PROMPT_BLOCKED`.

## Duplicate handling

- Exact SHA256: extras excluded from development pool (`EXCLUDED_DUPLICATE`)
- pHash ≤ 6: candidates recorded only (`results/v2/v2_cross_dataset_duplicate_audit_v1.csv`); no automatic removal

## Fold design

Four development folds (`metadata/v2_generator_holdout_folds_v1.csv`):

- Each fold holds out ≥1 legacy and ≥2 modern generators
- GPT Image 2 held out in ≥1 fold; Nano Banana 2 held out in ≥1 fold
- Stable Real validation pool shared across folds
- Smartphone validation reported separately
- Primary selection metrics: held-out generator ROC-AUC / AP; also Real/smartphone specificity

## Training sampling

- Provisional locked cap: **300 images per generator per fold** (`metadata/v2_generator_train_cap_plan_v1.csv`)
- No physical duplication of low-count generators
- Real domains balanced via role assignment (Tiny≤1500, MLLM≤500, COCO≤280, Smartphone train=2000)

## Input pipeline note

`V2_NATIVE_PIXEL_PRIMARY = YES`

Prefer native decoded pixels + model-official preprocessing (e.g. CLIP) rather than forcing V1 `controlled_v1` JPEG q96 on all sources. File format/metadata must never be classifier features.

## Final external test

`deepfakesMSU/NTIRE-RobustAIGenDetection-val` remains **LOCKED AND UNTOUCHED**.  
Smartphone internal holdout is **not** a final external test.

## Known limitations

- Smartphone pool skews toward older phones (source bias noted by LAION-Mobile)
- Public URL reconstruction subject to link rot
- Stage27 sources are development-only after observation
- Disk pressure required deleting local COCO zip cache (`data/external_v1/_coco_cache`); COCO stress natives retained

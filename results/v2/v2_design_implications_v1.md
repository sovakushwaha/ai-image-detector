# V2 Design Implications (from V2-2 failure analysis)

**Status:** Diagnostic only — no V2 training performed.  
**V1 model:** FINAL_RESEARCH_MODEL_V1 / C0 (frozen).  
**Selective policy:** lower80=0.263967, upper80=0.736033, T=1.904173 (unchanged).

## SMARTPHONE_REAL_ANALYSIS
PENDING V2-3 / smartphone URL reconstruction.  
Exploratory personal phone false-positive remains motivation only — not formal evidence.

---

## Supported failures → V2 responses

### Failure 1 — REAL_DOMAIN_SHIFT
**Evidence:** Tiny Real mean P(AI)=0.173, FPR=0.117; MLLM Real mean P=0.235, FPR=0.194; COCO Real mean P=0.298, FPR=0.280. Real-source diagnostic LR CV accuracy=0.690.

**V2 implication:** Broader genuine-camera data with **explicit Real-domain balancing** (Tiny + MLLM + COCO + smartphone once reconstructed). Do not let Tiny Real dominate.

### Failure 2 — MODERN_GENERATOR_MISS
**Evidence:** GPT Image 2 recall=0.244, mean P=0.272; Nano Banana 2 recall=0.197, mean P=0.230; Qwen macro recall=0.321. Legacy generators remain high-scoring. Stage27 MLLM overlap explains AUC≈0.516.

**V2 implication:** Modern commercial/MLLM generator diversity + **multiple generator holdouts**. Run **frozen CLIP + linear head** before any expensive CLIP fine-tuning (decision gate after V2-5/V2-6).

### Failure 3 — BLUR_OPERATING_POINT_COLLAPSE
**Evidence:** Under MLLM `blur_sigma2`, Real specificity≈0.076, FPR≈0.924, AI recall≈0.915.

**V2 implication:** Robustness validation that tracks **Real specificity under blur**; include blurred Real examples; reject interventions that succeed by predicting AI nearly universally.

### Failure 4 — CALIBRATION_TRANSFER_FAILURE
**Evidence:** External calibrated NLL/ECE far worse than clean Tiny validation; selective abstention elevated on modern AI.

**V2 implication:** Fit temperature/selective policy only after freezing the V2 candidate; verify on untouched NTIRE at V2-11.

### Failure 5 — SOURCE_STATISTIC_DEPENDENCE (associative)
**Evidence:** Diagnostic feature shifts and high Real-source separability. Features may proxy dataset/source differences. Not proven causal.

**V2 implication:** Prefer transferable representations; preserve SHA/prompt grouping; avoid handcrafted-only final detector.

### Failure 6 — GENERATOR_HETEROGENEITY
**Evidence:** Large generator-wise dispersion of mean P(AI)/recall.

**V2 implication:** Multi-fold generator holdouts; generator-balanced sampling; report generator-wise metrics.

---

## Minimum V2-3 dataset requirements (no splits yet)

**Real:** Tiny Real; MLLM Real; COCO Real; smartphone Real (after reconstruction) with dedicated smartphone validation/holdout.

**AI:** Legacy Tiny generators; GPT Image 2; Nano Banana 2; Qwen modern generator diversity.

**Protocol must:** balance Real domains; balance generator contribution; preserve prompt/source groups; support multiple generator holdouts; never touch NTIRE until V2-11.

## Non-prescriptions
Do **not** automatically start CLIP fine-tuning/LoRA solely because it is available.

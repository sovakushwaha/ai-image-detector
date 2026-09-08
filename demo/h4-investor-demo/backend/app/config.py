from __future__ import annotations

from pathlib import Path

# demo/h4-investor-demo/backend/app/config.py -> project root is parents[4]
DEMO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models" / "v2"
MANIFEST_PATH = MODELS_DIR / "final_v2_h4_freeze_manifest_v1.json"

MODEL_NAME = "FINAL_RESEARCH_MODEL_V2"
MODEL_VERSION = "H4"
MODEL_LABEL = "H4 frozen research ensemble"
THRESHOLD = 0.5
FOLDS = (1, 2, 3, 4)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}

BRAND_NAME = "SOVA VERIFY"
BRAND_SUBTITLE = "AI Image Integrity Research"

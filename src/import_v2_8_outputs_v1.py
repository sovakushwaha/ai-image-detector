"""Download V2-8 scientific outputs only (no materialized image caches)."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = PROJECT_ROOT / "results" / "v2" / "kaggle_v2_lora_v37"
IMPORT_DIR = PROJECT_ROOT / "results" / "v2" / "kaggle_v2_lora"
KERNEL_SLUG = "sovaakushwaha/v2-clip-lora-generalisation"
FILE_PATTERN = (
    r".*(v2_lora|\.pt$|\.json$|\.csv$|\.log$|\.zip$|heartbeat|smoke|"
    r"materialization|summary|predict|metric|checkpoint|environment|"
    r"param_audit|integrity|one_batch|config_v1|__results__).*"
)


def download_scientific_outputs(max_attempts: int = 5, wait_s: int = 90) -> tuple[bool, str]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "kernels",
        "output",
        KERNEL_SLUG,
        "-p",
        str(ARCHIVE_DIR),
        "--file-pattern",
        FILE_PATTERN,
        "-q",
    ]
    last = ""
    for attempt in range(1, max_attempts + 1):
        cp = subprocess.run(cmd, capture_output=True, text=True, check=False)
        last = ((cp.stdout or "") + (cp.stderr or "")).strip()
        if cp.returncode == 0:
            return True, last
        if "429" not in last and "Too Many Requests" not in last:
            return False, last
        if attempt < max_attempts:
            time.sleep(wait_s * attempt)
    return False, last


def import_checkpoints() -> list[Path]:
    for base in (ARCHIVE_DIR / "v2_lora_outputs" / "checkpoints", ARCHIVE_DIR / "checkpoints"):
        if base.exists():
            dst = PROJECT_ROOT / "models" / "v2"
            dst.mkdir(parents=True, exist_ok=True)
            copied = []
            for p in base.glob("clip_lora_fold*_best_v1.pt"):
                out = dst / p.name
                shutil.copy2(p, out)
                copied.append(out)
            return copied
    zip_path = ARCHIVE_DIR / "v2_lora_outputs.zip"
    if zip_path.exists():
        import zipfile

        extract = ARCHIVE_DIR / "_zip_extract"
        extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract)
        return import_checkpoints()
    return []


def main() -> None:
    ok, msg = download_scientific_outputs()
    print("download_ok", ok)
    if msg:
        print(msg[:500])
    if ok:
        ckpts = import_checkpoints()
        print("checkpoints_imported", len(ckpts))
        for c in ckpts:
            print(" ", c)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
SUPERSEDED — DO NOT RUN FOR STAGE 27A
Replaced by Stage 27A V2 public-dataset protocol (src/acquire_external_v2_public.py).

Stage 27A-1 — External data acquisition + readiness gate (fal.ai v1.1; historical).

Downloads COCO val2017 if needed, locks real selection + prompt set,
prepares Midjourney collection instructions, checks AI API credentials,
and enforces the DATA READINESS GATE before any model inference.

Does NOT run model inference.
Does NOT fabricate images.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

from fal_guard_v1 import block_fal_usage, strip_fal_env

strip_fal_env()

ROOT = Path(__file__).resolve().parents[1]
# Do not load FAL_KEY from .env — fal.ai permanently disabled (see fal_guard_v1.py).
EXT = ROOT / "data" / "external_v1"
CACHE = ROOT / "data" / "external_v1" / "_coco_cache"
META = EXT / "metadata"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

PROMPT_TEMPLATE = (
    "Photorealistic natural camera photograph. {caption}. "
    "No text, no watermark, realistic lighting and camera detail."
)

DIRS = [
    EXT / "native" / "real" / "coco2017",
    EXT / "native" / "ai" / "gpt_image_2",
    EXT / "native" / "ai" / "gemini_3_1_flash_image",
    EXT / "native" / "ai" / "midjourney_v82",
    EXT / "native" / "ai" / "stable_diffusion_3_5_large",
    EXT / "controlled" / "real",
    EXT / "controlled" / "ai",
    EXT / "transformed" / "jpeg_q50",
    EXT / "transformed" / "resize_112",
    EXT / "transformed" / "blur_sigma2",
    EXT / "transformed" / "screenshot_strong",
    META,
    CACHE,
]


def ensure_dirs() -> None:
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] already present: {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    print(f"[download] {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"[ok] {dest.name} ({dest.stat().st_size:,} bytes)")


def extract_zip(zip_path: Path, out_dir: Path, members_prefix: str | None = None) -> None:
    print(f"[extract] {zip_path.name} → {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        if members_prefix is None:
            zf.extractall(out_dir)
            return
        for name in zf.namelist():
            if name.startswith(members_prefix):
                zf.extract(name, out_dir)


def credential_status() -> dict:
    keys = {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "GEMINI_API_KEY": bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
        "STABILITY_API_KEY": bool(os.environ.get("STABILITY_API_KEY") or os.environ.get("STABILITY_KEY")),
        "FAL_KEY": bool(os.environ.get("FAL_KEY")),
    }
    return keys


def acquire_coco() -> tuple[Path, Path]:
    """Download and extract COCO val2017 + captions. Returns (images_dir, captions_json)."""
    val_zip = CACHE / "val2017.zip"
    ann_zip = CACHE / "annotations_trainval2017.zip"
    download(COCO_ANN_URL, ann_zip)
    download(COCO_VAL_URL, val_zip)

    ann_dir = CACHE / "annotations"
    captions = ann_dir / "captions_val2017.json"
    if not captions.exists():
        extract_zip(ann_zip, CACHE)

    images_dir = CACHE / "val2017"
    if not images_dir.exists() or len(list(images_dir.glob("*.jpg"))) < 1000:
        extract_zip(val_zip, CACHE)

    if not captions.exists():
        raise FileNotFoundError(f"Missing captions: {captions}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images: {images_dir}")
    return images_dir, captions


def select_real_and_prompts(images_dir: Path, captions_path: Path) -> tuple[list[dict], list[dict]]:
    with captions_path.open() as f:
        caps = json.load(f)

    # Map image_id -> list of caption strings
    by_id: dict[int, list[str]] = {}
    for ann in caps["annotations"]:
        iid = int(ann["image_id"])
        text = (ann.get("caption") or "").strip()
        if text:
            by_id.setdefault(iid, []).append(text)

    # Candidate images that exist on disk and have captions
    candidates = []
    for info in caps["images"]:
        iid = int(info["id"])
        fname = info["file_name"]
        path = images_dir / fname
        if path.exists() and iid in by_id:
            candidates.append(iid)
    candidates = sorted(set(candidates))
    if len(candidates) < 400:
        raise RuntimeError(f"Only {len(candidates)} COCO candidates with captions; need ≥400")

    rng = np.random.default_rng(42)
    selected = rng.choice(candidates, size=400, replace=False)
    # Preserve selection order from rng.choice (deterministic given sorted candidates)
    selected_ids = [int(x) for x in selected]

    # Build id -> file_name map
    id_to_file = {int(info["id"]): info["file_name"] for info in caps["images"]}

    real_rows = []
    native_real = EXT / "native" / "real" / "coco2017"
    for i, iid in enumerate(selected_ids, start=1):
        src = images_dir / id_to_file[iid]
        ext_id = f"EXT_REAL_{i:04d}"
        dest_name = f"{ext_id}_{iid}.jpg"
        dest = native_real / dest_name
        if not dest.exists() or dest.stat().st_size == 0:
            shutil.copy2(src, dest)
        with Image.open(dest) as im:
            w, h = im.size
            mode = im.mode
        real_rows.append(
            {
                "external_image_id": ext_id,
                "label": 0,
                "class_name": "Real",
                "generator": "none",
                "generator_version": "n/a",
                "source_type": "photograph",
                "source_dataset": "COCO_val2017",
                "prompt_id": "",
                "prompt": "",
                "source_image_id_if_real": str(iid),
                "native_path": str(dest.relative_to(ROOT)),
                "native_extension": dest.suffix.lower(),
                "native_width": w,
                "native_height": h,
                "native_mode": mode,
                "file_size_bytes": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "acquisition_date": str(date.today()),
                "generation_seed": "",
                "seed_supported": "false",
                "provenance_verified": "true",
                "notes": "Official COCO 2017 validation photograph; ImageNet excluded by protocol",
            }
        )

    # Prompt anchors = first 100 of locked selection order
    prompt_rows = []
    for pi, iid in enumerate(selected_ids[:100], start=1):
        captions_sorted = sorted(by_id[iid])
        caption = captions_sorted[0]
        prompt_id = f"P{pi:04d}"
        gen_prompt = PROMPT_TEMPLATE.format(caption=caption)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "coco_source_image_id": iid,
                "original_coco_caption": caption,
                "generation_prompt": gen_prompt,
                "selection_seed": 42,
            }
        )

    return real_rows, prompt_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_midjourney_instructions(prompt_rows: list[dict]) -> Path:
    rows = []
    for pr in prompt_rows:
        # prompt_index 1..100 → seed 420001..420100 conceptually; MJ may not support seed
        rows.append(
            {
                "prompt_id": pr["prompt_id"],
                "exact_prompt": pr["generation_prompt"],
                "required_version": "--v 8.2",
                "required_aspect_ratio": "--ar 1:1",
                "target_filename": f"MJ82_{pr['prompt_id']}.png",
                "target_folder": "data/external_v1/native/ai/midjourney_v82/",
                "mode": "standard_normal_generation",
                "forbidden": "no editing; no reference images; no personalization; no style references; no cherry-picking",
                "notes": "Authorised Midjourney account/UI only. Do not use unofficial scrapers.",
            }
        )
    out = META / "midjourney_v82_collection_instructions_v1.csv"
    # Also keep a copy under metadata/ at repo root as specified
    alt = ROOT / "metadata" / "midjourney_v82_collection_instructions_v1.csv"
    write_csv(out, rows)
    alt.parent.mkdir(parents=True, exist_ok=True)
    write_csv(alt, rows)
    return out


def count_valid_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    n = 0
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size > 0:
            try:
                with Image.open(p) as im:
                    im.verify()
                n += 1
            except Exception:
                continue
    return n


def readiness_gate(real_count: int, creds: dict) -> dict:
    counts = {
        "real": real_count,
        "gpt_image_2": count_valid_images(EXT / "native" / "ai" / "gpt_image_2"),
        "gemini_3_1_flash_image": count_valid_images(EXT / "native" / "ai" / "gemini_3_1_flash_image"),
        "midjourney_v82": count_valid_images(EXT / "native" / "ai" / "midjourney_v82"),
        "stable_diffusion_3_5_large": count_valid_images(EXT / "native" / "ai" / "stable_diffusion_3_5_large"),
    }
    counts["total"] = sum(v for k, v in counts.items() if k != "total")
    required = {
        "real": 400,
        "gpt_image_2": 100,
        "gemini_3_1_flash_image": 100,
        "midjourney_v82": 100,
        "stable_diffusion_3_5_large": 100,
        "total": 800,
    }
    missing = []
    for k, need in required.items():
        if counts[k] < need:
            missing.append(f"{k}: {counts[k]} / {need}")

    incomplete = len(missing) > 0
    actions = []
    if not creds["OPENAI_API_KEY"]:
        actions.append("Set OPENAI_API_KEY in the environment to generate GPT-Image-2 images (100).")
    if not creds["GEMINI_API_KEY"]:
        actions.append("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to generate Gemini 3.1 Flash Image images (100).")
    if not creds["STABILITY_API_KEY"]:
        actions.append("Set STABILITY_API_KEY to generate Stable Diffusion 3.5 Large images (100).")
    if counts["midjourney_v82"] < 100:
        actions.append(
            "Manually generate Midjourney V8.2 images using "
            "metadata/midjourney_v82_collection_instructions_v1.csv and place files in "
            "data/external_v1/native/ai/midjourney_v82/ (MJ82_P0001.png … MJ82_P0100.png)."
        )
    if counts["real"] < 400:
        actions.append("Complete COCO real acquisition (expected 400).")

    report = {
        "stage": "27A",
        "gate": "DATA_READINESS",
        "date": str(date.today()),
        "counts": counts,
        "required": required,
        "complete": not incomplete,
        "missing": missing,
        "credential_present": creds,
        "required_actions": actions,
        "model_inference": "NOT STARTED" if incomplete else "AUTHORISED (gate passed)",
        "integrity": {
            "fabricated_data": False,
            "model_accessed_before_gate": False,
            "thresholds_changed": False,
            "temperature_refit": False,
        },
    }
    return report


def write_readiness_report(report: dict) -> Path:
    path = RESULTS / "external_data_readiness_report_v1.txt"
    lines = [
        "==================================================",
        "STAGE 27A — DATA READINESS REPORT",
        "==================================================",
        f"Date: {report['date']}",
        "",
        "COUNTS",
        f"Real:                      {report['counts']['real']} / 400",
        f"GPT-Image-2:               {report['counts']['gpt_image_2']} / 100",
        f"Gemini 3.1 Flash Image:    {report['counts']['gemini_3_1_flash_image']} / 100",
        f"Midjourney V8.2:           {report['counts']['midjourney_v82']} / 100",
        f"Stable Diffusion 3.5 Large:{report['counts']['stable_diffusion_3_5_large']} / 100",
        f"Total:                     {report['counts']['total']} / 800",
        "",
        f"Gate complete: {report['complete']}",
        f"Model inference: {report['model_inference']}",
        "",
        "CREDENTIALS (presence only; secrets not printed)",
        f"OPENAI_API_KEY present:     {report['credential_present']['OPENAI_API_KEY']}",
        f"GEMINI_API_KEY present:     {report['credential_present']['GEMINI_API_KEY']}",
        f"STABILITY_API_KEY present:  {report['credential_present']['STABILITY_API_KEY']}",
        f"FAL_KEY present:            {report['credential_present'].get('FAL_KEY', False)}",
        "",
        "MISSING / REQUIRED ACTIONS",
    ]
    if report["missing"]:
        for m in report["missing"]:
            lines.append(f"- incomplete: {m}")
    else:
        lines.append("- none")
    lines.append("")
    for a in report["required_actions"]:
        lines.append(f"- {a}")
    lines += [
        "",
        "SCIENTIFIC INTEGRITY",
        "Fabricated data: NO",
        "Model accessed before gate: NO",
        "Training/recalibration/threshold tuning: NO",
        "",
        "STOP RULE",
        "If gate incomplete: DO NOT run model inference.",
        "Do not substitute unknown-provenance images.",
        "Do not reopen model development.",
        "",
    ]
    path.write_text("\n".join(lines))
    (RESULTS / "external_data_readiness_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    return path


def main() -> int:
    block_fal_usage("acquire_external_v1.py")
    print("=" * 50)
    print("STAGE 27A-0/27A-1 — PROTOCOL LOCK + ACQUISITION")
    print("=" * 50)

    protocol = RESULTS / "external_protocol_v1.json"
    if not protocol.exists():
        print("ERROR: protocol not locked. Create results/external_protocol_v1.json first.")
        return 2

    ensure_dirs()
    creds = credential_status()
    print("Credentials present:", creds)

    print("\n[1] Acquiring COCO val2017 (official)...")
    try:
        images_dir, captions_path = acquire_coco()
    except Exception as exc:
        print(f"COCO acquisition failed: {exc}")
        report = readiness_gate(0, creds)
        report["required_actions"].insert(0, f"Fix COCO download/extract: {exc}")
        write_readiness_report(report)
        print_terminal_incomplete(report)
        return 1

    print("\n[2] Selecting 400 real images + locking 100 prompts (seed=42)...")
    real_rows, prompt_rows = select_real_and_prompts(images_dir, captions_path)

    # Write prompt set to both locations specified in protocol
    prompt_path = META / "external_prompt_set_v1.csv"
    prompt_path_alt = ROOT / "metadata" / "external_prompt_set_v1.csv"
    write_csv(prompt_path, prompt_rows)
    write_csv(prompt_path_alt, prompt_rows)
    print(f"Prompt set locked: {prompt_path} ({len(prompt_rows)} prompts)")

    write_csv(META / "external_manifest_real_partial_v1.csv", real_rows)
    print(f"Real native images: {len(real_rows)}")

    print("\n[3] Writing Midjourney V8.2 collection instructions...")
    mj_path = write_midjourney_instructions(prompt_rows)
    print(f"Midjourney instructions: {mj_path}")

    print("\n[4] AI generation status...")
    # Without credentials we do not attempt generation and do not fabricate.
    ai_status = {
        "gpt_image_2": "BLOCKED — OPENAI_API_KEY absent" if not creds["OPENAI_API_KEY"] else "READY_TO_GENERATE",
        "gemini_3_1_flash_image": "BLOCKED — GEMINI_API_KEY absent" if not creds["GEMINI_API_KEY"] else "READY_TO_GENERATE",
        "stable_diffusion_3_5_large": "BLOCKED — STABILITY_API_KEY absent" if not creds["STABILITY_API_KEY"] else "READY_TO_GENERATE",
        "midjourney_v82": "MANUAL_COLLECTION_REQUIRED — see midjourney_v82_collection_instructions_v1.csv",
    }
    (RESULTS / "external_ai_generation_status_v1.json").write_text(json.dumps(ai_status, indent=2) + "\n")
    for k, v in ai_status.items():
        print(f"  {k}: {v}")

    print("\n[5] DATA READINESS GATE...")
    report = readiness_gate(len(real_rows), creds)
    report_path = write_readiness_report(report)
    print(f"Readiness report: {report_path}")

    if not report["complete"]:
        print_terminal_incomplete(report)
        return 1

    print("\nDATA READINESS GATE PASSED — inference authorised separately.")
    return 0


def print_terminal_incomplete(report: dict) -> None:
    print()
    print("=" * 50)
    print("STAGE 27A — DATA READINESS INCOMPLETE")
    print("=" * 50)
    print()
    print(f"Real:                       {report['counts']['real']} / 400")
    print(f"GPT-Image-2:                {report['counts']['gpt_image_2']} / 100")
    print(f"Gemini 3.1 Flash Image:     {report['counts']['gemini_3_1_flash_image']} / 100")
    print(f"Midjourney V8.2:            {report['counts']['midjourney_v82']} / 100")
    print(f"Stable Diffusion 3.5 Large: {report['counts']['stable_diffusion_3_5_large']} / 100")
    print()
    print("MODEL INFERENCE:")
    print("NOT STARTED")
    print()
    print("Missing data/actions:")
    for a in report["required_actions"]:
        print(f"- {a}")
    print()
    print("STOP.")


if __name__ == "__main__":
    sys.exit(main())

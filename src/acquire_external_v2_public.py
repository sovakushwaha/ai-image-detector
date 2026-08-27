#!/usr/bin/env python3
"""
Stage 27A V2 — public-dataset external acquisition (no fal.ai, no API keys).

Primary:   zr-zhang/MLLM-Generated-Image-Detection-Dataset
Secondary: Qwen/Qwen-Image-Bench (matched-prompt subset)
Tertiary:  existing 400 COCO val2017 real images (reuse, no redownload)
Optional:  Scam-AI/gpt-image-2 (record GATED_NOT_AVAILABLE if inaccessible)

Does NOT load FINAL_RESEARCH_MODEL_V1 or run detector inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image
from tqdm import tqdm

from external_v2_common import META, ROOT, sha256_file

DATA = ROOT / "data" / "external_v2" / "public"
RESULTS = ROOT / "results"
MLLM_CACHE = Path.home() / ".cache/huggingface/hub/datasets--zr-zhang--MLLM-Generated-Image-Detection-Dataset"
MLLM_SHA = "1498eead24292a9a2d134476b2c559193b68b9de"

MLLM_REPO = "zr-zhang/MLLM-Generated-Image-Detection-Dataset"
QWEN_REPO = "Qwen/Qwen-Image-Bench"
SCAM_REPO = "Scam-AI/gpt-image-2"
COCO_MANIFEST = ROOT / "data" / "external_v1" / "metadata" / "external_manifest_real_partial_v1.csv"

QWEN_SEED = 42
QWEN_PROMPT_N = 200
QWEN_PROMPT_N_FALLBACK = 100
DISK_FALLBACK_GIB = 20

GENERATOR_COLS = [
    "Qwen-Image-2.0-pro",
    "gpt-image-2",
    "FLUX.2_max",
    "nano-banana-2.0",
    "nano-banana-pro",
    "Seedream-4.0",
    "Seedream-4.5",
    "Seedream-5.0",
    "GLM-Image",
    "kling_v2_1",
    "Qwen-Image-2512",
    "Qwen-Image",
    "GPT-Image-1",
    "GPT-Image-1.5",
    "HunyuanImage-3.0",
    "Imagen-4.0",
    "Imagen-4.0-Ultra",
    "FLUX.2-pro",
]

MLLM_CLASS_MAP = {
    "real": (0, "Real"),
    "GPT-Image2-fake": (1, "GPT Image 2"),
    "Nano-Banana2-fake": (1, "Nano Banana 2"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def qwen_prompt_count() -> int:
    return QWEN_PROMPT_N if free_gib(ROOT) >= DISK_FALLBACK_GIB else QWEN_PROMPT_N_FALLBACK


def verify_mllm_class(class_name: str) -> tuple[int, str]:
    if class_name not in MLLM_CLASS_MAP:
        raise SystemExit(f"STOP: unknown MLLM class folder name: {class_name}")
    return MLLM_CLASS_MAP[class_name]


def copy_mllm_from_snapshot(out_root: Path) -> int:
    snap = MLLM_CACHE / "snapshots" / MLLM_SHA
    if not snap.exists():
        return 0
    copied = 0
    for src in snap.rglob("*"):
        if not src.is_symlink() and not src.is_file():
            continue
        rel = src.relative_to(snap)
        if len(rel.parts) < 5 or rel.parts[0] != "images" or rel.parts[1] != "Raw":
            continue
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        dest = out_root / rel.parts[2] / rel.parts[3] / rel.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
            copied += 1
    return copied


def resume_mllm_snapshot(max_retries: int = 8) -> Path:
    for attempt in range(max_retries):
        try:
            cache = snapshot_download(
                repo_id=MLLM_REPO,
                repo_type="dataset",
                allow_patterns=["images/Raw/**"],
                max_workers=2,
            )
            return Path(cache)
        except HfHubHTTPError as exc:
            if "429" in str(exc) and attempt + 1 < max_retries:
                wait = 45 + attempt * 15
                print(f"HF rate limit; sleeping {wait}s before retry ({attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("MLLM snapshot download failed after retries")


def acquire_mllm() -> dict:
    license_id = "unknown"
    sha = MLLM_SHA
    source_url = f"https://huggingface.co/datasets/{MLLM_REPO}"
    try:
        info = HfApi().dataset_info(MLLM_REPO)
        sha = info.sha
        license_id = (info.cardData or {}).get("license", "unknown")
    except HfHubHTTPError:
        print("HF API rate-limited; using cached revision SHA", MLLM_SHA)

    out_root = DATA / "mllm" / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    n_cached = copy_mllm_from_snapshot(out_root)
    if n_cached:
        print(f"Materialized {n_cached} MLLM images from local HF snapshot cache")

    print(f"Resuming MLLM snapshot download ({MLLM_REPO})...")
    cache_path = resume_mllm_snapshot()
    copy_mllm_from_snapshot(out_root)

    rows = []
    corrupt = 0
    image_paths = sorted(
        p
        for p in out_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Building MLLM manifest from {len(image_paths)} local files...")
    for dest in tqdm(image_paths, desc="MLLM manifest"):
        rel = dest.relative_to(out_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        domain = parts[0]
        class_name = parts[1]
        fname = parts[-1]
        label, generator = verify_mllm_class(class_name)
        try:
            with Image.open(dest) as im:
                w, h = im.size
                fmt = im.format or Path(fname).suffix.lstrip(".").upper()
                mode = im.mode
        except Exception:
            corrupt += 1
            continue
        rows.append(
            {
                "image_id": f"MLLM_{domain[:3].upper()}_{class_name}_{Path(fname).stem}",
                "source_dataset": MLLM_REPO,
                "dataset_revision": sha,
                "license": license_id,
                "acquisition_date": utc_now()[:10],
                "domain": domain,
                "class_name": class_name,
                "label": label,
                "generator": generator,
                "native_path": str(dest.relative_to(ROOT)),
                "source_hf_path": str(Path("images") / "Raw" / rel),
                "width": w,
                "height": h,
                "format": fmt,
                "mode": mode,
                "sha256": sha256_file(dest),
                "provenance_verified": "true",
            }
        )

    manifest_path = META / "external_mllm_manifest_v2.csv"
    fields = list(rows[0].keys()) if rows else []
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    counts = Counter(r["class_name"] for r in rows)
    meta = {
        "repo": MLLM_REPO,
        "revision_sha": sha,
        "license": license_id,
        "source_url": source_url,
        "acquisition_date": utc_now(),
        "n_images": len(rows),
        "n_expected_hf_files": 2178,
        "corrupt_unreadable": corrupt,
        "class_counts": dict(counts),
        "label_mapping_verified": {
            "real": {"binary_label": 0, "generator": "Real"},
            "GPT-Image2-fake": {"binary_label": 1, "generator": "GPT Image 2"},
            "Nano-Banana2-fake": {"binary_label": 1, "generator": "Nano Banana 2"},
        },
        "hf_features_classlabel_names": ["GPT-Image2-fake", "Nano-Banana2-fake", "real"],
        "subset": "images/Raw only",
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    (META / "external_mllm_metadata_v2.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"MLLM: {len(rows)} images; corrupt={corrupt}; classes={dict(counts)}")
    return meta


def discover_qwen_generators(records: list[dict]) -> list[str]:
    present = []
    for gen in GENERATOR_COLS:
        if any(rec.get(gen) for rec in records):
            present.append(gen)
    return present


def acquire_qwen() -> dict:
    api = HfApi()
    info = api.dataset_info(QWEN_REPO)
    sha = info.sha
    license_id = (info.cardData or {}).get("license", "unknown")
    n_prompts = qwen_prompt_count()

    jsonl_path = Path(hf_hub_download(QWEN_REPO, "qwen_image_bench_hf_v0518.jsonl", repo_type="dataset"))
    records = [json.loads(line) for line in jsonl_path.open()]
    generators = discover_qwen_generators(records)
    print(f"Qwen: {len(generators)} generator columns with images; target {n_prompts} matched prompts")

    complete_ids = sorted(
        int(r["ID"])
        for r in records
        if all(r.get(g) for g in generators)
    )
    rng = np.random.default_rng(QWEN_SEED)
    n_select = min(n_prompts, len(complete_ids))
    chosen = sorted(rng.choice(complete_ids, size=n_select, replace=False).tolist())
    chosen_set = set(chosen)
    selected = [r for r in records if int(r["ID"]) in chosen_set]

    prompt_subset_path = META / "qwen_image_bench_prompt_subset_v2.csv"
    with prompt_subset_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt_id", "seed", "n_generators"])
        w.writeheader()
        for pid in chosen:
            w.writerow({"prompt_id": pid, "seed": QWEN_SEED, "n_generators": len(generators)})
    (META / "qwen_image_bench_prompt_subset_meta_v2.json").write_text(
        json.dumps(
            {
                "seed": QWEN_SEED,
                "n_prompts_selected": n_select,
                "n_generators": len(generators),
                "generators": generators,
                "selection_protocol": f"{n_select} matched prompts x all {len(generators)} generators",
            },
            indent=2,
        )
        + "\n"
    )

    out_root = DATA / "qwen" / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for rec in tqdm(selected, desc="Qwen prompts"):
        pid = int(rec["ID"])
        for gen in generators:
            hf_rel = rec[gen]
            dest_dir = out_root / gen
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(hf_rel).name
            if not dest.exists():
                local = hf_hub_download(QWEN_REPO, hf_rel, repo_type="dataset")
                shutil.copy2(local, dest)
            try:
                with Image.open(dest) as im:
                    w, h = im.size
                    fmt = im.format or Path(hf_rel).suffix.lstrip(".").upper()
            except Exception:
                continue
            manifest_rows.append(
                {
                    "image_id": f"QWEN_{pid:04d}_{gen.replace('.', '').replace('-', '_')}",
                    "prompt_id": pid,
                    "generator": gen,
                    "source_dataset": QWEN_REPO,
                    "dataset_revision": sha,
                    "label": 1,
                    "native_path": str(dest.relative_to(ROOT)),
                    "source_hf_path": hf_rel,
                    "width": w,
                    "height": h,
                    "format": fmt,
                    "sha256": sha256_file(dest),
                    "provenance_status": "verified",
                }
            )

    manifest_path = META / "external_qwen_image_bench_manifest_v2.csv"
    fields = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)

    meta = {
        "repo": QWEN_REPO,
        "revision_sha": sha,
        "license": license_id,
        "n_images": len(manifest_rows),
        "n_prompts": len(selected),
        "n_generators": len(generators),
        "generators": generators,
        "prompt_subset_locked_before_inference": True,
        "manifest": str(manifest_path.relative_to(ROOT)),
        "prompt_subset": str(prompt_subset_path.relative_to(ROOT)),
    }
    (META / "external_qwen_metadata_v2.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Qwen: {len(manifest_rows)} AI-only images ({len(selected)} prompts x {len(generators)} generators)")
    return meta


def link_coco() -> dict:
    if not COCO_MANIFEST.exists():
        raise SystemExit(f"STOP: missing COCO manifest {COCO_MANIFEST}")
    rows = list(csv.DictReader(COCO_MANIFEST.open()))
    out_manifest = META / "external_coco_stress_manifest_v2.csv"
    out_rows = []
    for r in rows:
        p = ROOT / r["native_path"]
        if not p.exists():
            raise SystemExit(f"STOP: missing COCO file {p}")
        out_rows.append(
            {
                "image_id": r["external_image_id"],
                "source_dataset": "COCO_val2017",
                "label": 0,
                "generator": "Real",
                "native_path": r["native_path"],
                "sha256": r["sha256"],
                "provenance_verified": r.get("provenance_verified", "true"),
                "role": "real_only_false_positive_stress",
            }
        )
    with out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    meta = {"n_images": len(out_rows), "manifest": str(out_manifest.relative_to(ROOT)), "redownloaded": False}
    (META / "external_coco_metadata_v2.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"COCO stress set: {len(out_rows)} real images (reused, not reseeded)")
    return meta


def probe_scam_ai() -> dict:
    status = {"repo": SCAM_REPO, "status": "UNKNOWN"}
    try:
        hf_hub_download(SCAM_REPO, "images/2046380070747644095_3_2046379141344489472.jpg", repo_type="dataset")
        status["status"] = "AVAILABLE"
    except Exception as exc:
        msg = str(exc)
        if "GatedRepoError" in type(exc).__name__ or "gated" in msg.lower() or "403" in msg:
            status["status"] = "GATED_NOT_AVAILABLE"
        else:
            status["status"] = "ERROR"
        status["detail"] = type(exc).__name__
    (META / "external_scam_gpt2_status_v2.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"Optional Scam-AI/gpt-image-2: {status['status']}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mllm-only", action="store_true")
    parser.add_argument("--qwen-only", action="store_true")
    parser.add_argument("--skip-mllm", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args()

    META.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    mllm = qwen = coco = scam = {}
    if not args.qwen_only and not args.skip_mllm:
        mllm = acquire_mllm()
    if not args.mllm_only and not args.skip_qwen:
        qwen = acquire_qwen()
    if not args.qwen_only:
        coco = link_coco()
        scam = probe_scam_ai()

    if not mllm and (META / "external_mllm_metadata_v2.json").exists():
        mllm = json.loads((META / "external_mllm_metadata_v2.json").read_text())
    if not qwen and (META / "external_qwen_metadata_v2.json").exists():
        qwen = json.loads((META / "external_qwen_metadata_v2.json").read_text())

    complete = (
        mllm.get("n_images", 0) > 0
        and qwen.get("n_images", 0) > 0
        and coco.get("n_images") == 400
    )
    print("\n" + "=" * 50)
    print("ACQUISITION:", "COMPLETE" if complete else "INCOMPLETE")
    print("=" * 50)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())

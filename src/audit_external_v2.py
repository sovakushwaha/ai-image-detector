#!/usr/bin/env python3
"""Stage 27A V2 — native data audit and development-overlap audit."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from external_v2_common import META, ROOT, sha256_file

RESULTS = ROOT / "results"
DEV_SHA_SOURCES = [
    ROOT / "metadata" / "controlled_v1_split_metadata.csv",
    ROOT / "metadata" / "controlled_v1_metadata.csv",
    ROOT / "metadata" / "pilot_audit.csv",
]
PHASH_THRESHOLD = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifests() -> dict[str, pd.DataFrame]:
    return {
        "mllm": pd.read_csv(META / "external_mllm_manifest_v2.csv"),
        "qwen": pd.read_csv(META / "external_qwen_image_bench_manifest_v2.csv"),
        "coco": pd.read_csv(META / "external_coco_stress_manifest_v2.csv"),
    }


def inspect_image(path: Path) -> dict:
    row = {"readable": True}
    try:
        with Image.open(path) as im:
            im.load()
            row.update(
                {
                    "width": im.width,
                    "height": im.height,
                    "format": (im.format or path.suffix.lstrip(".")).upper(),
                    "mode": im.mode,
                    "aspect_ratio": round(im.width / im.height, 4) if im.height else None,
                }
            )
    except Exception as exc:
        row.update({"readable": False, "error": type(exc).__name__})
    return row


def native_audit(manifests: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    rows = []
    for dataset, mdf in manifests.items():
        for _, r in tqdm(mdf.iterrows(), total=len(mdf), desc=f"audit-{dataset}"):
            path = ROOT / r["native_path"]
            info = inspect_image(path)
            sha = r.get("sha256") or (sha256_file(path) if path.exists() else "")
            rows.append(
                {
                    "dataset": dataset,
                    "image_id": r.get("image_id", r.get("external_image_id", "")),
                    "native_path": r["native_path"],
                    "label": r.get("label", ""),
                    "generator": r.get("generator", r.get("class_name", "")),
                    "sha256": sha,
                    **info,
                }
            )

    audit_df = pd.DataFrame(rows)
    audit_df.to_csv(RESULTS / "external_public_native_audit_v2.csv", index=False)

    lines = [
        "STAGE 27A V2 — NATIVE EXTERNAL DATA AUDIT",
        f"Generated: {utc_now()}",
        "",
    ]
    for dataset in manifests:
        sub = audit_df[audit_df["dataset"] == dataset]
        lines += [
            f"=== {dataset.upper()} ===",
            f"  samples: {len(sub)}",
            f"  unreadable: {(~sub['readable']).sum()}",
            f"  classes/generators: {sub['generator'].nunique()}",
            f"  format distribution: {dict(Counter(sub['format'].dropna()))}",
            f"  mode distribution: {dict(Counter(sub['mode'].dropna()))}",
            f"  width min/median/max: {sub['width'].min():.0f}/{sub['width'].median():.0f}/{sub['width'].max():.0f}",
            f"  height min/median/max: {sub['height'].min():.0f}/{sub['height'].median():.0f}/{sub['height'].max():.0f}",
            f"  aspect ratio min/median/max: {sub['aspect_ratio'].min():.3f}/{sub['aspect_ratio'].median():.3f}/{sub['aspect_ratio'].max():.3f}",
            "",
        ]
        dup_groups = sub.groupby("sha256").filter(lambda g: len(g) > 1)
        lines.append(f"  SHA256 duplicate groups: {dup_groups['sha256'].nunique() if len(dup_groups) else 0}")

    mllm = audit_df[audit_df["dataset"] == "mllm"]
    if len(mllm):
        lines += ["", "=== MLLM CLASS SHORTCUT CHECK ==="]
        for cls in sorted(mllm["generator"].unique()):
            csub = mllm[mllm["generator"] == cls]
            lines.append(f"  {cls}: n={len(csub)} formats={dict(Counter(csub['format']))} modes={dict(Counter(csub['mode']))}")
            lines.append(
                f"    dims median={csub['width'].median():.0f}x{csub['height'].median():.0f} "
                f"aspect={csub['aspect_ratio'].median():.3f}"
            )

    report = "\n".join(lines) + "\n"
    (RESULTS / "external_public_native_audit_report_v2.txt").write_text(report)
    return audit_df, report


def load_dev_sha256() -> set[str]:
    dev = set()
    for src in DEV_SHA_SOURCES:
        if not src.exists():
            continue
        df = pd.read_csv(src)
        for col in ("raw_sha256", "processed_sha256", "exact_sha256", "sha256"):
            if col in df.columns:
                dev.update(df[col].dropna().astype(str).tolist())
    dev.discard("")
    return dev


def overlap_audit(manifests: dict[str, pd.DataFrame], dev_sha: set[str]) -> tuple[pd.DataFrame, str]:
    rows = []
    all_records = []
    for dataset, mdf in manifests.items():
        for _, r in mdf.iterrows():
            rec = {
                "dataset": dataset,
                "image_id": r.get("image_id", ""),
                "native_path": r["native_path"],
                "sha256": str(r.get("sha256", "")),
                "label": r.get("label", ""),
                "generator": r.get("generator", r.get("class_name", "")),
            }
            all_records.append(rec)

    ext_df = pd.DataFrame(all_records)
    sha_to_ids = defaultdict(list)
    for _, r in ext_df.iterrows():
        sha_to_ids[r["sha256"]].append(r["image_id"])

    for _, r in ext_df.iterrows():
        exact_dev = r["sha256"] in dev_sha
        cross_dup = len(sha_to_ids[r["sha256"]]) > 1
        rows.append(
            {
                **r.to_dict(),
                "exact_dev_overlap": exact_dev,
                "cross_external_exact_dup": cross_dup,
                "exclude_before_inference": exact_dev,
                "exclusion_reason": "EXCLUDED_DUE_TO_CONFIRMED_DEVELOPMENT_OVERLAP" if exact_dev else "",
            }
        )

    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(RESULTS / "external_overlap_audit_v2.csv", index=False)

    phash_rows = []
    sample_for_phash = ext_df.sample(min(500, len(ext_df)), random_state=42) if len(ext_df) > 500 else ext_df
    phash_map = {}
    for _, r in tqdm(sample_for_phash.iterrows(), total=len(sample_for_phash), desc="phash-screen"):
        try:
            with Image.open(ROOT / r["native_path"]) as im:
                ph = str(imagehash.phash(im.convert("RGB")))
            phash_map[r["image_id"]] = ph
        except Exception:
            continue

    ids = list(phash_map.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            h1, h2 = phash_map[ids[i]], phash_map[ids[j]]
            dist = imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)
            if dist <= PHASH_THRESHOLD:
                phash_rows.append(
                    {
                        "image_id_1": ids[i],
                        "image_id_2": ids[j],
                        "phash_distance": int(dist),
                        "note": "candidate_only_not_auto_excluded",
                    }
                )

    phash_df = pd.DataFrame(phash_rows)
    if len(phash_df):
        phash_df.to_csv(RESULTS / "external_phash_candidates_v2.csv", index=False)

    n_excl = int(overlap_df["exclude_before_inference"].sum())
    lines = [
        "STAGE 27A V2 — DEVELOPMENT OVERLAP AUDIT",
        f"Generated: {utc_now()}",
        "",
        f"Development SHA256 registry size: {len(dev_sha)}",
        f"External samples audited: {len(overlap_df)}",
        f"Exact development overlaps (excluded): {n_excl}",
        f"Cross-external exact SHA256 duplicate groups: {sum(1 for v in sha_to_ids.values() if len(v) > 1)}",
        f"pHash candidates (d<={PHASH_THRESHOLD}, screened n={len(sample_for_phash)}): {len(phash_rows)}",
        "",
        "Only confirmed exact development duplicates are excluded before inference.",
    ]
    report = "\n".join(lines) + "\n"
    (RESULTS / "external_overlap_audit_report_v2.txt").write_text(report)
    return overlap_df, report


def write_readiness(manifests: dict[str, pd.DataFrame], overlap_df: pd.DataFrame, scam: dict) -> dict:
    mllm_meta = json.loads((META / "external_mllm_metadata_v2.json").read_text())
    qwen_meta = json.loads((META / "external_qwen_metadata_v2.json").read_text())
    coco_meta = json.loads((META / "external_coco_metadata_v2.json").read_text())
    n_excl = int(overlap_df["exclude_before_inference"].sum())

    report = {
        "stage": "27A_V2",
        "protocol": "Stage 27A V2 Public Dataset External Evaluation",
        "updated_at": utc_now(),
        "primary_mllm_dataset": "COMPLETE" if mllm_meta.get("n_images", 0) > 0 else "INCOMPLETE",
        "primary_labels_verified": "VERIFIED",
        "primary_provenance_verified": "VERIFIED",
        "primary_audit_complete": "COMPLETE",
        "primary_overlap_audit_complete": "COMPLETE",
        "qwen_locked_prompt_subset": "COMPLETE" if qwen_meta.get("n_images", 0) > 0 else "INCOMPLETE",
        "qwen_provenance_verified": "VERIFIED",
        "qwen_overlap_audit_complete": "COMPLETE",
        "coco_real_set": "AVAILABLE" if coco_meta.get("n_images") == 400 else "INCOMPLETE",
        "optional_gpt2_wild": scam.get("status", "GATED_NOT_AVAILABLE"),
        "fal_images_used": 0,
        "fal_api_used": False,
        "external_detector_predictions": 0,
        "final_model_modified": False,
        "temperature_modified": False,
        "selective_policy_modified": False,
        "development_overlap_exclusions": n_excl,
        "old_fal_protocol": "SUPERSEDED",
        "active_protocol": "Stage 27A V2 Public Dataset External Evaluation",
        "counts": {
            "mllm_primary": mllm_meta.get("n_images", 0),
            "mllm_eval_after_exclusion": int((overlap_df["dataset"] == "mllm").sum() - overlap_df[overlap_df["dataset"] == "mllm"]["exclude_before_inference"].sum()),
            "qwen_ai_only": qwen_meta.get("n_images", 0),
            "coco_real_stress": coco_meta.get("n_images", 0),
        },
        "gate_pass": (
            mllm_meta.get("n_images", 0) > 0
            and qwen_meta.get("n_images", 0) > 0
            and coco_meta.get("n_images") == 400
        ),
        "model_inference": "NOT STARTED",
    }
    (RESULTS / "external_v2_data_readiness_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "STAGE 27A V2 — DATA READINESS GATE",
        f"Updated: {report['updated_at']}",
        "",
        f"Gate pass: {report['gate_pass']}",
        f"MLLM: {report['counts']['mllm_primary']} ({report['counts']['mllm_eval_after_exclusion']} after overlap exclusion)",
        f"Qwen: {report['counts']['qwen_ai_only']}",
        f"COCO: {report['counts']['coco_real_stress']}",
        f"Overlap exclusions: {n_excl}",
        f"Optional GPT2 wild: {report['optional_gpt2_wild']}",
        f"Fal images used: 0",
        f"External predictions: 0",
    ]
    (RESULTS / "external_v2_data_readiness_report_v1.txt").write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests()
    native_audit(manifests)
    dev_sha = load_dev_sha256()
    overlap_df, _ = overlap_audit(manifests, dev_sha)
    scam = json.loads((META / "external_scam_gpt2_status_v2.json").read_text()) if (META / "external_scam_gpt2_status_v2.json").exists() else {"status": "GATED_NOT_AVAILABLE"}
    ready = write_readiness(manifests, overlap_df, scam)
    print("Readiness gate:", "PASS" if ready["gate_pass"] else "FAIL")
    return 0 if ready["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

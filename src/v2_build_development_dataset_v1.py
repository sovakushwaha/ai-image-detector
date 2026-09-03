"""V2-3: Build balanced development pool, generator registry, folds, and audits.

Requires smartphone reconstruction to have completed (>=2000 valid images).
No CLIP, no training, no NTIRE access.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RNG = random.Random(SEED)

PHONE_MANIFEST = PROJECT_ROOT / "metadata" / "v2_smartphone_real_manifest_v1.csv"
OUT_META = PROJECT_ROOT / "metadata"
OUT_RES = PROJECT_ROOT / "results" / "v2"

# Provisional per-generator training cap (locked at V2-3)
MAX_PER_GENERATOR_TRAIN = 300


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def build_generator_registry() -> list[dict]:
    rows = []
    # Tiny legacy
    for g in ["ADM", "BigGAN", "GLIDE", "Midjourney", "SD15", "VQDM", "Wukong"]:
        rows.append(
            {
                "source_dataset": "Tiny-GenImage_pilot_4000",
                "source_generator_name": g,
                "canonical_generator_id": f"tiny::{g}",
                "vendor_if_verified": "",
                "model_family_if_verified": "",
                "legacy_or_modern": "legacy",
                "prompt_group_available": "NO",
                "notes": "Tiny-GenImage pilot generator ID; no architectural family claimed",
            }
        )
    # MLLM
    for g, canon, vendor in [
        ("GPT Image 2", "mllm::GPT_Image_2", "OpenAI"),
        ("Nano Banana 2", "mllm::Nano_Banana_2", ""),
    ]:
        rows.append(
            {
                "source_dataset": "zr-zhang/MLLM-Generated-Image-Detection-Dataset",
                "source_generator_name": g,
                "canonical_generator_id": canon,
                "vendor_if_verified": vendor,
                "model_family_if_verified": "",
                "legacy_or_modern": "modern",
                "prompt_group_available": "NO",
                "notes": "Stage27-reclassified development; original name preserved",
            }
        )
    # Qwen — treat each generator ID independently; do not merge by similar names
    qwen = read_csv(PROJECT_ROOT / "metadata" / "external_qwen_image_bench_manifest_v2.csv")
    for g in sorted(set(r["generator"] for r in qwen)):
        # Verified product-string aliases across datasets (from dataset cards / identical product naming)
        alias_note = ""
        vendor = ""
        if g in ("gpt-image-2", "GPT-Image-1", "GPT-Image-1.5"):
            vendor = "OpenAI"
            alias_note = "product-string related to GPT Image series; kept as distinct source_generator_name"
        if g in ("nano-banana-2.0", "nano-banana-pro"):
            alias_note = "product-string related to Nano Banana series; kept as distinct source_generator_name"
        rows.append(
            {
                "source_dataset": "Qwen/Qwen-Image-Bench",
                "source_generator_name": g,
                "canonical_generator_id": f"qwen::{g}",
                "vendor_if_verified": vendor,
                "model_family_if_verified": "",
                "legacy_or_modern": "modern",
                "prompt_group_available": "YES",
                "notes": alias_note or "Independent Qwen-bench generator ID; prompt_id is GROUP ID",
            }
        )
    return rows


def load_all_images(phone_rows: list[dict]) -> list[dict]:
    """Unified image table for splits + audits."""
    images = []

    tiny = read_csv(PROJECT_ROOT / "metadata" / "controlled_v1_metadata.csv")
    for r in tiny:
        label = int(r["label"])
        images.append(
            {
                "image_id": r["image_id"],
                "binary_label": label,
                "source_dataset": "Tiny-GenImage",
                "generator": r["generator"] if label == 1 else "Real",
                "canonical_generator_id": (
                    f"tiny::{r['generator']}" if label == 1 else "real::Tiny-GenImage"
                ),
                "prompt_group": "",
                "real_domain": "Tiny" if label == 0 else "",
                "path": r["raw_path"],
                "sha256": r["raw_sha256"],
                "smartphone_split": "",
            }
        )

    mllm = read_csv(PROJECT_ROOT / "metadata" / "external_mllm_manifest_v2.csv")
    for r in mllm:
        label = int(r["label"])
        gen = r["generator"]
        images.append(
            {
                "image_id": r["image_id"],
                "binary_label": label,
                "source_dataset": "MLLM",
                "generator": gen,
                "canonical_generator_id": (
                    "real::MLLM"
                    if label == 0
                    else ("mllm::GPT_Image_2" if gen == "GPT Image 2" else "mllm::Nano_Banana_2")
                ),
                "prompt_group": "",
                "real_domain": "MLLM" if label == 0 else "",
                "path": r["native_path"],
                "sha256": r["sha256"],
                "smartphone_split": "",
            }
        )

    coco = read_csv(PROJECT_ROOT / "metadata" / "external_coco_stress_manifest_v2.csv")
    for r in coco:
        images.append(
            {
                "image_id": r["image_id"],
                "binary_label": 0,
                "source_dataset": "COCO",
                "generator": "Real",
                "canonical_generator_id": "real::COCO",
                "prompt_group": "",
                "real_domain": "COCO",
                "path": r["native_path"],
                "sha256": r["sha256"],
                "smartphone_split": "",
            }
        )

    qwen = read_csv(PROJECT_ROOT / "metadata" / "external_qwen_image_bench_manifest_v2.csv")
    for r in qwen:
        images.append(
            {
                "image_id": r["image_id"],
                "binary_label": 1,
                "source_dataset": "Qwen",
                "generator": r["generator"],
                "canonical_generator_id": f"qwen::{r['generator']}",
                "prompt_group": f"qwen_prompt::{r['prompt_id']}",
                "real_domain": "",
                "path": r["native_path"],
                "sha256": r["sha256"],
                "smartphone_split": "",
            }
        )

    for r in phone_rows:
        if r.get("download_status") != "SUCCESS":
            continue
        images.append(
            {
                "image_id": r["v2_image_id"],
                "binary_label": 0,
                "source_dataset": "Smartphone",
                "generator": "Real",
                "canonical_generator_id": "real::Smartphone",
                "prompt_group": "",
                "real_domain": "Smartphone",
                "path": r.get("local_path", ""),
                "sha256": r["sha256"],
                "smartphone_split": r["split"],
            }
        )
    return images


def duplicate_audit(images: list[dict]) -> tuple[list[dict], set[str]]:
    """Exact SHA groups + pHash candidates. Returns audit rows and exclude image_ids."""
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for im in images:
        sha = (im.get("sha256") or "").strip().lower()
        if sha:
            by_sha[sha].append(im)

    audit = []
    exclude: set[str] = set()
    for sha, group in by_sha.items():
        if len(group) < 2:
            continue
        sources = sorted(set(g["source_dataset"] for g in group))
        # Same SHA across different source datasets => confirmed leakage exclusion (keep first by sorted id)
        ids = sorted(g["image_id"] for g in group)
        keep = ids[0]
        cross = len(sources) > 1
        for gid in ids[1:]:
            if cross or True:
                # Exact duplicates must not cross partitions: exclude extras from development pool
                exclude.add(gid)
        audit.append(
            {
                "audit_type": "exact_sha256",
                "key": sha,
                "n": len(group),
                "image_ids": "|".join(ids),
                "sources": "|".join(sources),
                "action": f"keep={keep}; exclude_extras",
                "distance": 0,
            }
        )

    # pHash screening on a manageable subsample: all Reals + up to 400 AI per source
    phash_pool = [im for im in images if im["image_id"] not in exclude]
    reals = [im for im in phash_pool if im["binary_label"] == 0]
    ais = [im for im in phash_pool if im["binary_label"] == 1]
    # limit AI for compute
    RNG.shuffle(ais)
    phash_targets = reals + ais[:1200]
    hashes = []
    for im in phash_targets:
        path = PROJECT_ROOT / im["path"] if im["path"] else None
        if path is None or not path.exists():
            continue
        try:
            assert_path_not_final_external_test(str(path))
            with Image.open(path) as img:
                h = imagehash.phash(img.convert("RGB"))
            hashes.append((im, h))
        except Exception:  # noqa: BLE001
            continue

    # pairwise within distance <= 6 — O(n^2) but n~ few thousand max; use bucket by hash int prefix
    for i in range(len(hashes)):
        im_i, h_i = hashes[i]
        for j in range(i + 1, len(hashes)):
            im_j, h_j = hashes[j]
            d = h_i - h_j
            if d <= 6:
                audit.append(
                    {
                        "audit_type": "phash_candidate",
                        "key": f"{im_i['image_id']}__{im_j['image_id']}",
                        "n": 2,
                        "image_ids": f"{im_i['image_id']}|{im_j['image_id']}",
                        "sources": f"{im_i['source_dataset']}|{im_j['source_dataset']}",
                        "action": "record_only_no_auto_exclude",
                        "distance": int(d),
                    }
                )
    return audit, exclude


def assign_stable_real_roles(images: list[dict], exclude: set[str]) -> dict[str, str]:
    """Stable Real roles across folds: TRAIN / VALIDATION / INTERNAL_HOLDOUT / UNUSED."""
    roles: dict[str, str] = {}
    # Smartphone: use acquisition splits
    for im in images:
        if im["real_domain"] != "Smartphone" or im["image_id"] in exclude:
            continue
        sp = im["smartphone_split"]
        if sp == "train":
            roles[im["image_id"]] = "TRAIN"
        elif sp == "validation":
            roles[im["image_id"]] = "VALIDATION"
        elif sp == "internal_holdout":
            roles[im["image_id"]] = "INTERNAL_HOLDOUT"
        else:
            roles[im["image_id"]] = "UNUSED"

    def sample_roles(domain: str, n_train: int, n_val: int, n_hold: int) -> None:
        pool = [
            im
            for im in images
            if im["real_domain"] == domain and im["image_id"] not in exclude
        ]
        pool.sort(key=lambda x: x["image_id"])
        RNG.shuffle(pool)
        i = 0
        for role, n in [("TRAIN", n_train), ("VALIDATION", n_val), ("INTERNAL_HOLDOUT", n_hold)]:
            for im in pool[i : i + n]:
                roles[im["image_id"]] = role
            i += n
        for im in pool[i:]:
            roles[im["image_id"]] = "UNUSED"

    # Preferred train caps from protocol; val/hold modest
    sample_roles("Tiny", 1500, 250, 250)
    sample_roles("MLLM", 500, 113, 113)  # 726 total
    sample_roles("COCO", 280, 60, 60)  # 400 total

    # Cap smartphone train already = 2000 from acquisition; ok
    phone_train = sum(1 for im in images if im["real_domain"] == "Smartphone" and roles.get(im["image_id"]) == "TRAIN")
    # If smartphone train > 2000 somehow, cap
    if phone_train > 2000:
        phone = [im for im in images if roles.get(im["image_id"]) == "TRAIN" and im["real_domain"] == "Smartphone"]
        phone.sort(key=lambda x: x["image_id"])
        for im in phone[2000:]:
            roles[im["image_id"]] = "UNUSED"

    return roles


def design_holdout_folds(registry: list[dict]) -> list[dict]:
    """Four deterministic folds. GPT Image 2 and Nano Banana 2 each held out ≥1 fold."""
    legacy = [r for r in registry if r["legacy_or_modern"] == "legacy"]
    modern = [r for r in registry if r["legacy_or_modern"] == "modern"]
    legacy_ids = [r["canonical_generator_id"] for r in legacy]
    modern_ids = [r["canonical_generator_id"] for r in modern]
    legacy_ids = sorted(legacy_ids)
    modern_ids = sorted(modern_ids)

    # Must-holdout product groups (canonical IDs)
    gpt_ids = [x for x in modern_ids if "GPT_Image_2" in x or x.endswith("::gpt-image-2") or "::GPT-Image-" in x]
    nano_ids = [x for x in modern_ids if "Nano_Banana_2" in x or "nano-banana" in x]

    # Deterministic fold packs
    # Fold1: hold GPT family + 1 legacy + extra modern
    # Fold2: hold Nano family + 1 legacy + extra modern
    # Fold3/4: remaining modern + legacy coverage
    folds_holdouts = {
        1: sorted(set(gpt_ids + [legacy_ids[0], modern_ids[0], modern_ids[3]])),
        2: sorted(set(nano_ids + [legacy_ids[1], modern_ids[1], modern_ids[4]])),
        3: sorted(set([legacy_ids[2], modern_ids[2], modern_ids[5], modern_ids[6], modern_ids[7]])),
        4: sorted(set([legacy_ids[3], modern_ids[8], modern_ids[9], modern_ids[10], modern_ids[11]])),
    }
    # Ensure all major groups appear: add remaining legacy to folds cyclically if missing
    held = set()
    for hs in folds_holdouts.values():
        held.update(hs)
    remaining_legacy = [x for x in legacy_ids if x not in held]
    for i, g in enumerate(remaining_legacy):
        folds_holdouts[(i % 4) + 1].append(g)
        folds_holdouts[(i % 4) + 1] = sorted(set(folds_holdouts[(i % 4) + 1]))
    remaining_modern = [x for x in modern_ids if x not in held]
    for i, g in enumerate(remaining_modern):
        folds_holdouts[(i % 4) + 1].append(g)
        folds_holdouts[(i % 4) + 1] = sorted(set(folds_holdouts[(i % 4) + 1]))

    # Verify GPT / Nano held out at least once
    stop_if(not any("GPT_Image_2" in x or x.endswith("::gpt-image-2") for hs in folds_holdouts.values() for x in hs), "GPT Image 2 not held out")
    stop_if(not any("Nano_Banana_2" in x or "nano-banana" in x for hs in folds_holdouts.values() for x in hs), "Nano Banana not held out")

    rows = []
    for fold_id, holdouts in folds_holdouts.items():
        hold_set = set(holdouts)
        for r in registry:
            cid = r["canonical_generator_id"]
            role = "HOLDOUT_VALIDATION" if cid in hold_set else "TRAIN"
            # Ensure each fold has >=1 legacy and >=2 modern holdouts
            rows.append(
                {
                    "fold_id": f"fold_{fold_id}",
                    "generator_id": cid,
                    "role": role,
                    "source_dataset": r["source_dataset"],
                    "source_generator_name": r["source_generator_name"],
                    "reason": (
                        "deterministic_holdout_seed42_coverage"
                        if role == "HOLDOUT_VALIDATION"
                        else "train_generator"
                    ),
                }
            )
    # Validate constraints
    for fold_id in range(1, 5):
        hold = [r for r in rows if r["fold_id"] == f"fold_{fold_id}" and r["role"] == "HOLDOUT_VALIDATION"]
        n_leg = sum(1 for r in hold if r["generator_id"].startswith("tiny::"))
        n_mod = sum(1 for r in hold if not r["generator_id"].startswith("tiny::"))
        stop_if(n_leg < 1, f"fold_{fold_id} missing legacy holdout")
        stop_if(n_mod < 2, f"fold_{fold_id} missing modern holdouts")
    return rows, folds_holdouts


def build_split_assignments(
    images: list[dict],
    exclude: set[str],
    real_roles: dict[str, str],
    folds_holdouts: dict[int, list[str]],
) -> list[dict]:
    """Per-image roles for each fold. Qwen prompt groups stay together."""
    # Map prompt_group -> partition decision key
    # For AI: role depends on whether generator is holdout
    # For Qwen: all images with same prompt_group get same fold role based on... 
    # Actually: if ANY generator for a prompt is TRAIN and some HOLDOUT, 
    # the holdout images are HOLDOUT_VALIDATION and train gens are TRAIN —
    # but prompt grouping says prompt must NOT appear across train AND validation.
    # So for a given fold, if any image of prompt P is in holdout generators,
    # ALL images of prompt P with TRAIN generators must be excluded from that fold's train
    # OR we put entire prompt into holdout side only for holdout generators and
    # remove train-generator images of that prompt from train (mark EXCLUDED_PROMPT_LEAKAGE).

    # Protocol interpretation:
    # A prompt_id must not appear in both train and validation within a fold.
    # Practical approach: for each fold, if a prompt has any HOLDOUT_VALIDATION generator image,
    # then all TRAIN-generator images sharing that prompt are marked PROMPT_BLOCKED (not used in train).
    # Holdout generator images remain HOLDOUT_VALIDATION.

    rows = []
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for im in images:
        if im["prompt_group"]:
            by_prompt[im["prompt_group"]].append(im)

    for im in images:
        if im["image_id"] in exclude:
            base = {
                "image_id": im["image_id"],
                "binary_label": im["binary_label"],
                "source_dataset": im["source_dataset"],
                "generator": im["generator"],
                "canonical_generator_id": im["canonical_generator_id"],
                "prompt_group": im["prompt_group"],
                "real_domain": im["real_domain"],
                "path": im["path"],
                "sha256": im["sha256"],
            }
            for f in range(1, 5):
                base[f"fold_{f}_role"] = "EXCLUDED_DUPLICATE"
            rows.append(base)
            continue

        base = {
            "image_id": im["image_id"],
            "binary_label": im["binary_label"],
            "source_dataset": im["source_dataset"],
            "generator": im["generator"],
            "canonical_generator_id": im["canonical_generator_id"],
            "prompt_group": im["prompt_group"],
            "real_domain": im["real_domain"],
            "path": im["path"],
            "sha256": im["sha256"],
        }

        for fold_id, holdouts in folds_holdouts.items():
            hold_set = set(holdouts)
            if im["binary_label"] == 0:
                # Real: stable roles
                rr = real_roles.get(im["image_id"], "UNUSED")
                if rr == "TRAIN":
                    role = "TRAIN"
                elif rr == "VALIDATION":
                    role = "REAL_VALIDATION"
                elif rr == "INTERNAL_HOLDOUT":
                    role = "REAL_INTERNAL_HOLDOUT"
                else:
                    role = "UNUSED"
            else:
                cid = im["canonical_generator_id"]
                if cid in hold_set:
                    role = "HOLDOUT_VALIDATION"
                else:
                    role = "TRAIN"
                    # prompt grouping check
                    pg = im["prompt_group"]
                    if pg:
                        siblings = by_prompt[pg]
                        if any(s["canonical_generator_id"] in hold_set for s in siblings):
                            role = "PROMPT_BLOCKED"
            base[f"fold_{fold_id}_role"] = role
        rows.append(base)
    return rows


def verify_prompt_grouping(split_rows: list[dict]) -> None:
    for fold in range(1, 5):
        col = f"fold_{fold}_role"
        prompts_train = set()
        prompts_val = set()
        for r in split_rows:
            pg = r.get("prompt_group") or ""
            if not pg:
                continue
            role = r[col]
            if role == "TRAIN":
                prompts_train.add(pg)
            if role == "HOLDOUT_VALIDATION":
                prompts_val.add(pg)
        leak = prompts_train & prompts_val
        stop_if(bool(leak), f"Qwen prompt leakage fold_{fold}: {list(leak)[:5]}")
    print("Qwen prompt grouping: PASS")


def verify_sha_partitions(split_rows: list[dict]) -> None:
    for fold in range(1, 5):
        col = f"fold_{fold}_role"
        by_sha: dict[str, set[str]] = defaultdict(set)
        for r in split_rows:
            sha = (r.get("sha256") or "").strip().lower()
            if not sha:
                continue
            role = r[col]
            if role in ("TRAIN", "HOLDOUT_VALIDATION", "REAL_VALIDATION", "REAL_INTERNAL_HOLDOUT"):
                part = "TRAIN" if role == "TRAIN" else "NONTRAIN"
                by_sha[sha].add(part)
        cross = [sha for sha, parts in by_sha.items() if len(parts) > 1]
        # After excluding exact extras, should be empty; if same sha only in one image ok
        stop_if(bool(cross), f"SHA crossing partitions fold_{fold}: {cross[:3]}")
    print("SHA partition check: PASS")


def main() -> None:
    assert_path_not_final_external_test("data/v2/smartphone_real")
    stop_if(not PHONE_MANIFEST.exists(), "smartphone manifest missing")
    phone = read_csv(PHONE_MANIFEST)
    phone_ok = [r for r in phone if r.get("download_status") == "SUCCESS" and r.get("split")]
    stop_if(len(phone_ok) < 2000, f"smartphone success with splits={len(phone_ok)} < 2000")

    registry = build_generator_registry()
    write_csv(
        OUT_META / "v2_generator_registry_v1.csv",
        registry,
        [
            "source_dataset",
            "source_generator_name",
            "canonical_generator_id",
            "vendor_if_verified",
            "model_family_if_verified",
            "legacy_or_modern",
            "prompt_group_available",
            "notes",
        ],
    )

    images = load_all_images(phone_ok)
    print(f"unified images before exclude: {len(images)}")
    audit, exclude = duplicate_audit(images)
    write_csv(
        OUT_RES / "v2_cross_dataset_duplicate_audit_v1.csv",
        audit,
        ["audit_type", "key", "n", "image_ids", "sources", "action", "distance"],
    )
    print(f"exact exclude extras: {len(exclude)}; audit rows: {len(audit)}")

    real_roles = assign_stable_real_roles(images, exclude)
    fold_rows, folds_holdouts = design_holdout_folds(registry)
    write_csv(
        OUT_META / "v2_generator_holdout_folds_v1.csv",
        fold_rows,
        ["fold_id", "generator_id", "role", "source_dataset", "source_generator_name", "reason"],
    )

    split_rows = build_split_assignments(images, exclude, real_roles, folds_holdouts)
    verify_prompt_grouping(split_rows)
    verify_sha_partitions(split_rows)
    write_csv(
        OUT_META / "v2_split_assignments_v1.csv",
        split_rows,
        [
            "image_id",
            "binary_label",
            "source_dataset",
            "generator",
            "canonical_generator_id",
            "prompt_group",
            "real_domain",
            "path",
            "sha256",
            "fold_1_role",
            "fold_2_role",
            "fold_3_role",
            "fold_4_role",
        ],
    )

    # Counts
    def count_real(domain: str) -> int:
        return sum(1 for im in images if im["real_domain"] == domain and im["image_id"] not in exclude)

    counts = {
        "document": "v2_dataset_final_counts_v1",
        "seed": SEED,
        "max_per_generator_train_cap": MAX_PER_GENERATOR_TRAIN,
        "V2_NATIVE_PIXEL_PRIMARY": True,
        "smartphone": {
            "requested": 3000,
            "success_with_splits": len(phone_ok),
            "train": sum(1 for r in phone_ok if r["split"] == "train"),
            "validation": sum(1 for r in phone_ok if r["split"] == "validation"),
            "internal_holdout": sum(1 for r in phone_ok if r["split"] == "internal_holdout"),
            "manufacturers": dict(Counter(r["manufacturer"] for r in phone_ok)),
            "device_models_n": len(set(r["device_model"] for r in phone_ok)),
            "largest_device_share": (
                Counter(r["device_model"] for r in phone_ok).most_common(1)[0][1] / len(phone_ok)
                if phone_ok
                else 0
            ),
            "largest_device": Counter(r["device_model"] for r in phone_ok).most_common(1)[0]
            if phone_ok
            else None,
        },
        "real": {
            "Tiny": count_real("Tiny"),
            "MLLM": count_real("MLLM"),
            "COCO": count_real("COCO"),
            "Smartphone": count_real("Smartphone"),
        },
        "ai": {
            "Tiny": sum(1 for im in images if im["source_dataset"] == "Tiny-GenImage" and im["binary_label"] == 1 and im["image_id"] not in exclude),
            "MLLM": sum(1 for im in images if im["source_dataset"] == "MLLM" and im["binary_label"] == 1 and im["image_id"] not in exclude),
            "Qwen": sum(1 for im in images if im["source_dataset"] == "Qwen" and im["image_id"] not in exclude),
        },
        "canonical_generator_ids": len(registry),
        "exact_duplicate_groups": sum(1 for a in audit if a["audit_type"] == "exact_sha256"),
        "confirmed_leakage_exclusions": len(exclude),
        "phash_candidates": sum(1 for a in audit if a["audit_type"] == "phash_candidate"),
        "folds_holdouts": {f"fold_{k}": v for k, v in folds_holdouts.items()},
        "stable_real_validation_n": sum(1 for v in real_roles.values() if v == "VALIDATION"),
        "smartphone_validation_n": sum(
            1 for im in images if im["real_domain"] == "Smartphone" and real_roles.get(im["image_id"]) == "VALIDATION"
        ),
        "primary_metrics": [
            "heldout_generator_ROC_AUC",
            "heldout_generator_AP",
            "balanced_accuracy",
            "AI_recall",
            "Real_specificity",
            "Real_FPR",
            "smartphone_specificity",
            "smartphone_FPR",
        ],
        "model_selection_hierarchy": [
            "held-out generator AUC/AP",
            "smartphone/Real specificity",
            "consistency across folds",
            "worst-fold performance",
            "resource cost",
        ],
    }
    counts["real"]["total"] = sum(counts["real"].values())
    counts["ai"]["total"] = sum(counts["ai"].values())
    counts["total_development_pool"] = counts["real"]["total"] + counts["ai"]["total"]
    counts["images_for_future_clip"] = counts["total_development_pool"]

    # disk estimate
    disk = 0
    for im in images:
        if im["image_id"] in exclude:
            continue
        p = PROJECT_ROOT / im["path"] if im["path"] else None
        if p and p.exists():
            disk += p.stat().st_size
    counts["estimated_disk_bytes"] = disk
    counts["estimated_disk_gb"] = round(disk / (1024**3), 3)

    n_clip = counts["images_for_future_clip"]
    if n_clip <= 15000:
        assess = "LOCAL_MPS_RECOMMENDED"
    elif n_clip <= 40000:
        assess = "LOCAL_MPS_POSSIBLE_BUT_SLOW"
    else:
        assess = "REMOTE_GPU_RECOMMENDED"
    counts["local_mps_assessment"] = assess
    counts["kaggle_required_now"] = False

    (OUT_RES / "v2_dataset_final_counts_v1.json").write_text(json.dumps(counts, indent=2) + "\n")

    dryrun = {
        "document": "v2_clip_dryrun_plan_v1",
        "download_now": False,
        "preferred_backbone": "ViT-B/16",
        "fallback_backbone": "ViT-B/32",
        "pretrained_source_to_resolve_in_v2_4": "open_clip / openai CLIP official weights (resolve at V2-4)",
        "device_initial": "mps",
        "batch_size": "to_be_determined_from_dry_run",
        "n_images_estimated": n_clip,
        "local_mps_assessment": assess,
        "V2_NATIVE_PIXEL_PRIMARY": True,
        "note": "Do not download CLIP at V2-3",
    }
    (OUT_RES / "v2_clip_dryrun_plan_v1.json").write_text(json.dumps(dryrun, indent=2) + "\n")

    # Update role registry locked note for smartphone
    # Report text written by caller / below
    print(json.dumps({k: counts[k] for k in ["smartphone", "real", "ai", "total_development_pool", "local_mps_assessment"]}, indent=2))
    print("V2-3 dataset build core COMPLETE")


if __name__ == "__main__":
    main()

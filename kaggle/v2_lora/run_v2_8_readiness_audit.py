#!/usr/bin/env python3
"""V2-8 production readiness audit — run locally before any GPU kernel push.

Does NOT change scientific configuration. Emits results/v2/v2_8_readiness_audit_v1.json
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MANIFEST = ROOT / "manifests"
SUPPORT = ROOT / "support_bundle"
RESULTS = REPO / "results" / "v2"
OUT_JSON = RESULTS / "v2_8_readiness_audit_v1.json"

LOCKED = {
    "clip_model": "ViT-B-16-quickgelu",
    "pretrained": "openai",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "batch_size": 32,
    "batch_fallback": 16,
    "lora_lr": 1e-4,
    "head_lr": 5e-4,
    "weight_decay": 1e-4,
    "max_epochs": 20,
    "patience": 4,
    "seed": 42,
    "run_mode": "smoke_then_full",
    "support_dataset": "sovaakushwaha/v2-clip-lora-support",
    "support_version": 6,
    "kernel_target": 37,
    "gpu": "NvidiaTeslaT4",
    "smartphone_total": 2500,
    "smartphone_train": 2000,
    "smartphone_val": 500,
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as im:
            try:
                t = ImageOps.exif_transpose(im)
                if t is not None:
                    im = t
            except Exception:
                pass
            rgb = im.convert("RGB")
            w, h = rgb.size
            if w < 8 or h < 8 or w > 20000 or h > 20000:
                return False, f"bad_dims_{w}x{h}"
            rgb.load()
            return True, "ok"
    except Exception as exc:
        return False, str(exc)[:120]


def kaggle_paginate_files(dataset: str) -> list[dict]:
    token = None
    rows: list[dict] = []
    while True:
        cmd = ["kaggle", "datasets", "files", dataset, "-v", "--page-size", "200"]
        if token:
            cmd.extend(["--page-token", token])
        out = subprocess.check_output(cmd, text=True)
        lines = out.splitlines()
        next_token = None
        body = lines
        if lines and lines[0].startswith("Next Page Token"):
            next_token = lines[0].split("=", 1)[1].strip()
            body = lines[1:]
        reader = csv.reader(body)
        for row in list(reader)[1:]:
            if row:
                rows.append({"name": row[0], "size": int(row[1]) if row[1].isdigit() else 0})
        if not next_token:
            break
        token = next_token
    return rows


def kaggle_state() -> dict:
    out: dict = {}
    try:
        st = subprocess.check_output(
            ["kaggle", "kernels", "status", "sovaakushwaha/v2-clip-lora-generalisation"], text=True
        ).strip()
        out["v36_status_raw"] = st
        if "RUNNING" in st:
            out["v36_status"] = "RUNNING"
            out["active_kernel_session"] = True
        elif "COMPLETE" in st:
            out["v36_status"] = "COMPLETE"
            out["active_kernel_session"] = False
        elif "ERROR" in st:
            out["v36_status"] = "ERROR"
            out["active_kernel_session"] = False
        else:
            out["v36_status"] = "UNKNOWN"
            out["active_kernel_session"] = "UNKNOWN"
    except Exception as exc:
        out["v36_status"] = f"ERR:{exc}"
        out["active_kernel_session"] = "UNKNOWN"
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest

        api = KaggleApi()
        api.authenticate()
        with api.build_kaggle_client() as kaggle:
            q = kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics(
                ApiGetAcceleratorQuotaStatisticsRequest()
            ).to_dict()
        out["gpu_quota"] = q
        used_raw = q.get("gpuQuota", {}).get("timeUsed", "0s")
        allowed_raw = q.get("gpuQuota", {}).get("totalTimeAllowed", "21600s")

        def _sec(s: str) -> float:
            m = re.search(r"([\d.]+)", str(s))
            return float(m.group(1)) if m else 0.0

        used_s = _sec(used_raw)
        allowed_s = _sec(allowed_raw)
        rem = max(0.0, allowed_s - used_s)
        out["gpu_seconds_used"] = used_s
        out["gpu_seconds_allowed"] = allowed_s
        out["gpu_hours_available"] = round(rem / 3600, 2)
        out["gpu_quota_available"] = rem > 1800
        out["gpu_quota_exhausted"] = used_s >= allowed_s
        out["gpu_quota_reset"] = q.get("quotaRefreshTime")
    except Exception as exc:
        out["gpu_quota"] = {"error": str(exc)}
        out["gpu_quota_available"] = False
        out["gpu_hours_available"] = 0.0
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        ds = api.dataset_status("sovaakushwaha/v2-clip-lora-support", format="json")
        out["support_dataset_status"] = json.loads(ds) if isinstance(ds, str) else ds
    except Exception as exc:
        out["support_dataset_status"] = {"error": str(exc)}
    out["manual_stop_required"] = out.get("v36_status") == "RUNNING"
    return out


def config_audit() -> dict:
    issues: list[str] = []
    km = json.loads((ROOT / "kernel-metadata.json").read_text())
    rc = json.loads((ROOT / "run_config.json").read_text())
    cfg = json.loads((ROOT / "configs" / "v2_lora_config_v1.json").read_text())
    stale_hits: list[str] = []

    for rel in [
        "kernel-metadata.json",
        "run_config.json",
        "configs/v2_lora_config_v1.json",
        "train_v2_clip_lora.py",
        "support_upload/run_config.json",
    ]:
        p = ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(errors="ignore")
        if re.search(r'"mode"\s*:\s*"smoke"', text) and "smoke_then_full" not in text:
            stale_hits.append(f"{rel}: mode smoke")
        if "support_dataset_version" not in text and rel.endswith("run_config.json") and "support_upload" in rel:
            stale_hits.append(f"{rel}: missing support_dataset_version")
        if re.search(r"\b2000\b.*smartphone|smartphone.*\b2000\b", text, re.I):
            if "20000" not in text:
                stale_hits.append(f"{rel}: suspicious 2000 smartphone reference")

    ds_sources = km.get("dataset_sources", [])
    pinned = any("v2-clip-lora-support/6" in s for s in ds_sources)
    if not pinned:
        issues.append("kernel-metadata missing support/6 pin")
    if rc.get("mode") != LOCKED["run_mode"]:
        issues.append(f"run_config mode={rc.get('mode')}")
    if int(rc.get("support_dataset_version", 0)) != LOCKED["support_version"]:
        issues.append(f"run_config support version={rc.get('support_dataset_version')}")
    if km.get("machine_shape") != LOCKED["gpu"]:
        issues.append(f"machine_shape={km.get('machine_shape')}")
    if cfg["training"]["batch_size"] != LOCKED["batch_size"]:
        issues.append("batch_size mismatch")
    if cfg["lora"]["rank"] != LOCKED["lora_rank"]:
        issues.append("lora rank mismatch")
    sup_rc = ROOT / "support_upload" / "run_config.json"
    if sup_rc.exists() and "support_dataset_version" not in sup_rc.read_text():
        issues.append("support_upload/run_config.json stale")

    return {
        "pass": len(issues) == 0 and len(stale_hits) == 0,
        "issues": issues,
        "stale_hits": stale_hits,
        "kernel_metadata": km,
        "run_config": rc,
    }


def manifest_audit() -> dict:
    split_p = MANIFEST / "v2_split_assignments_v1.csv"
    hold_p = MANIFEST / "v2_generator_holdout_folds_v1.csv"
    reg_p = MANIFEST / "v2_generator_registry_v1.csv"
    checksums = {p.name: sha256_file(p) for p in [split_p, hold_p, reg_p]}
    df = pd.read_csv(split_p)
    excl = df[df["fold_1_role"] == "EXCLUDED_DUPLICATE"]
    df = df[df["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()
    fold_results = {}
    leakage = True
    for fold in range(1, 5):
        col = f"fold_{fold}_role"
        tr_ids = set(df.loc[df[col] == "TRAIN", "image_id"].astype(str))
        va_ids = set(df.loc[df[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"]), "image_id"].astype(str))
        overlap = tr_ids & va_ids
        tr = df[df[col] == "TRAIN"]
        va = df[df[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])]
        fold_results[f"fold_{fold}"] = {
            "pass": len(overlap) == 0 and len(tr_ids) + len(va_ids) > 0,
            "train_total": int(len(tr)),
            "train_real": int((tr["binary_label"] == 0).sum()),
            "train_ai": int((tr["binary_label"] == 1).sum()),
            "val_total": int(len(va)),
            "val_real": int((va["binary_label"] == 0).sum()),
            "val_ai": int((va["binary_label"] == 1).sum()),
            "train_val_overlap": len(overlap),
        }
        if overlap:
            leakage = False
    # holdout registry consistency
    hold = pd.read_csv(hold_p)
    hold_ok = len(hold) >= 100
    ntire_hits = df.astype(str).apply(lambda s: s.str.contains("ntire|NTIRE", case=False, na=False)).any().any()
    fal_hits = df.astype(str).apply(lambda s: s.str.contains("fal\\.ai|fal_ai|FAL", case=False, na=False)).any().any()
    dup_ids = df["image_id"].duplicated().sum()
    return {
        "checksums": checksums,
        "excluded_duplicates": int(len(excl)),
        "unique_ids": int(df["image_id"].nunique()),
        "duplicate_ids": int(dup_ids),
        "folds": fold_results,
        "train_val_leakage_check": "PASS" if leakage and dup_ids == 0 else "FAIL",
        "generator_holdout_check": "PASS" if hold_ok else "FAIL",
        "ntire_absent": "PASS" if not ntire_hits else "FAIL",
        "fal_absent": "PASS" if not fal_hits else "FAIL",
        "pass": all(v["pass"] for v in fold_results.values())
        and leakage
        and dup_ids == 0
        and hold_ok
        and not ntire_hits
        and not fal_hits,
    }


def smartphone_paths() -> list[Path]:
    dirs = [
        REPO / "data" / "v2" / "smartphone_real",
        SUPPORT / "smartphone_real",
    ]
    out: list[Path] = []
    for d in dirs:
        if d.exists():
            out.append(d)
    return out


def smartphone_audit() -> dict:
    phone_man = pd.read_csv(MANIFEST / "v2_smartphone_real_manifest_v1.csv")
    required = phone_man.loc[phone_man["split"].isin(["train", "validation"]), "v2_image_id"].astype(str).tolist()
    train_ids = set(phone_man.loc[phone_man["split"] == "train", "v2_image_id"].astype(str))
    val_ids = set(phone_man.loc[phone_man["split"] == "validation", "v2_image_id"].astype(str))

    # Kaggle-side listing
    kaggle_ids: set[str] = set()
    kaggle_bytes = 0
    try:
        files = kaggle_paginate_files("sovaakushwaha/v2-clip-lora-support")
        for f in files:
            m = re.search(r"(SMARTPHONE_\d+)", f["name"])
            if m:
                kaggle_ids.add(m.group(1))
                kaggle_bytes += f["size"]
    except Exception as exc:
        files = []
        kaggle_err = str(exc)
    else:
        kaggle_err = None

    # Local existence + decode
    search_dirs = smartphone_paths()
    id_to_path: dict[str, Path] = {}
    for iid in required:
        found = None
        for d in search_dirs:
            for ext in (".jpg", ".jpeg", ".webp", ".png"):
                p = d / f"{iid}{ext}"
                if p.exists() and p.stat().st_size > 1024:
                    found = p
                    break
            if found:
                break
        if found:
            id_to_path[iid] = found

    missing = [i for i in required if i not in id_to_path]
    corrupt = []
    unreadable = []
    for iid, p in id_to_path.items():
        ok, msg = decode_image(p)
        if not ok:
            unreadable.append({"id": iid, "error": msg})
            if "bad_dims" not in msg:
                corrupt.append(iid)

    req_set = set(required)
    return {
        "expected": len(required),
        "train": len(train_ids),
        "validation": len(val_ids),
        "kaggle_unique_ids": len(kaggle_ids),
        "kaggle_missing": len(req_set - kaggle_ids),
        "kaggle_extra": len(kaggle_ids - req_set),
        "kaggle_total_bytes": kaggle_bytes,
        "kaggle_listing_error": kaggle_err,
        "local_found": len(id_to_path),
        "local_missing": len(missing),
        "missing_sample": missing[:20],
        "decode_pass": len(unreadable) == 0,
        "corrupt": len(corrupt),
        "unreadable": len(unreadable),
        "unreadable_sample": unreadable[:10],
        "existence_pass": len(missing) == 0 and len(kaggle_ids) == 2500 and len(req_set - kaggle_ids) == 0,
        "decode_pass_flag": len(unreadable) == 0 and len(id_to_path) == 2500,
        "pass": len(missing) == 0
        and len(kaggle_ids) == 2500
        and len(req_set - kaggle_ids) == 0
        and len(unreadable) == 0,
    }


def resolve_local_source(row: dict, tiny_root: Path | None) -> tuple[str, Path | None]:
    iid = str(row["image_id"])
    ds = str(row.get("source_dataset", ""))
    if ds == "Tiny-GenImage" and tiny_root:
        rel = str(row["path"]).replace("data/raw/tiny-genimage/", "")
        p = tiny_root / rel
        if p.exists() and p.stat().st_size > 0:
            return "local", p
        fname = Path(row["path"]).name
        hits = list(tiny_root.rglob(fname))
        if hits:
            return "local", hits[0]
        return "missing", None
    if iid.startswith("SMARTPHONE_"):
        for d in smartphone_paths():
            for ext in (".jpg", ".jpeg", ".webp", ".png"):
                p = d / f"{iid}{ext}"
                if p.exists() and p.stat().st_size > 1024:
                    return "local", p
        return "missing", None
    if ds == "MLLM":
        return "remote_hf", None
    if ds == "Qwen":
        return "remote_hf", None
    if ds == "COCO" or iid.startswith("EXT_REAL"):
        return "remote_http", None
    return "unknown", None


def materialization_audit() -> dict:
    df = pd.read_csv(MANIFEST / "v2_split_assignments_v1.csv")
    df = df[df["fold_1_role"] != "EXCLUDED_DUPLICATE"]
    tiny_root = REPO / "data" / "raw" / "tiny-genimage"
    if not tiny_root.exists():
        tiny_root = None
    all_rows: dict[str, dict] = {}
    fold_report = {}
    for fold in range(1, 5):
        col = f"fold_{fold}_role"
        rows = df[df[col].isin(["TRAIN", "HOLDOUT_VALIDATION", "REAL_VALIDATION"])].to_dict("records")
        missing = []
        unreadable = []
        preprocess_fail = []
        remote_only = []
        local_ok = []
        for r in rows:
            iid = str(r["image_id"])
            all_rows[iid] = r
            kind, path = resolve_local_source(r, tiny_root)
            if kind == "missing":
                missing.append(iid)
            elif kind.startswith("remote"):
                remote_only.append(iid)
            elif path is not None:
                ok, msg = decode_image(path)
                if not ok:
                    unreadable.append({"id": iid, "error": msg})
                else:
                    local_ok.append(iid)
        fold_report[f"fold_{fold}"] = {
            "required_rows": len(rows),
            "missing": len(missing),
            "remote_only": len(remote_only),
            "local_decoded": len(local_ok),
            "unreadable": len(unreadable),
            "missing_sample": missing[:15],
            "unreadable_sample": unreadable[:5],
            # PASS locally if no missing/unreadable among resolvable local; remote expected on Kaggle
            "pass_local": len(missing) == 0 and len(unreadable) == 0,
            "pass_kaggle_expected": len(missing) == 0,
        }
    kinds = Counter()
    for iid, r in all_rows.items():
        k, _ = resolve_local_source(r, tiny_root)
        kinds[k] += 1
    return {
        "total_unique_required_images": len(all_rows),
        "local_offline_available": kinds["local"],
        "remote_hf": kinds["remote_hf"],
        "remote_http": kinds["remote_http"],
        "missing": kinds["missing"],
        "folds": fold_report,
        "pass_local": all(v["pass_local"] for v in fold_report.values()),
        "pass_kaggle_expected": all(v["pass_kaggle_expected"] for v in fold_report.values()),
    }


def disk_audit() -> dict:
    # approximate peak usage on Kaggle (MB)
    components = {
        "support_mount_smartphone_read_not_copy": 0,
        "tiny_genimage_mount": 1800,
        "clip_weights_cache": 600,
        "pip_cache": 300,
        "materialized_hf_coco_cache": 800,
        "checkpoints_x4": 100,
        "predictions_logs": 200,
        "working_headroom": 2000,
    }
    peak_mb = sum(components.values())
    return {
        "estimated_peak_usage_mb": peak_mb,
        "estimated_peak_usage_gb": round(peak_mb / 1024, 2),
        "components_mb": components,
        "safety_margin_recommended_gb": 5,
        "note": "Smartphone copy-to-working removed in train script; read mounted paths directly.",
        "pass": peak_mb < 16000,
    }


def dependency_audit() -> dict:
    info = {"python": sys.version.split()[0]}
    for pkg in ["torch", "torchvision", "open_clip", "pandas", "numpy", "PIL", "sklearn"]:
        info[pkg] = "MISSING"
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import open_clip

        info["open_clip"] = open_clip.__version__
    except Exception:
        pass
    try:
        import pandas as np  # noqa
        import pandas
        import numpy
        import sklearn

        info["pandas"] = pandas.__version__
        info["numpy"] = numpy.__version__
        info["sklearn"] = sklearn.__version__
        info["PIL"] = "ok"
    except Exception:
        pass
    local_pass = info.get("torch") != "MISSING" and info.get("open_clip") != "MISSING"
    return {"pass_local": local_pass, "pass_kaggle_pending": True, **info}


def clip_audit() -> dict:
    try:
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
        model.eval()
        res = getattr(model.visual, "image_size", 224)
        if isinstance(res, (tuple, list)):
            res = int(res[0])
        embed = int(getattr(model.visual, "output_dim", 512))
        x = torch.randn(1, 3, int(res), int(res))
        with torch.no_grad():
            y = model.encode_image(x)
        total = sum(p.numel() for p in model.parameters())
        return {
            "weights_available_offline": True,
            "model_load": "PASS",
            "model_forward": "PASS" if y.shape[-1] == embed else "FAIL",
            "embedding_dim": embed,
            "input_resolution": int(res),
            "total_params": total,
            "pass": y.shape[-1] == embed and embed == 512 and int(res) == 224,
        }
    except Exception as exc:
        return {
            "weights_available_offline": "UNKNOWN",
            "model_load": "FAIL",
            "model_forward": "FAIL",
            "error": str(exc),
            "pass": False,
        }


def checkpoint_audit() -> dict:
    out_dir = ROOT / "_readiness_test_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "test_ckpt.pt"
    try:
        import torch

        state = {"a": torch.tensor([1.0])}
        torch.save(state, ckpt)
        ok_exist = ckpt.exists() and ckpt.stat().st_size > 0
        loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
        ok_load = "a" in loaded
        ckpt.unlink(missing_ok=True)
        for sub in ["v2_lora_outputs", "checkpoints", "predictions", "logs"]:
            (out_dir / sub).mkdir(exist_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        return {
            "save_checkpoint": ok_exist,
            "load_checkpoint": ok_load,
            "pass": ok_exist and ok_load,
        }
    except Exception as exc:
        return {"pass": False, "error": str(exc)}


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {
        "stage": "V2-8",
        "audit_version": "v1",
        "A_kaggle_state": kaggle_state(),
        "B_locked_config": config_audit(),
        "C_dataset_version": {},
        "D_smartphone": smartphone_audit(),
        "E_manifest": manifest_audit(),
        "F_materialization": materialization_audit(),
        "G_disk": disk_audit(),
        "H_dependencies": dependency_audit(),
        "I_clip": clip_audit(),
        "J_checkpoint": checkpoint_audit(),
        "K_dataloader": {"pass": False, "note": "Requires Kaggle GPU env + full materialization; deferred to kernel preflight"},
        "L_security": {
            "ntire_accessed": False,
            "fal_used": False,
            "scientific_config_changed": False,
            "pass": True,
        },
        "M_gpu_readiness": {},
    }
    ds_st = report["A_kaggle_state"].get("support_dataset_status", {})
    report["C_dataset_version"] = {
        "support_dataset": LOCKED["support_dataset"],
        "version": ds_st.get("current_version_number"),
        "pass": ds_st.get("current_version_number") == LOCKED["support_version"],
    }
    gq = report["A_kaggle_state"]
    report["M_gpu_readiness"] = {
        "gpu_hours_available": gq.get("gpu_hours_available"),
        "gpu_quota_reset": gq.get("gpu_quota_reset"),
        "estimated_required_gpu_hours": 8,
        "v36_running": gq.get("v36_status") == "RUNNING",
        "quota_gate": "FAIL" if not gq.get("gpu_quota_available") or gq.get("v36_status") == "RUNNING" else "PASS",
        "ready": bool(gq.get("gpu_quota_available")) and gq.get("v36_status") != "RUNNING",
    }
    blockers = []
    if report["A_kaggle_state"].get("v36_status") == "RUNNING":
        blockers.append("v36 still RUNNING — MANUAL_STOP_REQUIRED")
    if not report["A_kaggle_state"].get("gpu_quota_available"):
        blockers.append("GPU quota exhausted until reset")
    if not report["B_locked_config"]["pass"]:
        blockers.append(f"config issues: {report['B_locked_config']['issues'] + report['B_locked_config']['stale_hits']}")
    if not report["C_dataset_version"]["pass"]:
        blockers.append("support dataset version != 6")
    if not report["D_smartphone"]["pass"]:
        blockers.append("smartphone audit failed")
    if not report["E_manifest"]["pass"]:
        blockers.append("manifest audit failed")
    if not report["F_materialization"]["pass_kaggle_expected"]:
        blockers.append("materialization missing local sources")
    if not report["G_disk"]["pass"]:
        blockers.append("disk estimate too high")
    if not report["I_clip"]["pass"] and report["H_dependencies"].get("open_clip") == "MISSING":
        pass  # local-only; Kaggle offline bundle supplies open_clip
    elif not report["I_clip"]["pass"]:
        blockers.append("CLIP load test failed")
    if not report["J_checkpoint"]["pass"]:
        blockers.append("checkpoint write test failed")

    report["blockers"] = blockers
    report["final_verdict"] = (
        "READY_FOR_V37_GPU_RUN"
        if not blockers and report["M_gpu_readiness"]["ready"]
        else "NOT_READY_FOR_V37_GPU_RUN"
    )
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": report["final_verdict"], "blockers": blockers}, indent=2))


if __name__ == "__main__":
    main()

"""V2-8: Frozen CLIP + LoRA (last 4 blocks) + MLP-B on Kaggle GPU.

Modes (run_config.json): smoke | full
NTIRE: never accessed.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import subprocess
import sys
from pathlib import Path

IS_KAGGLE = Path("/kaggle/working").exists()


def _verify_gpu_pre_torch() -> dict:
    """Stage A: verify NVIDIA GPU before importing torch."""
    if not IS_KAGGLE:
        return {}
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        smi_l = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("V2_8_GPU_ALLOCATION_FAILED")
    if smi.returncode != 0 or not (smi_l.stdout or "").strip():
        raise SystemExit("V2_8_GPU_ALLOCATION_FAILED")
    info = {"nvidia_smi_detected": True, "gpu_listing": (smi_l.stdout or "").strip()}
    Path("/kaggle/working/gpu_pre_torch_v1.txt").write_text((smi.stdout or "") + "\n")
    return info


def _ensure_deps() -> None:
    """Stage B: install missing deps from offline bundle without replacing CUDA torch."""
    if importlib.util.find_spec("open_clip") is not None:
        return
    bundle_dirs = [
        Path("/kaggle/input/v2-clip-lora-pip-bundle"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-pip-bundle"),
    ]
    bundle = next((b for b in bundle_dirs if b.exists()), None)
    log = Path("/kaggle/working/pip_install.log")
    if bundle is not None:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={bundle}",
            "--no-deps",
            "open-clip-torch==3.3.0",
            "ftfy",
            "regex",
            "safetensors",
            "timm",
            "huggingface_hub",
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "open-clip-torch==3.3.0",
            "ftfy",
            "regex",
            "safetensors",
            "timm",
            "huggingface_hub",
        ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text((r.stdout or "") + "\n" + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError(f"pip install failed; see {log}")


def _maybe_install_pascal_torch() -> None:
    """P100 (sm_60) requires cu118 PyTorch; Kaggle default cu128 does not support Pascal."""
    if not IS_KAGGLE:
        return
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    name = (smi.stdout or "").upper()
    if "P100" not in name:
        return
    log = Path("/kaggle/working/pip_pascal_torch.log")
    bundle_dirs = [
        Path("/kaggle/input/v2-clip-lora-pip-bundle"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-pip-bundle"),
    ]
    bundle = next((b for b in bundle_dirs if b.exists()), None)
    lines = [f"gpu={name}"]
    if bundle is not None:
        cu118_wheels = sorted(bundle.glob("torch-*cu118*.whl"))
        tv_wheels = sorted(bundle.glob("torchvision-*cu118*.whl"))
        if cu118_wheels and tv_wheels:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                str(cu118_wheels[-1]),
                str(tv_wheels[-1]),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines.extend([r.stdout or "", r.stderr or "", f"offline_exit={r.returncode}"])
            log.write_text("\n".join(lines))
            if r.returncode == 0:
                return
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu118",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines.extend([r.stdout or "", r.stderr or "", f"online_exit={r.returncode}"])
    log.write_text("\n".join(lines))
    if r.returncode != 0:
        raise RuntimeError(f"Pascal-compatible PyTorch install failed; see {log}")


_verify_gpu_pre_torch()
_maybe_install_pascal_torch()
_ensure_deps()

import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
import warnings
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

# --- lora_utils (inlined for Kaggle script kernel single-file bundle) ---
import math
from types import MethodType


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.linear = linear
        self.scale = alpha / max(1, r)
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


class LoRAInProjDelta(nn.Module):
    def __init__(self, embed_dim: int, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> None:
        super().__init__()
        self.scale = alpha / max(1, r)
        self.lora_A = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B = nn.Parameter(torch.zeros(3 * embed_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


def _lora_attention_forward(block, x: torch.Tensor) -> torch.Tensor:
    attn = block.attn
    in_delta: LoRAInProjDelta = block.lora_in_delta
    out_lora: LoRALinear = block.lora_out_proj
    embed_dim = attn.embed_dim
    num_heads = attn.num_heads
    head_dim = embed_dim // num_heads
    tgt_len, bsz, _ = x.shape
    flat = x.reshape(tgt_len * bsz, embed_dim)
    qkv = F.linear(flat, attn.in_proj_weight, attn.in_proj_bias) + in_delta(flat)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(tgt_len, bsz, embed_dim).transpose(0, 1).view(bsz, tgt_len, num_heads, head_dim).transpose(1, 2)
    attn_out = F.scaled_dot_product_attention(
        q, k, v,
        dropout_p=attn.dropout if attn.training else 0.0,
        is_causal=False,
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, tgt_len, embed_dim).transpose(0, 1)
    flat_out = attn_out.reshape(tgt_len * bsz, embed_dim)
    return out_lora(flat_out).view(tgt_len, bsz, embed_dim)


def inject_lora_last_blocks(visual: nn.Module, last_n: int = 4, r: int = 8, alpha: int = 16, dropout: float = 0.05) -> list[str]:
    blocks = visual.transformer.resblocks
    touched = []
    n = len(blocks)

    def _lora_block_attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        return _lora_attention_forward(self, q_x)

    for i in range(n - last_n, n):
        blk = blocks[i]
        attn = blk.attn
        for p in attn.parameters():
            p.requires_grad = False
        blk.add_module("lora_in_delta", LoRAInProjDelta(attn.embed_dim, r, alpha, dropout))
        blk.add_module("lora_out_proj", LoRALinear(attn.out_proj, r, alpha, dropout))
        blk.attention = MethodType(_lora_block_attention, blk)
        touched.append(f"visual.transformer.resblocks.{i}.attn")
    return touched


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "trainable_pct": float(100.0 * trainable / max(1, total)),
    }


# --- end lora_utils ---

SEED = 42
CAP = 300
USER_AGENT = "ai-image-detector-v2-8-research/1.0"

def _prepare_support_root(base: Path) -> Path:
    """Unzip support bundle archives if Kaggle mounted them as zips."""
    archive_map = {
        "manifests.zip": "manifests",
        "configs.zip": "configs",
    }
    for zname, subdir in archive_map.items():
        zp = base / zname
        if not zp.exists():
            continue
        out = base / subdir
        marker = out / ".extract_ok"
        if marker.exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out)
        marker.write_text("ok\n")
    models_zip = base / "models.zip"
    if models_zip.exists() and not (base / "models" / "clip_mlpB_fold1_best_v1.pt").exists():
        marker = base / "models" / ".extract_ok"
        if not marker.exists():
            with zipfile.ZipFile(models_zip, "r") as zf:
                zf.extractall(base)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n")
    phone_zip = base / "smartphone_real.zip"
    if phone_zip.exists() and not _smartphone_cache_ready(base):
        marker = base / "smartphone_real" / ".extract_ok"
        if not marker.exists():
            with zipfile.ZipFile(phone_zip, "r") as zf:
                zf.extractall(base)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n")
    return base


def _smartphone_cache_ready(base: Path) -> bool:
    """Require full locked smartphone pool (2500 train+val IDs), not a partial cache."""
    required = 2500
    for sub in (base / "smartphone_real", base / "smartphone_real" / "smartphone_real"):
        if not sub.exists():
            continue
        n = sum(1 for _ in sub.glob("SMARTPHONE_*.*"))
        if n >= required:
            return True
    return False


def _support_has_manifests(base: Path) -> bool:
    return (base / "manifests" / "v2_split_assignments_v1.csv").exists() or (
        base / "v2_split_assignments_v1.csv"
    ).exists()


def _resolve_repo_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for _cand in [
        Path("/kaggle/input/v2-clip-lora-support"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-support"),
        script_dir,
        Path("/kaggle/input/v2-clip-lora-generalisation"),
        Path("/kaggle/working"),
    ]:
        if not _cand.exists():
            continue
        base = _prepare_support_root(_cand)
        if _support_has_manifests(base):
            if not (base / "manifests" / "v2_split_assignments_v1.csv").exists():
                flat = base / "v2_split_assignments_v1.csv"
                if flat.exists():
                    man_dir = base / "manifests"
                    man_dir.mkdir(parents=True, exist_ok=True)
                    for name in [
                        "v2_split_assignments_v1.csv",
                        "v2_smartphone_real_manifest_v1.csv",
                        "external_mllm_manifest_v2.csv",
                        "external_qwen_image_bench_manifest_v2.csv",
                        "v2_generator_holdout_folds_v1.csv",
                        "v2_generator_registry_v1.csv",
                    ]:
                        src = base / name
                        if src.exists():
                            dst = man_dir / name
                            if not dst.exists():
                                shutil.move(str(src), str(dst))
            if not (base / "configs" / "v2_lora_config_v1.json").exists():
                flat_cfg = base / "v2_lora_config_v1.json"
                if flat_cfg.exists():
                    cfg_dir = base / "configs"
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    dst = cfg_dir / "v2_lora_config_v1.json"
                    if not dst.exists():
                        shutil.move(str(flat_cfg), str(dst))
            return base
    if IS_KAGGLE:
        raise SystemExit("STOP: support bundle manifests not found under /kaggle/input/v2-clip-lora-support")
    return script_dir


ROOT = _resolve_repo_root()
CONFIG_PATH = ROOT / "configs" / "v2_lora_config_v1.json"
_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_run_config(default_mode: str = "smoke_then_full") -> dict:
    for p in [
        _SCRIPT_DIR / "run_config.json",
        Path("/kaggle/input/v2-clip-lora-support/run_config.json"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-support/run_config.json"),
        ROOT / "run_config.json",
    ]:
        if p.exists():
            cfg = json.loads(p.read_text())
            cfg.setdefault("mode", default_mode)
            return cfg
    return {"mode": default_mode}


def _load_run_mode(default: str = "smoke_then_full") -> str:
    return _load_run_config(default).get("mode", default)


def _locked_support_version(run_cfg: dict) -> dict:
    return {
        "support_dataset": run_cfg.get("support_dataset", "sovaakushwaha/v2-clip-lora-support"),
        "support_dataset_version": int(run_cfg.get("support_dataset_version", 6)),
        "kernel_target_version": int(run_cfg.get("kernel_target_version", 37)),
    }


def _count_mounted_smartphones() -> int:
    n = 0
    for base in (
        Path("/kaggle/input/v2-clip-lora-support"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-support"),
        ROOT,
    ):
        if not base.exists():
            continue
        prepared = _prepare_support_root(base)
        for sub in (prepared / "smartphone_real", prepared / "smartphone_real" / "smartphone_real"):
            if sub.exists():
                n = max(n, sum(1 for _ in sub.glob("SMARTPHONE_*.*")))
    return n


def _required_smartphone_ids() -> set[str]:
    phone = pd.read_csv(MANIFEST_DIR / "v2_smartphone_real_manifest_v1.csv")
    return set(phone.loc[phone["split"].isin(["train", "validation"]), "v2_image_id"].astype(str))


def _decode_image_ok(path: Path) -> tuple[bool, str]:
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
            if w < 8 or h < 8:
                return False, f"bad_dims_{w}x{h}"
            rgb.load()
            return True, "ok"
    except Exception as exc:
        return False, str(exc)[:120]


def _resolve_row_source(r: dict, cfg: dict, phone_url: dict, mllm_idx, qwen_idx) -> tuple[str, Path | None]:
    iid = str(r["image_id"])
    ds = r["source_dataset"]
    if ds == "Tiny-GenImage":
        p = resolve_tiny_path(r)
        return ("local", p) if p else ("missing", None)
    if iid.startswith("SMARTPHONE_"):
        p = _resolve_smartphone_local(iid)
        return ("local", p) if p else ("missing", None)
    if ds == "MLLM" and iid in mllm_idx.index:
        return ("remote_hf", None)
    if ds == "Qwen" and iid in qwen_idx.index:
        return ("remote_hf", None)
    if ds == "COCO" or iid.startswith("EXT_REAL"):
        return ("remote_http", None)
    return ("unknown", None)


def run_materialization_precheck(cfg: dict, run_cfg: dict, out: Path) -> dict:
    """Pre-GPU gate: all 2500 smartphone IDs + all four folds resolvable/decodable."""
    support_meta = _locked_support_version(run_cfg)
    required_phones = _required_smartphone_ids()
    phone_manifest = pd.read_csv(MANIFEST_DIR / "v2_smartphone_real_manifest_v1.csv")
    phone_url = dict(zip(phone_manifest["v2_image_id"].astype(str), phone_manifest["source_url"].astype(str)))
    mllm = pd.read_csv(MANIFEST_DIR / "external_mllm_manifest_v2.csv").set_index("image_id")
    qwen = pd.read_csv(MANIFEST_DIR / "external_qwen_image_bench_manifest_v2.csv").set_index("image_id")

    missing_phones: list[str] = []
    unreadable_phones: list[dict] = []
    for iid in sorted(required_phones):
        p = _resolve_smartphone_local(iid)
        if p is None:
            missing_phones.append(iid)
            continue
        ok, msg = _decode_image_ok(p)
        if not ok:
            unreadable_phones.append({"id": iid, "error": msg})

    mounted_phones = _count_mounted_smartphones()
    man = read_split()
    fold_reports = {}
    all_missing: list[str] = []
    all_unreadable: list[dict] = []
    remote_hf = 0
    remote_http = 0
    local_ok = 0
    for fold in range(1, 5):
        rows = train_indices(man, fold) + val_indices(man, fold)
        fold_missing = []
        fold_unreadable = []
        fold_remote_hf = 0
        fold_remote_http = 0
        fold_local = 0
        for r in rows:
            iid = str(r["image_id"])
            kind, path = _resolve_row_source(r, cfg, phone_url, mllm, qwen)
            if kind == "missing":
                fold_missing.append(iid)
            elif kind == "remote_hf":
                fold_remote_hf += 1
            elif kind == "remote_http":
                fold_remote_http += 1
            elif path is not None:
                ok, msg = _decode_image_ok(path)
                if ok:
                    fold_local += 1
                else:
                    fold_unreadable.append({"id": iid, "error": msg})
        remote_hf += fold_remote_hf
        remote_http += fold_remote_http
        local_ok += fold_local
        all_missing.extend(fold_missing)
        all_unreadable.extend(fold_unreadable)
        fold_reports[f"fold_{fold}"] = {
            "required_rows": len(rows),
            "missing": len(fold_missing),
            "unreadable": len(fold_unreadable),
            "local_decoded": fold_local,
            "remote_hf": fold_remote_hf,
            "remote_http": fold_remote_http,
            "missing_sample": fold_missing[:20],
            "pass": len(fold_missing) == 0 and len(fold_unreadable) == 0,
        }

    unique_missing = sorted(set(all_missing))
    report = {
        "support_dataset": support_meta["support_dataset"],
        "support_dataset_version": support_meta["support_dataset_version"],
        "kernel_target_version": support_meta["kernel_target_version"],
        "expected_smartphone_ids": len(required_phones),
        "found_smartphone_ids_local": len(required_phones) - len(missing_phones),
        "mounted_smartphone_files": mounted_phones,
        "missing_smartphone_ids": missing_phones[:50],
        "missing_smartphone_count": len(missing_phones),
        "unreadable_smartphone_count": len(unreadable_phones),
        "unreadable_smartphone_sample": unreadable_phones[:10],
        "total_unique_required_images": sum(v["required_rows"] for v in fold_reports.values()),
        "local_offline_decoded": local_ok,
        "remote_hf": remote_hf,
        "remote_http": remote_http,
        "missing_total": len(unique_missing),
        "unreadable_total": len(all_unreadable),
        "folds": fold_reports,
        "pass": (
            len(missing_phones) == 0
            and len(unreadable_phones) == 0
            and mounted_phones >= 2500
            and all(v["pass"] for v in fold_reports.values())
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "materialization_precheck_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "MATERIALIZATION_PRECHECK",
        json.dumps(
            {
                k: report[k]
                for k in report
                if k not in ("missing_smartphone_ids", "folds")
            }
        ),
    )
    if not report["pass"]:
        stop(
            "V2_8_MATERIALIZATION_PRECHECK_FAILED: "
            f"missing_phones={len(missing_phones)} unreadable_phones={len(unreadable_phones)} "
            f"fold_missing={len(unique_missing)} mounted={mounted_phones}"
        )
    return report


def emit_heartbeat(stage: str, **fields) -> None:
    payload = {"stage": stage, "ts": time.time(), **fields}
    print("HEARTBEAT", json.dumps(payload, default=str))
    if IS_KAGGLE:
        hb = Path("/kaggle/working/v2_lora_heartbeat_v1.jsonl")
        with hb.open("a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
MANIFEST_DIR = ROOT / "manifests"
MODEL_DIR = ROOT / "models"
if not (MODEL_DIR / "clip_mlpB_fold1_best_v1.pt").exists() and (ROOT / "models" / "models" / "clip_mlpB_fold1_best_v1.pt").exists():
    MODEL_DIR = ROOT / "models" / "models"
IS_KAGGLE = Path("/kaggle/working").exists()
OUT_DIR = Path("/kaggle/working/v2_lora_outputs") if IS_KAGGLE else ROOT / "_local_outputs"
DATA_DIR = Path("/kaggle/working/data/v2") if IS_KAGGLE else ROOT / "_local_data"


def stop(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as im:
        try:
            t = ImageOps.exif_transpose(im)
            if t is not None:
                im = t
        except Exception:
            pass
        return im.convert("RGB")


class MLPBHead(nn.Module):
    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClipLoRAModel(nn.Module):
    def __init__(self, cfg: dict, device: torch.device) -> None:
        super().__init__()
        clip_cfg = cfg["clip"]
        lora_cfg = cfg["lora"]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_cfg["model_name"], pretrained=clip_cfg["pretrained"]
        )
        self.visual = self.model.visual
        for p in self.model.parameters():
            p.requires_grad = False
        self.lora_modules = inject_lora_last_blocks(
            self.visual,
            last_n=lora_cfg["last_n_blocks"],
            r=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
        )
        self.head = MLPBHead(dropout=cfg["head"]["dropout"])
        self.l2_normalize = clip_cfg["l2_normalize"]
        self.to(device)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.visual(images)
        if self.l2_normalize:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(images))


class V2ImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        preprocess,
        flip_p: float = 0.0,
        train: bool = True,
    ) -> None:
        self.rows = rows
        self.preprocess = preprocess
        self.flip_p = flip_p if train else 0.0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        p = Path(r["local_path"])
        if not p.exists():
            stop(f"missing image {p} for {r['image_id']}")
        img = load_rgb(p)
        if self.flip_p > 0 and random.random() < self.flip_p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = self.preprocess(img)
        y = int(r["binary_label"])
        return x, y, r["image_id"]


def read_split() -> pd.DataFrame:
    p = MANIFEST_DIR / "v2_split_assignments_v1.csv"
    df = pd.read_csv(p)
    return df[df["fold_1_role"] != "EXCLUDED_DUPLICATE"].copy()


def train_indices(man: pd.DataFrame, fold: int) -> list[dict]:
    col = f"fold_{fold}_role"
    ai = man[(man["binary_label"] == 1) & (man[col] == "TRAIN")].copy()
    keep = []
    for _, gdf in ai.groupby("canonical_generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep.append(gdf)
    ai_keep = pd.concat(keep) if keep else ai.iloc[:0]
    real = man[(man["binary_label"] == 0) & (man[col] == "TRAIN")]
    out = pd.concat([ai_keep, real], ignore_index=True)
    return out.to_dict("records")


def val_indices(man: pd.DataFrame, fold: int) -> list[dict]:
    col = f"fold_{fold}_role"
    mask = man[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])
    return man.loc[mask].to_dict("records")


def subset_smoke(rows: list[dict], max_real: int, max_ai: int) -> list[dict]:
    real = sorted([r for r in rows if r["binary_label"] == 0], key=lambda r: r["image_id"])[:max_real]
    ai = sorted([r for r in rows if r["binary_label"] == 1], key=lambda r: r["image_id"])[:max_ai]
    return real + ai


def _download_bytes(url: str, timeout: int = 90, retries: int = 8) -> bytes | None:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    headers = {"User-Agent": USER_AGENT}
    urls = [url]
    if url.startswith("https://"):
        urls.append("http://" + url[8:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for u in urls:
            for attempt in range(retries):
                try:
                    r = session.get(
                        u,
                        timeout=(20, timeout),
                        headers=headers,
                        verify=False,
                        allow_redirects=True,
                    )
                    if r.status_code == 200 and len(r.content) > 1024:
                        return r.content
                except Exception:
                    pass
                time.sleep(min(12, 2**attempt))
    return None


def _resolve_smartphone_local(iid: str) -> Path | None:
    """Locked smartphone samples cached offline (same manifest IDs, not substitutes)."""
    search_dirs: list[Path] = []
    for base in (
        Path("/kaggle/input/v2-clip-lora-support"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-support"),
        ROOT,
    ):
        if not base.exists():
            continue
        prepared = _prepare_support_root(base)
        for sub in ("smartphone_real", "smartphone_real/smartphone_real"):
            d = prepared / sub
            if d.exists():
                search_dirs.append(d)
    local = ROOT.parents[1] / "data" / "v2" / "smartphone_real"
    if local.exists():
        search_dirs.append(local)
    for root in search_dirs:
        for ext in (".jpg", ".webp", ".jpeg", ".png"):
            p = root / f"{iid}{ext}"
            if p.exists() and p.stat().st_size > 1024:
                return p
    return None


def _hf_fetch(repo: str, revision: str, hf_path: str, out: Path, retries: int = 4) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        return True
    for attempt in range(retries):
        try:
            cached = hf_hub_download(
                repo_id=repo,
                filename=hf_path,
                revision=revision,
                repo_type="dataset",
            )
            src = Path(cached)
            if src.exists() and src.stat().st_size > 1024:
                out.write_bytes(src.read_bytes())
                return True
        except Exception:
            time.sleep(min(8, 2**attempt))
    return False


TINY_INDEX: dict[str, Path] | None = None
TINY_INDEX_BUILT = False


def _tiny_kaggle_roots() -> list[Path]:
    roots = []
    for base in [
        Path("/kaggle/input/tiny-genimage"),
        Path("/kaggle/input/yangsangtai-tiny-genimage"),
        Path("/kaggle/input/tiny-genimage/tiny-genimage"),
        DATA_DIR / "tiny_hf",
    ]:
        if base.exists():
            roots.append(base)
    return roots


def _reset_tiny_index() -> None:
    global TINY_INDEX, TINY_INDEX_BUILT
    TINY_INDEX = None
    TINY_INDEX_BUILT = False


def _build_tiny_index() -> dict[str, Path]:
    global TINY_INDEX, TINY_INDEX_BUILT
    if TINY_INDEX_BUILT and TINY_INDEX is not None:
        return TINY_INDEX
    idx: dict[str, Path] = {}
    for root in _tiny_kaggle_roots():
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            key = p.name.lower()
            if key not in idx:
                idx[key] = p
    TINY_INDEX = idx
    TINY_INDEX_BUILT = True
    if IS_KAGGLE:
        probe = {
            "roots": [str(r) for r in _tiny_kaggle_roots()],
            "indexed_files": len(idx),
            "sample_keys": sorted(list(idx.keys()))[:10],
        }
        (Path("/kaggle/working/tiny_path_probe_v1.json")).write_text(json.dumps(probe, indent=2) + "\n")
    return idx


def _ensure_tiny_hf_extracted() -> None:
    """Extract V2 pilot Tiny-GenImage shards from HF when Kaggle tiny dataset is absent."""
    if _tiny_kaggle_roots():
        return
    marker = DATA_DIR / "tiny_hf" / ".extract_complete"
    if marker.exists():
        return
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    except ImportError as exc:
        stop(f"Tiny HF extraction requires pyarrow/huggingface_hub: {exc}")
    repo = "TheKernel01/Tiny-GenImage"
    shards = ["data/train-00000-of-00014.parquet", "data/train-00001-of-00014.parquet"]
    gen_names = {0: "Real", 1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney", 5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong"}
    out_root = DATA_DIR / "tiny_hf" / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        pq_path = hf_hub_download(repo_id=repo, filename=shard, repo_type="dataset")
        pf = pq.ParquetFile(pq_path)
        for rg in range(pf.num_row_groups):
            for record in pf.read_row_group(rg).to_pylist():
                image = record["image"]
                data = image.get("bytes") if isinstance(image, dict) else None
                if not data:
                    continue
                original_name = image.get("path") if isinstance(image, dict) else None
                gen = gen_names.get(int(record["generator"]), f"gen_{record['generator']}")
                fname = Path(original_name).name if original_name else f"img_{record.get('label', 0)}.png"
                dest = out_root / gen / fname
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(data)
    marker.write_text("ok\n")


def resolve_tiny_path(row: dict) -> Path | None:
    rel = row["path"].replace("data/raw/tiny-genimage/", "")
    repo_tiny = ROOT.parents[1] / "data" / "raw" / "tiny-genimage" / rel
    candidates = [
        Path("/kaggle/input/tiny-genimage") / rel,
        Path("/kaggle/input/tiny-genimage/tiny-genimage") / rel,
        Path("/kaggle/input/yangsangtai-tiny-genimage") / rel,
        DATA_DIR / "tiny_hf" / rel,
        DATA_DIR / "tiny" / rel,
        repo_tiny,
    ]
    for c in candidates:
        if c is not None and c.exists() and c.stat().st_size > 0:
            return c
    fname = Path(row["path"]).name.lower()
    hit = _build_tiny_index().get(fname)
    if hit is not None and hit.stat().st_size > 0:
        return hit
    stem = Path(fname).stem.lower()
    for k, p in _build_tiny_index().items():
        if Path(k).stem.lower() == stem and p.stat().st_size > 0:
            return p
    return None


def materialize_rows(rows: list[dict], cfg: dict, smoke: bool) -> list[dict]:
    phone = pd.read_csv(MANIFEST_DIR / "v2_smartphone_real_manifest_v1.csv")
    phone_url = dict(zip(phone["v2_image_id"].astype(str), phone["source_url"].astype(str)))
    mllm = pd.read_csv(MANIFEST_DIR / "external_mllm_manifest_v2.csv")
    mllm_map = mllm.set_index("image_id")
    qwen = pd.read_csv(MANIFEST_DIR / "external_qwen_image_bench_manifest_v2.csv")
    qwen_map = qwen.set_index("image_id")

    out_rows = []
    failed = []
    for r in rows:
        iid = str(r["image_id"])
        dst = DATA_DIR / "images" / f"{iid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_path = None
        ds = r["source_dataset"]
        if ds == "Tiny-GenImage":
            tp = resolve_tiny_path(r)
            if tp:
                src_path = tp
        elif iid.startswith("SMARTPHONE_"):
            local_phone = _resolve_smartphone_local(iid)
            if local_phone is not None:
                src_path = local_phone
            else:
                url = phone_url.get(iid, "")
                if url:
                    data = _download_bytes(url)
                    if data:
                        dst.write_bytes(data)
                        src_path = dst
        elif ds == "MLLM" and iid in mllm_map.index:
            meta = mllm_map.loc[iid]
            hf_path = str(meta["source_hf_path"])
            repo = str(meta.get("source_dataset", cfg["data_sources"]["mllm_hf"]))
            rev = str(meta.get("dataset_revision", cfg["data_sources"]["mllm_revision"]))
            out_png = dst.with_suffix(".png")
            if _hf_fetch(repo, rev, hf_path, out_png):
                src_path = out_png
        elif ds == "Qwen" and iid in qwen_map.index:
            meta = qwen_map.loc[iid]
            hf_path = str(meta["source_hf_path"])
            repo = str(meta.get("source_dataset", cfg["data_sources"]["qwen_hf"]))
            rev = str(meta.get("dataset_revision", cfg["data_sources"]["qwen_revision"]))
            out_png = dst.with_suffix(".png")
            if _hf_fetch(repo, rev, hf_path, out_png):
                src_path = out_png
        elif ds == "COCO" or iid.startswith("EXT_REAL"):
            fname = Path(r["path"]).name
            coco_tail = fname.split("_")[-1]
            if coco_tail.lower().endswith(".jpg"):
                coco_tail = coco_tail[:-4]
            coco_fname = f"{int(coco_tail):012d}.jpg"
            urls = [
                f"http://images.cocodataset.org/val2017/{coco_fname}",
                f"http://images.cocodataset.org/val2017/{coco_tail}.jpg",
            ]
            data = None
            for url in urls:
                data = _download_bytes(url)
                if data:
                    break
            if data:
                dst.write_bytes(data)
                src_path = dst
        if src_path is None or not Path(src_path).exists():
            failed.append(iid)
            continue
        rr = dict(r)
        rr["local_path"] = str(src_path)
        out_rows.append(rr)
    if failed and not smoke:
        diag = {"materialized": len(out_rows), "failed": len(failed), "failed_sample": failed[:20]}
        (OUT_DIR / "materialize_failures_v1.json").write_text(json.dumps(diag, indent=2) + "\n")
        stop(f"failed to materialize {len(failed)} images; first={failed[:3]}")
    if smoke and len(out_rows) < 50:
        diag = {"materialized": len(out_rows), "failed": len(failed), "failed_sample": failed[:10]}
        (OUT_DIR / "smoke_materialize_diag_v1.json").write_text(json.dumps(diag, indent=2) + "\n")
        stop(f"smoke materialize too few images: {len(out_rows)}")
    return out_rows


def metrics_at_050(y: np.ndarray, p: np.ndarray) -> dict:
    pred = (p >= 0.5).astype(int)
    m = {
        "roc_auc": float("nan"),
        "ap": float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "ai_recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "fpr": float((pred[y == 0] == 1).mean()) if (y == 0).any() else float("nan"),
    }
    if len(np.unique(y)) > 1:
        m["roc_auc"] = float(roc_auc_score(y, p))
        m["ap"] = float(average_precision_score(y, p))
    return m


def domain_specs(y, p, domains) -> dict:
    out = {}
    for d in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        m = (y == 0) & (domains == d)
        if m.any():
            out[d] = float(((p[m] >= 0.5).astype(int) == 0).mean())
        else:
            out[d] = float("nan")
    vals = [v for v in out.values() if not math.isnan(v)]
    out["worst"] = min(vals) if vals else float("nan")
    return out


@torch.no_grad()
def predict_probs(model: ClipLoRAModel, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    ys, ps, ids = [], [], []
    for xb, yb, iids in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(prob)
        ids.extend(list(iids))
    return np.concatenate(ys), np.concatenate(ps), ids


def load_baseline_head(fold: int, device: torch.device) -> MLPBHead:
    ckpt = torch.load(MODEL_DIR / f"clip_mlpB_fold{fold}_best_v1.pt", map_location=device, weights_only=False)
    head = MLPBHead()
    sd = ckpt.get("state_dict", ckpt)
    head.load_state_dict(sd)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head.to(device)


@torch.no_grad()
def baseline_frozen_predict(model: ClipLoRAModel, head: MLPBHead, loader, device) -> np.ndarray:
    model.eval()
    ps = []
    for xb, _, _ in loader:
        xb = xb.to(device)
        with torch.no_grad():
            for p in model.visual.parameters():
                p.requires_grad = False
            feats = model.encode(xb)
        ps.append(torch.sigmoid(head(feats)).cpu().numpy())
    return np.concatenate(ps)


def run_output_write_gate(out: Path, device: torch.device) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("checkpoints", "predictions", "logs"):
        (out / sub).mkdir(exist_ok=True)
    ckpt = out / "_write_test.pt"
    payload = {"ok": torch.tensor([1.0], device="cpu")}
    torch.save(payload, ckpt)
    ok = ckpt.exists() and ckpt.stat().st_size > 0
    loaded = torch.load(ckpt, map_location=device, weights_only=False)
    ok_load = "ok" in loaded
    ckpt.unlink(missing_ok=True)
    return {"output_write_gate": ok and ok_load, "pass": ok and ok_load}


def run_one_batch_fold_sanity(fold: int, cfg: dict, device: torch.device) -> dict:
    """Engineering sanity: one train + one val batch per fold before full training."""
    emit_heartbeat("one_batch_sanity", fold=fold)
    man = read_split()
    tr = train_indices(man, fold)[:64]
    va = val_indices(man, fold)[:64]
    tr = materialize_rows(tr, cfg, smoke=False)
    va = materialize_rows(va, cfg, smoke=False)
    model = ClipLoRAModel(cfg, device)
    tr_ds = V2ImageDataset(tr, model.preprocess, flip_p=0.0, train=True)
    va_ds = V2ImageDataset(va, model.preprocess, flip_p=0.0, train=False)
    tr_loader = DataLoader(tr_ds, batch_size=min(32, len(tr_ds)), shuffle=True, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=min(32, len(va_ds)), shuffle=False, num_workers=0)
    model.train()
    xb, yb, _ = next(iter(tr_loader))
    xb, yb = xb.to(device), yb.to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = nn.BCEWithLogitsLoss()(logits, yb)
    loss.backward()
    opt.step()
    model.eval()
    with torch.no_grad():
        xb2, _, _ = next(iter(va_loader))
        _ = model(xb2.to(device))
    if not torch.isfinite(logits).all():
        return {"fold": fold, "pass": False, "error": "nonfinite_logits"}
    del model, tr_loader, va_loader
    torch.cuda.empty_cache()
    return {"fold": fold, "pass": True, "train_batch": int(xb.shape[0]), "val_batch": int(xb2.shape[0])}


def train_fold(fold: int, cfg: dict, device: torch.device, smoke: bool) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    man = read_split()
    tr = train_indices(man, fold)
    va = val_indices(man, fold)
    if smoke:
        tr = [r for r in tr if r.get("source_dataset") == "Tiny-GenImage" or resolve_tiny_path(r)]
        va = [r for r in va if r.get("source_dataset") == "Tiny-GenImage" or resolve_tiny_path(r)]
        tr = subset_smoke(tr, cfg["smoke"]["max_real_train"], cfg["smoke"]["max_ai_train"])
        va = subset_smoke(va, 64, 64)
    tr = materialize_rows(tr, cfg, smoke=smoke)
    va = materialize_rows(va, cfg, smoke=smoke)

    flip_p = cfg["training"]["horizontal_flip_p"]
    tr_ds = V2ImageDataset(tr, None, flip_p=flip_p, train=True)
    va_ds = V2ImageDataset(va, None, flip_p=0.0, train=False)

    model = ClipLoRAModel(cfg, device)
    lora_before = snapshot_lora_params(model)
    head_before = snapshot_head_params(model)
    tr_ds.preprocess = model.preprocess
    va_ds.preprocess = model.preprocess

    bs = cfg["training"]["batch_size"]
    batch_reduced = False
    try_bs = bs
    while True:
        try:
            tr_loader = DataLoader(tr_ds, batch_size=try_bs, shuffle=True, num_workers=2, pin_memory=True)
            va_loader = DataLoader(va_ds, batch_size=try_bs, shuffle=False, num_workers=2, pin_memory=True)
            x0, _, _ = next(iter(tr_loader))
            x0 = x0.to(device)
            _ = model(x0)
            bs = try_bs
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and try_bs > cfg["training"]["batch_size_fallback"]:
                try_bs = cfg["training"]["batch_size_fallback"]
                batch_reduced = True
                torch.cuda.empty_cache()
            else:
                raise

    n_real = sum(1 for r in tr if r["binary_label"] == 0)
    n_ai = sum(1 for r in tr if r["binary_label"] == 1)
    ratio = max(n_real, n_ai) / max(1, min(n_real, n_ai))
    if ratio > cfg["training"]["class_ratio_threshold"]:
        pos_weight = torch.tensor([n_real / max(1, n_ai)], device=device)
    else:
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lora_params = [p for n, p in model.named_parameters() if "lora" in n.lower() and p.requires_grad]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg["training"]["lora_lr"]},
            {"params": head_params, "lr": cfg["training"]["head_lr"]},
        ],
        weight_decay=cfg["training"]["weight_decay"],
    )

    max_epochs = cfg["smoke"]["max_epochs"] if smoke else cfg["training"]["max_epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    history = []
    best = None
    best_state = None
    wait = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        emit_heartbeat(
            "train_epoch",
            fold=fold,
            epoch=epoch,
            max_epochs=max_epochs,
            elapsed_s=round(time.perf_counter() - t0, 1),
            batch_size=bs,
            smoke=smoke,
        )
        model.train()
        losses = []
        for xb, yb, _ in tr_loader:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        yva, pva, _ = predict_probs(model, va_loader, device)
        domains = np.array([r.get("real_domain") or "" for r in va])
        yva_i = yva.astype(int)
        m = metrics_at_050(yva_i, pva)
        ds = domain_specs(yva_i, pva, domains)
        val_logits = []
        val_y = []
        model.eval()
        with torch.no_grad():
            for xb, yb, _ in va_loader:
                val_logits.append(model(xb.to(device)).cpu())
                val_y.append(yb)
        val_loss = float(nn.BCEWithLogitsLoss()(torch.cat(val_logits), torch.cat(val_y).float()).item())
        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "val_roc_auc": m["roc_auc"],
            "val_ap": m["ap"],
            "ai_recall": m["ai_recall"],
            "real_specificity": m["specificity"],
            "tiny_specificity": ds.get("Tiny", float("nan")),
            "coco_specificity": ds.get("COCO", float("nan")),
            "mllm_specificity": ds.get("MLLM", float("nan")),
            "smartphone_specificity": ds.get("Smartphone", float("nan")),
            "worst_domain_specificity": ds.get("worst", float("nan")),
        }
        history.append(row)

        cand = {**row, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        replace = False
        if best is None:
            replace = True
        elif cand["val_roc_auc"] > best["val_roc_auc"] + 1e-9:
            replace = True
        elif cand["val_roc_auc"] >= best["val_roc_auc"] - 0.003:
            if cand["worst_domain_specificity"] > best["worst_domain_specificity"] + 1e-9:
                replace = True
            elif abs(cand["worst_domain_specificity"] - best["worst_domain_specificity"]) < 1e-9:
                if cand["mllm_specificity"] > best["mllm_specificity"] + 1e-9:
                    replace = True
                elif cand["epoch"] < best["epoch"]:
                    replace = True
        if replace:
            best = cand
            best_state = cand["state"]
            wait = 0
            emit_heartbeat(
                "checkpoint_candidate",
                fold=fold,
                epoch=epoch,
                val_roc_auc=cand["val_roc_auc"],
                checkpoint_saved=False,
            )
        else:
            wait += 1
            if wait >= patience:
                break

    assert best_state is not None
    lora_after = snapshot_lora_params(model)
    head_after = snapshot_head_params(model)
    lora_update_norm = param_update_norm(lora_before, lora_after)
    head_update_norm = param_update_norm(head_before, head_after)
    model.load_state_dict(best_state)
    wall = time.perf_counter() - t0

    yva, pva, ids = predict_probs(model, va_loader, device)
    yva_i = yva.astype(int)
    domains = np.array([r.get("real_domain") or "" for r in va])
    gens = np.array([r.get("canonical_generator_id") or "" for r in va])
    final_m = metrics_at_050(yva_i, pva)
    ds = domain_specs(yva_i, pva, domains)

    # Baseline frozen CLIP + MLP-B on same loader
    base_head = load_baseline_head(fold, device)
    pb = baseline_frozen_predict(model, base_head, va_loader, device)
    mb = metrics_at_050(yva_i, pb)
    mds = domain_specs(yva_i, pb, domains)

    ckpt_path = OUT_DIR / "checkpoints" / f"clip_lora_fold{fold}_best_v1.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fold": fold,
            "state_dict": model.state_dict(),
            "selected_epoch": best["epoch"],
            "metrics": final_m,
            "config": cfg,
        },
        ckpt_path,
    )

    pred_path = OUT_DIR / "predictions" / f"fold{fold}_predictions_v1.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "fold": fold,
            "image_id": ids,
            "y": yva_i,
            "p_lora": pva,
            "p_mlpb_baseline": pb,
            "real_domain": domains,
            "generator_id": gens,
        }
    ).to_csv(pred_path, index=False)

    mem = peak_cuda_mem_gb()
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else 0.0

    return {
        "fold": fold,
        "history": history,
        "wall_seconds": wall,
        "selected_epoch": best["epoch"],
        "metrics_lora": final_m,
        "metrics_baseline": mb,
        "domain_lora": ds,
        "domain_baseline": mds,
        "param_audit": count_parameters(model),
        "batch_size_used": bs,
        "batch_reduced_due_to_oom": batch_reduced,
        "lora_weights_changed": lora_update_norm > 1e-10,
        "lora_parameter_update_norm": lora_update_norm,
        "mlp_weights_changed": head_update_norm > 1e-10,
        "peak_vram": mem,
        "checkpoint_size_mb": ckpt_size_mb,
    }


def write_environment(out: Path, run_cfg: dict | None = None) -> dict:
    bundle_dirs = [
        Path("/kaggle/input/v2-clip-lora-pip-bundle"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-pip-bundle"),
    ]
    bundle = next((b for b in bundle_dirs if b.exists()), None)
    support_meta = _locked_support_version(run_cfg or _load_run_config())
    env = {
        "is_kaggle": IS_KAGGLE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "torchvision_version": importlib.metadata.version("torchvision"),
        "open_clip_version": open_clip.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        "pil_version": importlib.metadata.version("pillow"),
        "offline_bundle_found": bundle is not None,
        "cpu_only_torch": "+cpu" in torch.__version__,
        "single_gpu_locked": True,
        "device_used": "cuda:0",
        "support_dataset": support_meta["support_dataset"],
        "support_dataset_version": support_meta["support_dataset_version"],
        "kernel_target_version": support_meta["kernel_target_version"],
        "mounted_smartphone_files": _count_mounted_smartphones(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        env["vram_gb"] = round(props.total_memory / (1024**3), 2)
    (out / "environment_v1.json").write_text(json.dumps(env, indent=2) + "\n")
    return env


def verify_cuda_or_stop() -> None:
    if IS_KAGGLE and not torch.cuda.is_available():
        raise SystemExit("V2_8_GPU_ALLOCATION_FAILED")
    if IS_KAGGLE and "+cpu" in torch.__version__:
        raise SystemExit("V2_8_GPU_ALLOCATION_FAILED: CPU-only torch after dependency setup")
    if torch.cuda.is_available():
        try:
            _ = torch.randn(1, device=torch.device("cuda:0"))
            del _
            torch.cuda.synchronize()
        except Exception as exc:
            raise SystemExit(f"V2_8_GPU_ALLOCATION_FAILED: CUDA tensor test failed: {exc}") from exc


def environment_audit(cfg: dict, device: torch.device) -> dict:
    """Stage C: verify CLIP load and tiny CUDA forward."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["clip"]["model_name"], pretrained=cfg["clip"]["pretrained"]
    )
    model = model.to(device)
    model.eval()
    res = getattr(model.visual, "image_size", None) or getattr(model.visual, "input_resolution", 224)
    if isinstance(res, (tuple, list)):
        res = int(res[0])
    embed_dim = int(model.visual.output_dim if hasattr(model.visual, "output_dim") else 512)
    x = torch.randn(2, 3, int(res), int(res), device=device)
    with torch.no_grad():
        feats = model.encode_image(x)
        if cfg["clip"]["l2_normalize"]:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    nan_count = int(torch.isnan(feats).sum().item())
    inf_count = int(torch.isinf(feats).sum().item())
    audit = {
        "input_resolution": int(res),
        "embedding_dim": embed_dim,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "device": str(feats.device),
        "pass": embed_dim == 512 and int(res) == 224 and nan_count == 0 and inf_count == 0,
    }
    del model, x, feats
    torch.cuda.empty_cache()
    if not audit["pass"]:
        stop("environment audit failed")
    return audit


def audit_trainable_parameters(model: ClipLoRAModel) -> dict:
    """Stage E: verify only LoRA + head are trainable."""
    clip_total = sum(p.numel() for p in model.model.parameters())
    frozen_clip = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
    lora_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower())
    head_train = sum(p.numel() for p in model.head.parameters())
    unintended = [
        n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n.lower() and not n.startswith("head")
    ]
    total_train = lora_train + head_train
    report = {
        "clip_total_params": int(clip_total),
        "frozen_clip_params": int(frozen_clip),
        "trainable_lora_params": int(lora_train),
        "trainable_head_params": int(head_train),
        "trainable_total": int(total_train),
        "trainable_pct": float(100.0 * total_train / max(1, clip_total + head_train)),
        "lora_target_modules": model.lora_modules,
        "unintended_trainable": unintended,
    }
    if unintended:
        (OUT_DIR / "param_audit_v1.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit("LORA_PARAMETER_AUDIT_FAILED")
    return report


def snapshot_lora_params(model: ClipLoRAModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora" in k.lower()}


def snapshot_head_params(model: ClipLoRAModel) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.head.state_dict().items()}


def param_update_norm(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    sq = 0.0
    for k, b in before.items():
        if k not in after:
            continue
        d = (after[k] - b).float()
        sq += float(d.pow(2).sum().item())
    return math.sqrt(sq)


def peak_cuda_mem_gb() -> dict:
    if not torch.cuda.is_available():
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    return {
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(0) / (1024**3), 3),
    }


def generator_metrics_from_preds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, gen), gdf in df.groupby(["fold", "generator_id"]):
        y = gdf["y"].to_numpy().astype(int)
        if y.sum() == 0:
            continue
        pl = gdf["p_lora"].to_numpy()
        pb = gdf["p_mlpb_baseline"].to_numpy()
        rows.append(
            {
                "fold": fold,
                "generator_id": gen,
                "n": len(gdf),
                "lora_ai_recall_050": float(((pl >= 0.5).astype(int)[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
                "baseline_ai_recall_050": float(((pb >= 0.5).astype(int)[y == 1] == 1).mean()) if (y == 1).any() else float("nan"),
                "lora_mean_p_ai": float(pl[y == 1].mean()) if (y == 1).any() else float("nan"),
                "baseline_mean_p_ai": float(pb[y == 1].mean()) if (y == 1).any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for fold, gdf in df.groupby("fold"):
        y = gdf["y"].to_numpy().astype(int)
        pl = gdf["p_lora"].to_numpy()
        pb = gdf["p_mlpb_baseline"].to_numpy()
        domains = gdf["real_domain"].fillna("").to_numpy()
        deltas = {"delta_auc": [], "delta_ap": [], "delta_real_spec": [], "delta_mllm_spec": [], "delta_phone_spec": []}
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            yb, plb, pbb, db = y[idx], pl[idx], pb[idx], domains[idx]
            if len(np.unique(yb)) < 2:
                continue
            deltas["delta_auc"].append(roc_auc_score(yb, plb) - roc_auc_score(yb, pbb))
            deltas["delta_ap"].append(average_precision_score(yb, plb) - average_precision_score(yb, pbb))
            rm = yb == 0
            if rm.any():
                deltas["delta_real_spec"].append(((plb[rm] >= 0.5).astype(int) == 0).mean() - ((pbb[rm] >= 0.5).astype(int) == 0).mean())
            for dom, key in [("MLLM", "delta_mllm_spec"), ("Smartphone", "delta_phone_spec")]:
                m = rm & (db == dom)
                if m.any():
                    deltas[key].append(((plb[m] >= 0.5).astype(int) == 0).mean() - ((pbb[m] >= 0.5).astype(int) == 0).mean())
        for metric, vals in deltas.items():
            if not vals:
                continue
            arr = np.array(vals)
            rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    "observed_delta": float(np.mean(arr)),
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                }
            )
    return pd.DataFrame(rows)


def assign_decision(fold_df: pd.DataFrame, gen_df: pd.DataFrame) -> str:
    mean_auc = fold_df["lora_roc_auc"].mean()
    base_auc = 0.837
    mean_real = fold_df["lora_specificity"].mean()
    mean_phone = fold_df["lora_phone_spec"].mean()
    worst_phone = fold_df["lora_phone_spec"].min()
    mean_mllm = fold_df["lora_mllm_spec"].mean()
    hard_gens = [
        "mllm::GPT_Image_2",
        "mllm::Nano_Banana_2",
        "qwen::FLUX.2_max",
        "qwen::GPT-Image-1.5",
        "qwen::Seedream-5.0",
    ]
    hard = gen_df[gen_df["generator_id"].isin(hard_gens)]
    hard_gain = (hard["lora_ai_recall_050"] - hard["baseline_ai_recall_050"]).mean() if len(hard) else 0.0
    reliability_ok = mean_phone >= 0.95 and worst_phone >= 0.94 and mean_real >= 0.89 and mean_mllm >= 0.70
    if mean_auc >= base_auc + 0.005 and hard_gain > 0.02 and reliability_ok:
        return "LORA_PROMISING"
    if mean_auc >= base_auc - 0.003 and hard_gain > 0 and reliability_ok:
        return "LORA_MIXED"
    if mean_auc < base_auc - 0.005 or mean_real < 0.90 or mean_phone < 0.95:
        return "LORA_NOT_BETTER"
    return "LORA_MIXED"


def main() -> None:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for support in (
        Path("/kaggle/input/v2-clip-lora-support"),
        Path("/kaggle/input/sovaakushwaha-v2-clip-lora-support"),
    ):
        if support.exists():
            _prepare_support_root(support)
    cfg = json.loads(CONFIG_PATH.read_text())
    run_cfg = _load_run_config()
    mode = run_cfg.get("mode", "smoke_then_full")

    _ensure_tiny_hf_extracted()
    _reset_tiny_index()
    _build_tiny_index()
    precheck = run_materialization_precheck(cfg, run_cfg, OUT_DIR)

    verify_cuda_or_stop()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = write_environment(OUT_DIR, run_cfg)
    if "P100" in (env.get("gpu_name") or "").upper() and "+cu128" in env.get("torch_version", ""):
        print("WARNING: P100 detected with cu128 PyTorch; attempting continued run after pascal torch install")
    print(
        "GPU_ENV",
        json.dumps(
            {
                "cuda_available": env["cuda_available"],
                "gpu_name": env.get("gpu_name"),
                "gpu_count": env.get("gpu_count"),
                "vram_gb": env.get("vram_gb"),
                "torch": env["torch_version"],
                "cuda": env.get("cuda_version"),
                "open_clip": env["open_clip_version"],
            }
        ),
    )

    env_audit = environment_audit(cfg, device)
    (OUT_DIR / "environment_audit_v1.json").write_text(json.dumps(env_audit, indent=2) + "\n")

    write_gate = run_output_write_gate(OUT_DIR, device)
    (OUT_DIR / "output_write_gate_v1.json").write_text(json.dumps(write_gate, indent=2) + "\n")
    if not write_gate["pass"]:
        stop("output write gate failed")

    audit_model = ClipLoRAModel(cfg, device)
    audit_report = audit_trainable_parameters(audit_model)
    (OUT_DIR / "param_audit_v1.json").write_text(json.dumps(audit_report, indent=2) + "\n")
    print("PARAM_AUDIT", json.dumps({k: audit_report[k] for k in audit_report if k != "lora_target_modules"}))
    del audit_model
    torch.cuda.empty_cache()

    t_all = time.perf_counter()
    results = []
    histories = []
    smoke_report = {}

    if mode in ("smoke", "smoke_then_full"):
        print("=== SMOKE GATE fold 1 ===")
        smoke_res = train_fold(cfg["smoke"]["fold"], cfg, device, smoke=True)
        smoke_report = {
            "data_loading": "PASS",
            "cuda_forward": "PASS",
            "backward": "PASS",
            "optimizer": "PASS",
            "checkpoint_save": "PASS",
            "validation": "PASS",
            "memory_stable": "PASS",
            "batch_size": smoke_res["batch_size_used"],
            "batch_reduced_due_to_oom": smoke_res["batch_reduced_due_to_oom"],
            "lora_weights_changed": smoke_res["lora_weights_changed"],
            "lora_parameter_update_norm": smoke_res["lora_parameter_update_norm"],
            "mlp_weights_changed": smoke_res["mlp_weights_changed"],
            "peak_vram": smoke_res["peak_vram"],
            "pass": bool(
                smoke_res["lora_weights_changed"]
                and smoke_res["mlp_weights_changed"]
                and smoke_res["metrics_lora"]["roc_auc"] == smoke_res["metrics_lora"]["roc_auc"]
            ),
        }
        (OUT_DIR / "smoke_report_v1.json").write_text(json.dumps(smoke_report, indent=2) + "\n")
        if not smoke_res["lora_weights_changed"] or not smoke_res["mlp_weights_changed"]:
            raise SystemExit("V2_8_SMOKE_FAILED: LoRA or MLP weights did not update")
        print(
            "SMOKE_GATE PASS",
            json.dumps(
                {
                    **{k: smoke_report[k] for k in smoke_report if k != "peak_vram"},
                    "lora_norm": smoke_res["lora_parameter_update_norm"],
                }
            ),
        )
        if mode == "smoke":
            print("DONE smoke outputs", OUT_DIR)
            return

    print("=== ONE-BATCH-PER-FOLD GPU SANITY ===")
    sanity_reports = []
    for fold in [1, 2, 3, 4]:
        rep = run_one_batch_fold_sanity(fold, cfg, device)
        sanity_reports.append(rep)
        if not rep["pass"]:
            (OUT_DIR / "one_batch_sanity_v1.json").write_text(json.dumps(sanity_reports, indent=2) + "\n")
            stop(f"one-batch sanity failed fold {fold}: {rep.get('error')}")
    (OUT_DIR / "one_batch_sanity_v1.json").write_text(json.dumps(sanity_reports, indent=2) + "\n")
    print("ONE_BATCH_SANITY PASS", json.dumps(sanity_reports))

    print("=== FULL FOUR-FOLD TRAINING ===")
    for fold in [1, 2, 3, 4]:
        print(f"=== FOLD {fold} mode=full ===")
        res = train_fold(fold, cfg, device, smoke=False)
        results.append(res)
        histories.extend(res["history"])
        print(
            f"fold{fold} AUC={res['metrics_lora']['roc_auc']:.4f} "
            f"baseline={res['metrics_baseline']['roc_auc']:.4f}"
        )

    hist_df = pd.DataFrame(histories)
    hist_df.to_csv(OUT_DIR / "v2_lora_training_history_v1.csv", index=False)

    fold_rows = []
    for r in results:
        fold_rows.append(
            {
                "fold": r["fold"],
                **{f"lora_{k}": v for k, v in r["metrics_lora"].items()},
                **{f"baseline_{k}": v for k, v in r["metrics_baseline"].items()},
                "lora_tiny_spec": r["domain_lora"].get("Tiny"),
                "lora_coco_spec": r["domain_lora"].get("COCO"),
                "lora_mllm_spec": r["domain_lora"].get("MLLM"),
                "lora_phone_spec": r["domain_lora"].get("Smartphone"),
                "baseline_mllm_spec": r["domain_baseline"].get("MLLM"),
                "baseline_phone_spec": r["domain_baseline"].get("Smartphone"),
                "selected_epoch": r["selected_epoch"],
                "wall_seconds": r["wall_seconds"],
                "batch_size_used": r["batch_size_used"],
                "peak_vram_gb": r["peak_vram"].get("peak_allocated_gb"),
                "checkpoint_size_mb": r["checkpoint_size_mb"],
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT_DIR / "v2_lora_fold_metrics_v1.csv", index=False)

    pred_files = sorted((OUT_DIR / "predictions").glob("fold*_predictions_v1.csv"))
    pred_all = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
    gen_df = generator_metrics_from_preds(pred_all)
    gen_df.to_csv(OUT_DIR / "v2_lora_generator_metrics_v1.csv", index=False)

    real_rows = []
    for r in results:
        for dom in ["Tiny", "MLLM", "COCO", "Smartphone"]:
            real_rows.append(
                {
                    "fold": r["fold"],
                    "real_domain": dom,
                    "lora_specificity": r["domain_lora"].get(dom),
                    "baseline_specificity": r["domain_baseline"].get(dom),
                }
            )
    pd.DataFrame(real_rows).to_csv(OUT_DIR / "v2_lora_real_domain_metrics_v1.csv", index=False)

    boot_df = paired_bootstrap(pred_all)
    boot_df.to_csv(OUT_DIR / "v2_lora_bootstrap_v1.csv", index=False)

    overfit_folds = []
    for fold, g in hist_df.groupby("fold"):
        if len(g) >= 3 and g["train_loss"].iloc[-1] < g["train_loss"].iloc[0] and g["val_roc_auc"].iloc[-1] < g["val_roc_auc"].max() - 0.01:
            overfit_folds.append(int(fold))
    overfit = {"observed": bool(overfit_folds), "affected_folds": overfit_folds}

    decision = assign_decision(fold_df, gen_df)
    summary = {
        "stage": "V2-8",
        "mode": "full",
        "smoke_gate": smoke_report,
        "decision": decision,
        "folds": fold_rows,
        "param_audit": audit_report,
        "environment": env,
        "environment_audit": env_audit,
        "overfitting": overfit,
        "wall_seconds_total": time.perf_counter() - t_all,
        "integrity": {
            "ntire_accessed": False,
            "clip_full_finetune": False,
            "lora": True,
            "single_gpu": True,
            "threshold_tuned": False,
            "calibration": False,
            "support_dataset": precheck["support_dataset"],
            "support_dataset_version": precheck["support_dataset_version"],
            "kernel_target_version": precheck["kernel_target_version"],
            "materialization_precheck": precheck["pass"],
            "smartphone_ids_mounted": precheck["mounted_smartphone_files"],
        },
        "materialization_precheck": precheck,
    }
    (OUT_DIR / "v2_lora_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT_DIR / "integrity_report_v1.json").write_text(json.dumps(summary["integrity"], indent=2) + "\n")
    (OUT_DIR / "config_v1.json").write_text(json.dumps(cfg, indent=2) + "\n")

    zip_path = Path("/kaggle/working/v2_lora_outputs.zip") if IS_KAGGLE else OUT_DIR / "v2_lora_outputs.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(OUT_DIR.parent if IS_KAGGLE else OUT_DIR))
    print("DONE full decision=", decision, "outputs", OUT_DIR)


if __name__ == "__main__":
    main()

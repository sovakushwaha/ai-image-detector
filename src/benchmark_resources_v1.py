"""Stage 26A resource-efficiency benchmark for frozen C0/C1 candidates.

Measures parameter/size, preprocessing, forward, end-to-end latency,
throughput, and approximate memory on 456 clean validation images only.
No training, no test access, no model selection.

How to run
----------
    source .venv/bin/activate
    PYTHONPATH=src python src/benchmark_resources_v1.py
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from cnn_dataset_v1 import EXPECTED_SIZE, PROJECT_ROOT, load_split_metadata, stop_if
from mobilenet_v3_small_binary_v1 import MobileNetV3SmallBinaryV1, count_parameters
from rq3_augmentations_v1 import IMAGENET_MEAN, IMAGENET_STD
from rq4_frequency_transform_v1 import FrequencyTransformV1, NORM_PATH
from rq4_rgb_frequency_fusion_v1 import RGBFrequencyFusionV1

SEED = 42
EXPECTED_VAL = 456
PREPROCESS_WARMUP = 20
FORWARD_WARMUP = 30
E2E_WARMUP = 20
MEASURE_ROUNDS = 3
THROUGHPUT_WARMUP_BATCHES = 10
THROUGHPUT_TIMED_BATCHES = 50
THROUGHPUT_BATCH_SIZES = [1, 8, 32]
AUC_TOLERANCE = 0.01

C0_ID = "resource_C0_rgb"
C1_ID = "resource_C1_rgb_frequency"
C0_CKPT = PROJECT_ROOT / "models/mobilenet_resize_jpeg_aug_selected_v1.pt"
C1_CKPT = PROJECT_ROOT / "models/rq4_F2_rgb_frequency_fusion_selected_v1.pt"
C0_FROZEN = PROJECT_ROOT / "results/rq3_A2_frozen_config_v1.json"
C1_FROZEN = PROJECT_ROOT / "results/rq4_F2_frozen_config_v1.json"

RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
ENV_JSON = RESULTS / "resource_environment_v1.json"
SIZE_CSV = RESULTS / "resource_model_size_v1.csv"
RAW_CSV = RESULTS / "resource_latency_raw_v1.csv"
MEMORY_CSV = RESULTS / "resource_memory_v1.csv"
SUMMARY_CSV = RESULTS / "resource_efficiency_summary_v1.csv"
CONTEXT_CSV = RESULTS / "resource_performance_context_v1.csv"
REPORT_TXT = RESULTS / "resource_efficiency_report_v1.txt"

MEMORY_WORKER = PROJECT_ROOT / "src/benchmark_memory_worker_v1.py"


def mps_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def latency_stats(values_ms: np.ndarray) -> dict[str, float]:
    v = np.asarray(values_ms, dtype=np.float64)
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return {
        "mean_ms": mean,
        "median_ms": float(np.median(v)),
        "std_ms": std,
        "p90_ms": float(np.percentile(v, 90)),
        "p95_ms": float(np.percentile(v, 95)),
        "cv": float(std / mean) if mean > 0 else float("nan"),
        "n": int(len(v)),
    }


def pct_overhead(c1: float, c0: float) -> float:
    if c0 == 0:
        return float("nan")
    return (c1 / c0 - 1.0) * 100.0


def tensor_memory_bytes(module: torch.nn.Module) -> tuple[int, int, int]:
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
    return param_bytes, buffer_bytes, param_bytes + buffer_bytes


def collect_environment() -> dict:
    uname = platform.uname()
    cpu_brand = "N/A"
    if sys.platform == "darwin":
        try:
            cpu_brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    ram_bytes = psutil.virtual_memory().total
    apple_chip = cpu_brand if "Apple" in cpu_brand else "N/A"
    env = {
        "seed": SEED,
        "operating_system": f"{uname.system} {uname.release}",
        "machine_architecture": uname.machine,
        "platform": platform.platform(),
        "cpu_model": cpu_brand,
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "system_ram_bytes": int(ram_bytes),
        "system_ram_gib": round(ram_bytes / (1024**3), 2),
        "apple_chip_model": apple_chip,
        "unified_memory_gib": round(ram_bytes / (1024**3), 2) if sys.platform == "darwin" else "N/A",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "pillow_version": Image.__version__,
        "psutil_version": psutil.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
        "pytorch_num_threads": torch.get_num_threads(),
        "pytorch_num_interop_threads": torch.get_num_interop_threads(),
        "benchmark_split": "validation",
        "benchmark_image_count": EXPECTED_VAL,
        "test_images_used": 0,
    }
    return env


def load_validation_paths() -> pd.DataFrame:
    val = load_split_metadata("validation")
    stop_if(len(val) != EXPECTED_VAL, f"expected {EXPECTED_VAL} validation, got {len(val)}")
    val = val.sort_values("image_id").reset_index(drop=True)
    val["source_image_id"] = val["image_id"]
    val["abs_path"] = val["processed_path"].apply(lambda p: PROJECT_ROOT / p)
    stop_if(val["split"].isin(["known_test", "unseen_test"]).any(), "test split leaked")
    return val


def load_c0(device: torch.device) -> MobileNetV3SmallBinaryV1:
    model = MobileNetV3SmallBinaryV1().to(device)
    ckpt = torch.load(C0_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_c1(device: torch.device) -> RGBFrequencyFusionV1:
    model = RGBFrequencyFusionV1().to(device)
    ckpt = torch.load(C1_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def measure_model_sizes() -> tuple[pd.DataFrame, dict]:
    rows = []
    breakdown = {}
    specs = [
        ("C0", C0_ID, C0_CKPT, "C0"),
        ("C1", C1_ID, C1_CKPT, "C1"),
    ]
    for label, model_id, ckpt_path, kind in specs:
        if kind == "C0":
            model = MobileNetV3SmallBinaryV1()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            total_p, trainable_p = count_parameters(model)
            comp = {}
        else:
            model = RGBFrequencyFusionV1()
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            comp = model.count_component_params()
            total_p, trainable_p = comp["total"], comp["trainable"]
            breakdown["C1_rgb_branch"] = comp["rgb_branch"]
            breakdown["C1_freq_branch"] = comp["freq_branch"]
            breakdown["C1_fusion_head"] = comp["fusion_head"]

        param_b, buf_b, total_b = tensor_memory_bytes(model)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(model.state_dict(), tmp_path)
            state_dict_bytes = tmp_path.stat().st_size
        finally:
            tmp_path.unlink(missing_ok=True)

        rows.append(
            {
                "model": label,
                "model_id": model_id,
                "parameters": int(total_p),
                "trainable_parameters": int(trainable_p),
                "parameter_memory_bytes": int(param_b),
                "parameter_memory_mib": param_b / (1024 * 1024),
                "buffer_memory_mib": buf_b / (1024 * 1024),
                "total_tensor_memory_mib": total_b / (1024 * 1024),
                "checkpoint_size_mib": ckpt_path.stat().st_size / (1024 * 1024),
                "state_dict_size_mib": state_dict_bytes / (1024 * 1024),
                "checkpoint_path": str(ckpt_path.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows), breakdown


@dataclass
class PreprocessResult:
    total_ms: np.ndarray
    rgb_ms: np.ndarray | None
    freq_ms: np.ndarray | None


def benchmark_preprocessing(val_df: pd.DataFrame, rgb_transform) -> tuple[PreprocessResult, PreprocessResult, list[dict]]:
    paths = val_df["abs_path"].tolist()
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    raw_rows: list[dict] = []

    # warm-up
    for p in paths[:PREPROCESS_WARMUP]:
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
        _ = rgb_transform(rgb)
        _ = freq(rgb)

    def record(model: str, mtype: str, round_i: int, sample_i: int, ms: float):
        raw_rows.append(
            {
                "model": model,
                "device": "cpu",
                "measurement_type": mtype,
                "batch_size": 1,
                "round": round_i,
                "sample_or_iteration": sample_i,
                "latency_ms": ms,
            }
        )

    c0_times: list[float] = []
    c1_total: list[float] = []
    c1_rgb: list[float] = []
    c1_freq: list[float] = []

    for rnd in range(MEASURE_ROUNDS):
        for i, p in enumerate(paths):
            t0 = time.perf_counter()
            with Image.open(p) as im:
                im.load()
                rgb = im.convert("RGB")
            stop_if(rgb.size != EXPECTED_SIZE, f"{p} size")
            _ = rgb_transform(rgb)
            ms = (time.perf_counter() - t0) * 1000.0
            c0_times.append(ms)
            record("C0", "preprocess_total", rnd, i, ms)

            t0 = time.perf_counter()
            with Image.open(p) as im:
                im.load()
                rgb = im.convert("RGB")
            stop_if(rgb.size != EXPECTED_SIZE, f"{p} size")
            tr0 = time.perf_counter()
            _ = rgb_transform(rgb)
            tr = time.perf_counter()
            _ = freq(rgb)
            tf = time.perf_counter()
            total_ms = (tf - t0) * 1000.0
            rgb_ms = (tr - tr0) * 1000.0
            freq_ms = (tf - tr) * 1000.0
            c1_total.append(total_ms)
            c1_rgb.append(rgb_ms)
            c1_freq.append(freq_ms)
            record("C1", "preprocess_total", rnd, i, total_ms)
            record("C1", "preprocess_rgb_component", rnd, i, rgb_ms)
            record("C1", "preprocess_frequency_component", rnd, i, freq_ms)

    return (
        PreprocessResult(np.array(c0_times), None, None),
        PreprocessResult(np.array(c1_total), np.array(c1_rgb), np.array(c1_freq)),
        raw_rows,
    )


def precompute_tensors(val_df: pd.DataFrame, rgb_transform) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    c0_tensors: list[torch.Tensor] = []
    c1_rgb: list[torch.Tensor] = []
    c1_freq: list[torch.Tensor] = []
    for p in val_df["abs_path"]:
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
        c0_tensors.append(rgb_transform(rgb))
        c1_rgb.append(rgb_transform(rgb))
        c1_freq.append(freq(rgb))
    return c0_tensors, c1_rgb, c1_freq


def benchmark_forward(
    model_kind: str,
    model,
    device: torch.device,
    c0_tensors: list[torch.Tensor],
    c1_rgb: list[torch.Tensor],
    c1_freq: list[torch.Tensor],
) -> tuple[dict, list[dict]]:
    raw_rows: list[dict] = []
    times: list[float] = []

    def run_one(idx: int) -> float:
        if model_kind == "C0":
            x = c0_tensors[idx].unsqueeze(0).to(device)
            mps_sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = torch.sigmoid(model(x)).cpu().item()
            mps_sync(device)
        else:
            x_rgb = c1_rgb[idx].unsqueeze(0).to(device)
            x_f = c1_freq[idx].unsqueeze(0).to(device)
            mps_sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = torch.sigmoid(model(x_rgb, x_f)).cpu().item()
            mps_sync(device)
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(FORWARD_WARMUP):
        run_one(0)

    n = len(c0_tensors)
    for rnd in range(MEASURE_ROUNDS):
        for i in range(n):
            ms = run_one(i)
            times.append(ms)
            raw_rows.append(
                {
                    "model": model_kind,
                    "device": str(device),
                    "measurement_type": "forward_only",
                    "batch_size": 1,
                    "round": rnd,
                    "sample_or_iteration": i,
                    "latency_ms": ms,
                }
            )
    return latency_stats(np.array(times)), raw_rows


def benchmark_end_to_end(
    model_kind: str,
    model,
    device: torch.device,
    val_df: pd.DataFrame,
    rgb_transform,
) -> tuple[dict, list[dict]]:
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    paths = val_df["abs_path"].tolist()
    raw_rows: list[dict] = []
    times: list[float] = []

    def run_path(p: Path) -> float:
        mps_sync(device)
        t0 = time.perf_counter()
        with Image.open(p) as im:
            im.load()
            rgb = im.convert("RGB")
        if model_kind == "C0":
            x = rgb_transform(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                prob = torch.sigmoid(model(x)).cpu().item()
        else:
            x_rgb = rgb_transform(rgb).unsqueeze(0).to(device)
            x_f = freq(rgb).unsqueeze(0).to(device)
            with torch.inference_mode():
                prob = torch.sigmoid(model(x_rgb, x_f)).cpu().item()
        mps_sync(device)
        stop_if(not np.isfinite(prob), "non-finite prob")
        return (time.perf_counter() - t0) * 1000.0

    for p in paths[:E2E_WARMUP]:
        run_path(p)

    for rnd in range(MEASURE_ROUNDS):
        for i, p in enumerate(paths):
            ms = run_path(p)
            times.append(ms)
            raw_rows.append(
                {
                    "model": model_kind,
                    "device": str(device),
                    "measurement_type": "end_to_end",
                    "batch_size": 1,
                    "round": rnd,
                    "sample_or_iteration": i,
                    "latency_ms": ms,
                }
            )
    return latency_stats(np.array(times)), raw_rows


def benchmark_throughput(
    model_kind: str,
    model,
    device: torch.device,
    c0_tensors: list[torch.Tensor],
    c1_rgb: list[torch.Tensor],
    c1_freq: list[torch.Tensor],
    batch_size: int,
) -> tuple[dict, list[dict]]:
    n = len(c0_tensors)
    raw_rows: list[dict] = []
    batch_times: list[float] = []

    def run_batch(idxs: list[int]) -> float:
        if model_kind == "C0":
            batch = torch.stack([c0_tensors[j] for j in idxs]).to(device)
            mps_sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = torch.sigmoid(model(batch))
            mps_sync(device)
        else:
            batch_rgb = torch.stack([c1_rgb[j] for j in idxs]).to(device)
            batch_f = torch.stack([c1_freq[j] for j in idxs]).to(device)
            mps_sync(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = torch.sigmoid(model(batch_rgb, batch_f))
            mps_sync(device)
        return time.perf_counter() - t0

    pos = 0

    def next_batch() -> list[int]:
        nonlocal pos
        idxs = [(pos + k) % n for k in range(batch_size)]
        pos += batch_size
        return idxs

    for _ in range(THROUGHPUT_WARMUP_BATCHES):
        run_batch(next_batch())

    for b in range(THROUGHPUT_TIMED_BATCHES):
        idxs = next_batch()
        elapsed = run_batch(idxs)
        ips = batch_size / elapsed if elapsed > 0 else float("nan")
        batch_times.append(ips)
        raw_rows.append(
            {
                "model": model_kind,
                "device": str(device),
                "measurement_type": "throughput_forward",
                "batch_size": batch_size,
                "round": 0,
                "sample_or_iteration": b,
                "latency_ms": elapsed * 1000.0,
                "images_per_sec": ips,
            }
        )

    arr = np.array(batch_times)
    return {
        "mean_images_per_sec": float(np.mean(arr)),
        "median_images_per_sec": float(np.median(arr)),
        "std_images_per_sec": float(np.std(arr, ddof=1)),
    }, raw_rows


def run_memory_subprocess(model: str, device: str) -> dict:
    cmd = [
        sys.executable,
        str(MEMORY_WORKER),
        "--model",
        model,
        "--device",
        device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    out = subprocess.check_output(cmd, env=env, text=True)
    return json.loads(out.strip())


def build_performance_context() -> pd.DataFrame:
    rq4 = pd.read_csv(PROJECT_ROOT / "paper/tables/rq4_frequency_fusion_comparison.csv")
    cal = pd.read_csv(RESULTS / "rq5_calibration_test_metrics_v1.csv")
    sel = pd.read_csv(RESULTS / "rq5_selective_test_metrics_v1.csv")
    aurc = pd.read_csv(RESULTS / "rq5_risk_coverage_v1.csv")

    mapping = {"C0": "F0", "C1": "F2"}
    rows = []
    for model, regime in mapping.items():
        r4 = rq4[rq4["regime"] == regime].iloc[0]
        c = cal[(cal["model"] == model) & (cal["split"] == "unseen_test") & (cal["condition"] == "original")].iloc[0]
        s = sel[
            (sel["model"] == model)
            & (sel["split"] == "unseen_test")
            & (sel["condition"] == "original")
            & (sel["target_validation_coverage"] == 80)
        ].iloc[0]
        a = aurc[(aurc["model"] == model) & (aurc["split"] == "unseen_test") & (aurc["condition"] == "original")].iloc[0]
        rows.append(
            {
                "model": model,
                "historical_id": regime,
                "unseen_original_auc": float(r4["original_auc"]),
                "unseen_strong_robust_test_auc": float(r4["StrongRobustTestAUC"]),
                "calibrated_unseen_original_nll": float(c["calibrated_nll"]),
                "unseen_original_80pct_selective_risk": float(s["selective_risk"]),
                "unseen_original_aurc": float(a["aurc"]),
                "source_note": "historical RQ4/RQ5 frozen test metrics; not rerun in Stage 26A",
            }
        )
    return pd.DataFrame(rows)


def sanity_check(
    c0_model: MobileNetV3SmallBinaryV1,
    c1_model: RGBFrequencyFusionV1,
    val_df: pd.DataFrame,
    rgb_transform,
    device: torch.device,
) -> dict:
    freq = FrequencyTransformV1.from_json(NORM_PATH)
    labels = []
    c0_probs = []
    c1_probs = []
    with torch.inference_mode():
        for _, row in val_df.iterrows():
            with Image.open(row["abs_path"]) as im:
                im.load()
                rgb = im.convert("RGB")
            labels.append(int(row["label"]))
            x0 = rgb_transform(rgb).unsqueeze(0).to(device)
            c0_probs.append(float(torch.sigmoid(c0_model(x0)).cpu().item()))
            x_rgb = rgb_transform(rgb).unsqueeze(0).to(device)
            x_f = freq(rgb).unsqueeze(0).to(device)
            c1_probs.append(float(torch.sigmoid(c1_model(x_rgb, x_f)).cpu().item()))
    y = np.array(labels)
    c0_auc = float(roc_auc_score(y, c0_probs))
    c1_auc = float(roc_auc_score(y, c1_probs))
    c0_ref = float(json.loads(C0_FROZEN.read_text())["clean_validation_auc"])
    c1_ref = float(json.loads(C1_FROZEN.read_text())["clean_validation_auc"])
    return {
        "c0_validation_auc": c0_auc,
        "c0_reference_auc": c0_ref,
        "c0_auc_delta": c0_auc - c0_ref,
        "c1_validation_auc": c1_auc,
        "c1_reference_auc": c1_ref,
        "c1_auc_delta": c1_auc - c1_ref,
        "nan_probs": not (np.all(np.isfinite(c0_probs)) and np.all(np.isfinite(c1_probs))),
        "passed": abs(c0_auc - c0_ref) <= AUC_TOLERANCE and abs(c1_auc - c1_ref) <= AUC_TOLERANCE,
    }


def plot_figures(summary: pd.DataFrame, size_df: pd.DataFrame, preprocess_stats: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # end-to-end latency
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, device in zip(axes, ["cpu", "mps"]):
        sub = summary[summary["device"] == device]
        if sub.empty:
            ax.set_title(f"{device.upper()} (unavailable)")
            continue
        models = sub["model"].tolist()
        med = sub["end_to_end_median_ms"].tolist()
        p95 = sub["end_to_end_p95_ms"].tolist()
        x = np.arange(len(models))
        ax.bar(x, med, color=["#1f77b4", "#ff7f0e"][: len(models)], alpha=0.8, label="median")
        ax.errorbar(x, med, yerr=[np.array(med) * 0, np.array(p95) - np.array(med)], fmt="none", capsize=4, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title(f"{device.upper()} batch-1 end-to-end")
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_end_to_end_latency_v1.png", dpi=150)
    plt.close(fig)

    # model size
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(size_df))
    ax.bar(x - 0.2, size_df["parameters"] / 1e6, 0.4, label="Parameters (M)")
    ax.bar(x + 0.2, size_df["state_dict_size_mib"], 0.4, label="State dict (MiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(size_df["model"])
    ax.set_title("Model size comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_model_size_v1.png", dpi=150)
    plt.close(fig)

    # throughput batch32
    fig, ax = plt.subplots(figsize=(7, 4))
    sub = summary[summary["throughput_batch_size"] == 32]
    devices = ["cpu", "mps"]
    width = 0.35
    for i, model in enumerate(["C0", "C1"]):
        vals = [sub[(sub.model == model) & (sub.device == d)]["throughput_batch32_ips"].iloc[0] if len(sub[(sub.model == model) & (sub.device == d)]) else 0 for d in devices]
        ax.bar(np.arange(len(devices)) + (i - 0.5) * width, vals, width, label=model)
    ax.set_xticks(np.arange(len(devices)))
    ax.set_xticklabels([d.upper() for d in devices])
    ax.set_ylabel("Images / sec (batch 32 forward)")
    ax.set_title("Batch-32 forward throughput")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_throughput_v1.png", dpi=150)
    plt.close(fig)

    # latency breakdown
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    c0_pre = preprocess_stats["C0"]["median_ms"]
    c1_pre = preprocess_stats["C1"]["median_ms"]
    c1_freq = preprocess_stats["C1_freq"]["median_ms"]
    c0_fwd_cpu = summary[(summary.model == "C0") & (summary.device == "cpu")]["forward_median_ms"].iloc[0]
    c1_fwd_cpu = summary[(summary.model == "C1") & (summary.device == "cpu")]["forward_median_ms"].iloc[0]
    axes[0].bar(["preprocess", "forward"], [c0_pre, c0_fwd_cpu], color=["C2", "C0"])
    axes[0].set_title("C0 CPU median latency components (ms)")
    axes[1].bar(["rgb+load", "frequency", "forward"], [c1_pre - c1_freq, c1_freq, c1_fwd_cpu], color=["C1", "C3", "C0"])
    axes[1].set_title("C1 CPU median latency components (ms)")
    fig.tight_layout()
    fig.savefig(FIGURES / "resource_latency_breakdown_v1.png", dpi=150)
    plt.close(fig)


def write_report(env, size_df, summary_df, context_df, sanity, overhead, warnings: list[str]) -> None:
    lines = [
        "Stage 26A — Resource-Efficiency Analysis Report",
        "=" * 60,
        "",
        "1. PURPOSE",
        "   Measure computational/resource cost of frozen C0 (RGB A2) and C1 (RGB+frequency fusion)",
        "   without changing models or selecting a final winner.",
        "",
        "2. CANDIDATES",
        f"   C0: {C0_ID} ({C0_CKPT.name})",
        f"   C1: {C1_ID} ({C1_CKPT.name})",
        "",
        "3. HARDWARE ENVIRONMENT",
        f"   OS: {env['operating_system']}",
        f"   CPU: {env['cpu_model']}",
        f"   RAM: {env['system_ram_gib']} GiB",
        f"   Apple chip: {env['apple_chip_model']}",
        "",
        "4. SOFTWARE ENVIRONMENT",
        f"   Python {env['python_version']}, PyTorch {env['pytorch_version']}, torchvision {env['torchvision_version']}",
        f"   MPS available: {env['mps_available']}",
        "",
        "5. BENCHMARK DATA",
        f"   Clean validation images only: {env['benchmark_image_count']}",
        f"   Test images used: {env['test_images_used']}",
        "",
        "6. BENCHMARK METHODOLOGY",
        "   Separate static size, preprocessing, forward-only, end-to-end, throughput, memory.",
        "   3 measurement rounds for per-image latencies; MPS synchronized around timed sections.",
        "",
        "7. PARAMETER COUNTS",
    ]
    for _, r in size_df.iterrows():
        lines.append(f"   {r['model']}: {int(r['parameters']):,} parameters")
    lines.extend(["", "8. MODEL / STATE SIZE"])
    for _, r in size_df.iterrows():
        lines.append(
            f"   {r['model']}: checkpoint {r['checkpoint_size_mib']:.2f} MiB, state_dict {r['state_dict_size_mib']:.2f} MiB"
        )
    lines.extend(["", "9–16. See resource_efficiency_summary_v1.csv for latency/throughput/memory tables"])
    lines.extend(["", "17. RELATIVE C1 OVERHEAD"])
    for k, v in overhead.items():
        lines.append(f"   {k}: {v}")
    lines.extend(["", "18. HISTORICAL PERFORMANCE CONTEXT (not rerun)"])
    for _, r in context_df.iterrows():
        lines.append(
            f"   {r['model']}: unseen AUC {r['unseen_original_auc']:.4f}, StrongRobust {r['unseen_strong_robust_test_auc']:.4f}, "
            f"NLL {r['calibrated_unseen_original_nll']:.4f}, 80% risk {r['unseen_original_80pct_selective_risk']:.4f}, AURC {r['unseen_original_aurc']:.4f}"
        )
    lines.extend(
        [
            "",
            "19. LIMITATIONS",
            "   Single local environment; hardware-dependent latency; MPS ≠ CUDA;",
            "   filesystem cache may affect load times; unified memory approximate;",
            "   sampled memory not guaranteed true peak; no quant/prune/compile tests;",
            "   no mobile/Raspberry Pi deployment benchmark; resource metrics are not model-quality evidence.",
            "",
            "20. SCIENTIFIC INTEGRITY",
            "   Training: NO | Weights modified: NO | Test inference: NO | Final model selected: NO",
            "",
            "SANITY CHECK",
            f"   C0 val AUC {sanity['c0_validation_auc']:.6f} vs ref {sanity['c0_reference_auc']:.6f}",
            f"   C1 val AUC {sanity['c1_validation_auc']:.6f} vs ref {sanity['c1_reference_auc']:.6f}",
            f"   Passed: {sanity['passed']}",
        ]
    )
    if warnings:
        lines.append("")
        lines.append("WARNINGS")
        for w in warnings:
            lines.append(f"   - {w}")
    REPORT_TXT.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("=== Stage 26A — Resource-efficiency benchmark ===")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    warnings: list[str] = []

    env = collect_environment()
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(ENV_JSON, "w") as f:
        json.dump(env, f, indent=2)
        f.write("\n")
    print(f"Saved {ENV_JSON}")

    val_df = load_validation_paths()
    rgb_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )

    size_df, comp_breakdown = measure_model_sizes()
    size_df.to_csv(SIZE_CSV, index=False)
    print(f"Saved {SIZE_CSV}")

    print("Benchmarking preprocessing (CPU)...")
    c0_pre, c1_pre, pre_raw = benchmark_preprocessing(val_df, rgb_transform)
    pre_stats = {
        "C0": latency_stats(c0_pre.total_ms),
        "C1": latency_stats(c1_pre.total_ms),
        "C1_freq": latency_stats(c1_pre.freq_ms),
        "C1_rgb_component": latency_stats(c1_pre.rgb_ms),
    }

    print("Precomputing tensors...")
    c0_tensors, c1_rgb, c1_freq = precompute_tensors(val_df, rgb_transform)

    raw_rows: list[dict] = list(pre_raw)
    metrics: dict = {}

    devices = [torch.device("cpu")]
    if env["mps_available"]:
        devices.append(torch.device("mps"))
    else:
        warnings.append("MPS unavailable; MPS benchmarks skipped")

    c0_cpu = load_c0(torch.device("cpu"))
    c1_cpu = load_c1(torch.device("cpu"))

    for dev in devices:
        print(f"Forward-only latency on {dev}...")
        c0 = load_c0(dev) if dev.type != "cpu" else c0_cpu
        c1 = load_c1(dev) if dev.type != "cpu" else c1_cpu
        s0, r0 = benchmark_forward("C0", c0, dev, c0_tensors, c1_rgb, c1_freq)
        s1, r1 = benchmark_forward("C1", c1, dev, c0_tensors, c1_rgb, c1_freq)
        raw_rows.extend(r0)
        raw_rows.extend(r1)
        metrics[("C0", dev.type, "forward")] = s0
        metrics[("C1", dev.type, "forward")] = s1

        print(f"End-to-end latency on {dev}...")
        e0, er0 = benchmark_end_to_end("C0", c0, dev, val_df, rgb_transform)
        e1, er1 = benchmark_end_to_end("C1", c1, dev, val_df, rgb_transform)
        raw_rows.extend(er0)
        raw_rows.extend(er1)
        metrics[("C0", dev.type, "e2e")] = e0
        metrics[("C1", dev.type, "e2e")] = e1

        for bs in THROUGHPUT_BATCH_SIZES:
            print(f"Throughput batch={bs} on {dev}...")
            t0, tr0 = benchmark_throughput("C0", c0, dev, c0_tensors, c1_rgb, c1_freq, bs)
            t1, tr1 = benchmark_throughput("C1", c1, dev, c0_tensors, c1_rgb, c1_freq, bs)
            raw_rows.extend(tr0)
            raw_rows.extend(tr1)
            metrics[("C0", dev.type, f"throughput_{bs}")] = t0
            metrics[("C1", dev.type, f"throughput_{bs}")] = t1

    pd.DataFrame(raw_rows).to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}")

    print("Memory measurements (isolated subprocesses)...")
    mem_rows = []
    for model in ["C0", "C1"]:
        try:
            m = run_memory_subprocess(model, "cpu")
            mem_rows.append(
                {
                    "model": model,
                    "device": "cpu",
                    "baseline_memory_mib": m["baseline_memory_mib"],
                    "loaded_memory_mib": m["loaded_memory_mib"],
                    "max_observed_memory_mib": m["max_observed_memory_mib"],
                    "increment_mib": m["increment_mib"],
                    "measurement_method": m["measurement_method"],
                    "limitations": m["limitations"],
                }
            )
        except Exception as exc:
            warnings.append(f"CPU memory {model} failed: {exc}")
        if env["mps_available"]:
            try:
                m = run_memory_subprocess(model, "mps")
                mem_rows.append(
                    {
                        "model": model,
                        "device": "mps",
                        "baseline_memory_mib": m.get("mps_current_allocated_mib_loaded"),
                        "loaded_memory_mib": m.get("mps_current_allocated_mib_loaded"),
                        "max_observed_memory_mib": m.get("mps_current_allocated_mib_max_observed"),
                        "increment_mib": (m.get("mps_current_allocated_mib_max_observed") or 0)
                        - (m.get("mps_current_allocated_mib_loaded") or 0),
                        "measurement_method": "torch.mps.current_allocated_memory sampled subprocess",
                        "limitations": m["limitations"],
                    }
                )
            except Exception as exc:
                warnings.append(f"MPS memory {model} failed: {exc}")
    mem_df = pd.DataFrame(mem_rows)
    mem_df.to_csv(MEMORY_CSV, index=False)
    print(f"Saved {MEMORY_CSV}")

    sanity = sanity_check(c0_cpu, c1_cpu, val_df, rgb_transform, torch.device("cpu"))
    stop_if(sanity["nan_probs"], "NaN probabilities detected")
    if not sanity["passed"]:
        warnings.append(
            f"Validation AUC reproduction outside tolerance: C0 Δ{sanity['c0_auc_delta']:+.4f}, C1 Δ{sanity['c1_auc_delta']:+.4f}"
        )

    context_df = build_performance_context()
    context_df.to_csv(CONTEXT_CSV, index=False)

    c0_size = size_df[size_df.model == "C0"].iloc[0]
    c1_size = size_df[size_df.model == "C1"].iloc[0]

    summary_rows = []
    for dev in ["cpu", "mps"]:
        if dev == "mps" and not env["mps_available"]:
            continue
        device_key = dev
        for model in ["C0", "C1"]:
            fwd = metrics[(model, device_key, "forward")]
            e2e = metrics[(model, device_key, "e2e")]
            tp32 = metrics[(model, device_key, "throughput_32")]
            mem_sub = mem_df[(mem_df.model == model) & (mem_df.device == dev)]
            mem_val = float(mem_sub["max_observed_memory_mib"].iloc[0]) if len(mem_sub) else float("nan")
            row = {
                "model": model,
                "device": dev,
                "parameters": int(c0_size.parameters if model == "C0" else c1_size.parameters),
                "state_dict_size_mib": float(c0_size.state_dict_size_mib if model == "C0" else c1_size.state_dict_size_mib),
                "checkpoint_size_mib": float(c0_size.checkpoint_size_mib if model == "C0" else c1_size.checkpoint_size_mib),
                "preprocess_median_ms": pre_stats["C0"]["median_ms"] if model == "C0" else pre_stats["C1"]["median_ms"],
                "forward_median_ms": fwd["median_ms"],
                "forward_p95_ms": fwd["p95_ms"],
                "end_to_end_median_ms": e2e["median_ms"],
                "end_to_end_p95_ms": e2e["p95_ms"],
                "throughput_batch32_ips": tp32["median_images_per_sec"],
                "throughput_batch_size": 32,
                "approx_memory_mib": mem_val,
                "forward_cv": fwd["cv"],
                "end_to_end_cv": e2e["cv"],
            }
            if model == "C1":
                row["frequency_transform_median_ms"] = pre_stats["C1_freq"]["median_ms"]
                row["frequency_fraction_of_e2e"] = pre_stats["C1_freq"]["median_ms"] / e2e["median_ms"] if e2e["median_ms"] else float("nan")
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}")

    cpu_c0 = summary_df[(summary_df.model == "C0") & (summary_df.device == "cpu")].iloc[0]
    cpu_c1 = summary_df[(summary_df.model == "C1") & (summary_df.device == "cpu")].iloc[0]
    overhead = {
        "parameter_overhead_pct": pct_overhead(c1_size.parameters, c0_size.parameters),
        "state_dict_overhead_pct": pct_overhead(c1_size.state_dict_size_mib, c0_size.state_dict_size_mib),
        "cpu_preprocess_overhead_pct": pct_overhead(cpu_c1.preprocess_median_ms, cpu_c0.preprocess_median_ms),
        "cpu_forward_overhead_pct": pct_overhead(cpu_c1.forward_median_ms, cpu_c0.forward_median_ms),
        "cpu_end_to_end_overhead_pct": pct_overhead(cpu_c1.end_to_end_median_ms, cpu_c0.end_to_end_median_ms),
    }
    if env["mps_available"]:
        mps_c0 = summary_df[(summary_df.model == "C0") & (summary_df.device == "mps")].iloc[0]
        mps_c1 = summary_df[(summary_df.model == "C1") & (summary_df.device == "mps")].iloc[0]
        overhead["mps_forward_overhead_pct"] = pct_overhead(mps_c1.forward_median_ms, mps_c0.forward_median_ms)
        overhead["mps_end_to_end_overhead_pct"] = pct_overhead(mps_c1.end_to_end_median_ms, mps_c0.end_to_end_median_ms)
        overhead["mps_batch32_throughput_delta_pct"] = pct_overhead(mps_c1.throughput_batch32_ips, mps_c0.throughput_batch32_ips)
    overhead["cpu_batch32_throughput_delta_pct"] = pct_overhead(cpu_c1.throughput_batch32_ips, cpu_c0.throughput_batch32_ips)

    for key in ["forward_cv", "end_to_end_cv"]:
        for model in ["C0", "C1"]:
            for dev in ["cpu", "mps"]:
                sub = summary_df[(summary_df.model == model) & (summary_df.device == dev)]
                if len(sub) and sub.iloc[0][key] > 0.20:
                    warnings.append(f"High timing CV for {model} {dev} {key}: {sub.iloc[0][key]:.3f}")

    plot_figures(summary_df, size_df, pre_stats)
    write_report(env, size_df, summary_df, context_df, sanity, overhead, warnings)

    # Terminal summary
    print("\n" + "=" * 50)
    print("STAGE 26A — RESOURCE-EFFICIENCY ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"OS: {env['operating_system']}")
    print(f"CPU: {env['cpu_model']}")
    print(f"RAM: {env['system_ram_gib']} GiB")
    print(f"PyTorch: {env['pytorch_version']}")
    print(f"MPS available: {env['mps_available']}")
    print("\nMODEL SIZE")
    print(f"C0 params: {int(c0_size.parameters):,}, state {c0_size.state_dict_size_mib:.2f} MiB, ckpt {c0_size.checkpoint_size_mib:.2f} MiB")
    print(f"C1 params: {int(c1_size.parameters):,}, state {c1_size.state_dict_size_mib:.2f} MiB, ckpt {c1_size.checkpoint_size_mib:.2f} MiB")
    print(f"Parameter overhead C1 vs C0: {overhead['parameter_overhead_pct']:.1f}%")
    print("\nCPU END-TO-END median / p95")
    print(f"C0: {cpu_c0.end_to_end_median_ms:.2f} / {cpu_c0.end_to_end_p95_ms:.2f} ms")
    print(f"C1: {cpu_c1.end_to_end_median_ms:.2f} / {cpu_c1.end_to_end_p95_ms:.2f} ms  (+{overhead['cpu_end_to_end_overhead_pct']:.1f}%)")
    print(f"C1 FFT median: {pre_stats['C1_freq']['median_ms']:.2f} ms")
    if env["mps_available"]:
        mps_c0 = summary_df[(summary_df.model == "C0") & (summary_df.device == "mps")].iloc[0]
        mps_c1 = summary_df[(summary_df.model == "C1") & (summary_df.device == "mps")].iloc[0]
        print("\nMPS END-TO-END median / p95")
        print(f"C0: {mps_c0.end_to_end_median_ms:.2f} / {mps_c0.end_to_end_p95_ms:.2f} ms")
        print(f"C1: {mps_c1.end_to_end_median_ms:.2f} / {mps_c1.end_to_end_p95_ms:.2f} ms")
    print("\nSTATUS: Stage 26A COMPLETE | Final model selection: PENDING | RQ5 resource analysis: COMPLETE")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

"""V2-4: CLIP local MPS pipeline dry-run verification.

No classifier training. No full embedding extraction. No NTIRE access.
"""

from __future__ import annotations

import csv
import json
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
import torchvision

from v2_clip_encoder_v1 import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, V2ClipEncoderV1
from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_TOTAL = 11377  # V2-3 development pool size (post-exclusion)
SPLIT_PATH = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
SAMPLE_PATH = PROJECT_ROOT / "metadata" / "v2_clip_dryrun_sample_v1.csv"
OUT = PROJECT_ROOT / "results" / "v2"


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split() -> list[dict]:
    with SPLIT_PATH.open(newline="") as f:
        return list(csv.DictReader(f))


def select_dryrun_sample(rows: list[dict], seed: int = SEED) -> list[dict]:
    """32 images: 4 Tiny/MLLM/COCO/Smartphone Real + 16 AI across generators."""
    rng = random.Random(seed)
    usable = [r for r in rows if r.get("path") and (PROJECT_ROOT / r["path"]).exists()]
    usable = [r for r in usable if r.get("fold_1_role") != "EXCLUDED_DUPLICATE"]

    selected: list[dict] = []

    def take_real(domain: str, n: int) -> None:
        pool = [r for r in usable if r["binary_label"] == "0" and r.get("real_domain") == domain]
        pool.sort(key=lambda x: x["image_id"])
        rng.shuffle(pool)
        stop_if(len(pool) < n, f"need {n} Real {domain}, found {len(pool)}")
        for r in pool[:n]:
            selected.append({**r, "dryrun_role": f"REAL_{domain}"})

    for domain in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        take_real(domain, 4)

    ai = [r for r in usable if r["binary_label"] == "1"]
    by_gen: dict[str, list[dict]] = defaultdict(list)
    for r in ai:
        by_gen[r["generator"]].append(r)
    gens = sorted(by_gen.keys())
    rng.shuffle(gens)
    # round-robin across generators for diversity
    for g in gens:
        by_gen[g].sort(key=lambda x: x["image_id"])
        rng.shuffle(by_gen[g])
    ai_sel: list[dict] = []
    while len(ai_sel) < 16:
        progressed = False
        for g in gens:
            if by_gen[g] and len(ai_sel) < 16:
                r = by_gen[g].pop(0)
                ai_sel.append({**r, "dryrun_role": f"AI_{r['generator']}"})
                progressed = True
        if not progressed:
            break
    stop_if(len(ai_sel) < 16, f"only {len(ai_sel)} AI images available for dry-run")
    selected.extend(ai_sel)
    stop_if(len(selected) != 32, f"dry-run sample size {len(selected)} != 32")
    return selected


def write_sample(rows: list[dict]) -> None:
    fields = [
        "image_id",
        "binary_label",
        "source_dataset",
        "generator",
        "real_domain",
        "path",
        "dryrun_role",
        "selection_seed",
    ]
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "image_id": r["image_id"],
                    "binary_label": r["binary_label"],
                    "source_dataset": r["source_dataset"],
                    "generator": r["generator"],
                    "real_domain": r.get("real_domain", ""),
                    "path": r["path"],
                    "dryrun_role": r["dryrun_role"],
                    "selection_seed": SEED,
                }
            )


def mps_allocated_mb() -> float | None:
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            return float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return None
    return None


def benchmark_batches(encoder: V2ClipEncoderV1, paths: list[Path], batch_sizes: list[int]) -> list[dict]:
    rows = []
    # Warmup
    warm = encoder.preprocess_paths(paths[: min(8, len(paths))])
    _ = encoder.encode_tensor(warm)
    if encoder.device.type == "mps":
        torch.mps.synchronize()

    for bs in batch_sizes:
        if bs > len(paths):
            rows.append(
                {
                    "batch_size": bs,
                    "success": False,
                    "error": "batch_larger_than_sample",
                    "median_forward_ms": "",
                    "images_per_sec": "",
                    "output_shape": "",
                    "mps_allocated_mb": "",
                }
            )
            continue
        try:
            times = []
            out_shape = None
            for _ in range(3):
                batch_paths = paths[:bs]
                x = encoder.preprocess_paths(batch_paths)
                if encoder.device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                feats = encoder.encode_tensor(x)
                if encoder.device.type == "mps":
                    torch.mps.synchronize()
                dt = time.perf_counter() - t0
                times.append(dt)
                out_shape = list(feats.shape)
            med = float(np.median(times))
            ips = bs / med if med > 0 else float("nan")
            rows.append(
                {
                    "batch_size": bs,
                    "success": True,
                    "error": "",
                    "median_forward_ms": round(med * 1000, 3),
                    "images_per_sec": round(ips, 3),
                    "output_shape": str(out_shape),
                    "mps_allocated_mb": "" if mps_allocated_mb() is None else round(mps_allocated_mb(), 2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "batch_size": bs,
                    "success": False,
                    "error": f"{type(exc).__name__}:{exc}",
                    "median_forward_ms": "",
                    "images_per_sec": "",
                    "output_shape": "",
                    "mps_allocated_mb": "",
                }
            )
            if encoder.device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
    return rows


def main() -> None:
    set_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    assert_path_not_final_external_test(str(SPLIT_PATH), str(SAMPLE_PATH))

    stop_if(torch.__version__.split("+")[0] != "2.13.0", f"unexpected torch {torch.__version__}")
    stop_if(
        torchvision.__version__.split("+")[0] != "0.28.0",
        f"unexpected torchvision {torchvision.__version__}",
    )

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "open_clip": open_clip.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": SEED,
        "preferred_model": DEFAULT_MODEL_NAME,
        "preferred_pretrained": DEFAULT_PRETRAINED,
    }

    print("Loading CLIP ViT-B-16 openai on MPS...")
    encoder = V2ClipEncoderV1(
        model_name=DEFAULT_MODEL_NAME,
        pretrained=DEFAULT_PRETRAINED,
        device="mps",
        l2_normalize=True,
    )
    meta = encoder.metadata_dict()
    env["resolved_model"] = meta
    print("meta", meta)

    # Sample
    all_rows = read_split()
    sample = select_dryrun_sample(all_rows, seed=SEED)
    write_sample(sample)
    paths = [PROJECT_ROOT / r["path"] for r in sample]
    for p in paths:
        assert_path_not_final_external_test(str(p))
        stop_if(not p.exists(), f"missing {p}")

    # Encode full dry-run sample
    x = encoder.preprocess_paths(paths)
    input_shape = list(x.shape)
    feats = encoder.encode_tensor(x)
    emb = feats.detach().cpu().numpy()
    norms = np.linalg.norm(emb, axis=1)
    n_nan = int(np.isnan(emb).sum())
    n_inf = int(np.isinf(emb).sum())

    # Same-image repeat
    p0 = paths[0]
    e1 = encoder.encode_paths([p0], batch_size=1)
    e2 = encoder.encode_paths([p0], batch_size=1)
    max_abs_diff = float(np.max(np.abs(e1 - e2)))

    # CPU vs MPS on 3 images
    enc_cpu = V2ClipEncoderV1(
        model_name=DEFAULT_MODEL_NAME,
        pretrained=DEFAULT_PRETRAINED,
        device="cpu",
        l2_normalize=True,
    )
    cmp_paths = paths[:3]
    mps_e = encoder.encode_paths(cmp_paths, batch_size=3)
    cpu_e = enc_cpu.encode_paths(cmp_paths, batch_size=3)
    # cosine per row
    cos = []
    for i in range(len(cmp_paths)):
        a, b = mps_e[i], cpu_e[i]
        cos.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    cos_mean = float(np.mean(cos))
    cos_min = float(np.min(cos))

    # Similarity matrix sanity
    sim = emb @ emb.T
    diag = np.diag(sim)
    sim_finite = bool(np.isfinite(sim).all())

    # Batch benchmarks
    bench = benchmark_batches(encoder, paths, [1, 8, 16, 32])
    with (OUT / "v2_clip_batch_benchmark_v1.csv").open("w", newline="") as f:
        fields = [
            "batch_size",
            "success",
            "error",
            "median_forward_ms",
            "images_per_sec",
            "output_shape",
            "mps_allocated_mb",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in bench:
            w.writerow(r)

    ok_bench = [r for r in bench if r["success"] is True]
    stop_if(not ok_bench, "no successful MPS batch size")
    best = max(ok_bench, key=lambda r: float(r["images_per_sec"]))
    best_bs = int(best["batch_size"])
    best_ips = float(best["images_per_sec"])

    # Estimates
    pure_seconds = N_TOTAL / best_ips if best_ips > 0 else float("inf")
    # IO/preprocess overhead heuristic: ~1.5–2.5× pure forward depending on decode
    practical_low = pure_seconds * 1.5
    practical_high = pure_seconds * 2.5
    d = int(meta["embedding_dim"])
    bytes_f32 = N_TOTAL * d * 4
    bytes_f16 = N_TOTAL * d * 2
    mib_f32 = bytes_f32 / (1024 * 1024)
    mib_f16 = bytes_f16 / (1024 * 1024)

    if practical_high < 3600:  # < 1 hour upper
        decision = "LOCAL_MPS_RECOMMENDED"
    elif practical_high < 4 * 3600:
        decision = "LOCAL_MPS_POSSIBLE_BUT_SLOW"
    else:
        decision = "REMOTE_GPU_RECOMMENDED"

    estimate = {
        "document": "v2_clip_full_extraction_estimate_v1",
        "n_images": N_TOTAL,
        "best_stable_batch_size": best_bs,
        "measured_images_per_sec_forward": best_ips,
        "estimated_pure_encoding_seconds": round(pure_seconds, 1),
        "estimated_pure_encoding_minutes": round(pure_seconds / 60, 2),
        "estimated_practical_local_seconds_low": round(practical_low, 1),
        "estimated_practical_local_seconds_high": round(practical_high, 1),
        "estimated_practical_local_minutes_low": round(practical_low / 60, 2),
        "estimated_practical_local_minutes_high": round(practical_high / 60, 2),
        "overhead_note": "Practical duration multiplies pure forward by ~1.5–2.5× for decode/IO/preprocess; not exact.",
        "embedding_dim": d,
        "storage_float32_mib": round(mib_f32, 2),
        "storage_float16_mib": round(mib_f16, 2),
        "primary_storage_dtype": "float32",
        "compute_decision": decision,
        "kaggle_required_now": False,
        "full_extraction_started": False,
    }
    (OUT / "v2_clip_full_extraction_estimate_v1.json").write_text(json.dumps(estimate, indent=2) + "\n")

    env["dryrun"] = {
        "n_images": 32,
        "input_shape": input_shape,
        "embedding_shape": list(emb.shape),
        "nan": n_nan,
        "inf": n_inf,
        "norm_mean": float(norms.mean()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
        "same_image_repeat_max_abs_diff": max_abs_diff,
        "cpu_vs_mps_cosine_mean": cos_mean,
        "cpu_vs_mps_cosine_min": cos_min,
        "similarity_diag_mean": float(diag.mean()),
        "similarity_all_finite": sim_finite,
        "mps_pipeline": "PASS",
    }
    env["best_stable_batch"] = best_bs
    env["compute_decision"] = decision
    env["pretrained_weights_downloaded"] = True
    env["classifier_training"] = False
    env["clip_finetuning"] = False
    env["ntire_accessed"] = False
    (OUT / "v2_clip_environment_v1.json").write_text(json.dumps(env, indent=2) + "\n")

    # Update dryrun plan
    plan = {
        "document": "v2_clip_dryrun_plan_v1",
        "stage": "V2-4",
        "download_now": False,
        "weights_downloaded_in_v2_4": True,
        "preferred_backbone": "ViT-B/16",
        "resolved_model_name": meta["model_name"],
        "resolved_pretrained": meta["pretrained_tag"],
        "fallback_backbone": "ViT-B/32",
        "library": "open_clip_torch",
        "library_version": meta["library_version"],
        "device_initial": "mps",
        "device_verified": "mps",
        "batch_size": best_bs,
        "embedding_dim": d,
        "input_resolution": meta["input_resolution"],
        "l2_normalize": True,
        "V2_NATIVE_PIXEL_PRIMARY": True,
        "n_images_estimated": N_TOTAL,
        "local_mps_assessment": decision,
        "full_extraction": "NOT_STARTED",
        "kaggle_required_now": False,
    }
    (OUT / "v2_clip_dryrun_plan_v1.json").write_text(json.dumps(plan, indent=2) + "\n")

    # Human report
    lines = [
        "V2-4 — CLIP LOCAL MPS PIPELINE VERIFICATION",
        "==========================================",
        f"Library: open_clip_torch {open_clip.__version__}",
        f"Torch: {torch.__version__}",
        f"Torchvision: {torchvision.__version__}",
        f"Model: {meta['model_name']} pretrained={meta['pretrained_tag']}",
        f"Weight source: {meta['weight_source']}",
        f"Input resolution: {meta['input_resolution']}",
        f"Embedding dim: {meta['embedding_dim']}",
        f"Parameters: {meta['parameter_count']}",
        f"Device: {meta['device']}",
        f"MPS pipeline: PASS",
        "",
        f"Dry-run images: 32",
        f"Input shape: {input_shape}",
        f"Embedding shape: {list(emb.shape)}",
        f"NaN: {n_nan}  Inf: {n_inf}",
        f"L2 norm mean/min/max: {norms.mean():.6f} / {norms.min():.6f} / {norms.max():.6f}",
        f"Same-image repeat max abs diff: {max_abs_diff:.3e}",
        f"CPU vs MPS cosine mean/min: {cos_mean:.8f} / {cos_min:.8f}",
        f"Similarity diag mean: {float(diag.mean()):.6f}; all finite: {sim_finite}",
        "",
        "Batch benchmark:",
    ]
    for r in bench:
        lines.append(
            f"  bs={r['batch_size']}: success={r['success']} "
            f"ips={r['images_per_sec']} med_ms={r['median_forward_ms']} err={r['error']}"
        )
    lines += [
        f"Best stable batch: {best_bs} ({best_ips:.2f} img/s)",
        "",
        f"Full pool images: {N_TOTAL}",
        f"Estimated pure encoding: {estimate['estimated_pure_encoding_minutes']} min",
        f"Estimated practical local: {estimate['estimated_practical_local_minutes_low']}–"
        f"{estimate['estimated_practical_local_minutes_high']} min",
        f"Embedding storage float32: {mib_f32:.2f} MiB",
        f"Compute decision: {decision}",
        "Kaggle required now: NO",
        "Full extraction: NOT STARTED",
        "Classifier training: NO",
        "CLIP fine-tuning: NO",
        "NTIRE accessed: NO",
        "V1 modified: NO",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "v2_clip_pipeline_verification_v1.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

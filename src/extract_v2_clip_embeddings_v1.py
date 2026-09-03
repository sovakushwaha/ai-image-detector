"""Extract frozen CLIP embeddings for the full V2 development pool (MPS).

Resumable chunked extraction. No classifier training. No NTIRE access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from v2_clip_encoder_v1 import V2ClipEncoderV1
from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "metadata" / "v2_split_assignments_v1.csv"
CHUNK_DIR = PROJECT_ROOT / "results" / "v2" / "clip_embedding_chunks_v1"
OUT_NPZ = PROJECT_ROOT / "results" / "v2" / "v2_clip_embeddings_v1.npz"
OUT_MANIFEST = PROJECT_ROOT / "metadata" / "v2_clip_embedding_manifest_v1.csv"
OUT_REPORT = PROJECT_ROOT / "results" / "v2" / "v2_clip_embedding_extraction_report_v1.txt"
EXPECTED_N = 11377
CHUNK_SIZE = 500
BATCH_SIZE = 32


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def load_usable_rows() -> list[dict]:
    with SPLIT_PATH.open(newline="") as f:
        rows = list(csv.DictReader(f))
    usable = [r for r in rows if r.get("fold_1_role") != "EXCLUDED_DUPLICATE"]
    usable.sort(key=lambda r: r["image_id"])
    stop_if(len(usable) != EXPECTED_N, f"usable rows {len(usable)} != {EXPECTED_N}")
    for r in usable:
        p = PROJECT_ROOT / r["path"]
        assert_path_not_final_external_test(str(p))
        stop_if(not p.exists(), f"missing image {p}")
    return usable


def chunk_path(i: int) -> Path:
    return CHUNK_DIR / f"chunk_{i:04d}.npz"


def extract(device: str = "mps", batch_size: int = BATCH_SIZE, chunk_size: int = CHUNK_SIZE) -> None:
    assert_path_not_final_external_test(str(CHUNK_DIR), str(OUT_NPZ))
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_usable_rows()
    encoder = V2ClipEncoderV1(device=device, l2_normalize=True)
    stop_if(encoder.meta.embedding_dim != 512, f"unexpected dim {encoder.meta.embedding_dim}")

    n_chunks = (len(rows) + chunk_size - 1) // chunk_size
    t0 = time.perf_counter()
    done_images = 0

    for ci in range(n_chunks):
        out_p = chunk_path(ci)
        start = ci * chunk_size
        end = min(len(rows), start + chunk_size)
        chunk_rows = rows[start:end]
        if out_p.exists():
            data = np.load(out_p, allow_pickle=False)
            stop_if(data["embeddings"].shape[0] != len(chunk_rows), f"bad chunk size {out_p}")
            done_images += len(chunk_rows)
            print(f"skip existing chunk {ci+1}/{n_chunks} ({len(chunk_rows)} images)")
            continue

        ids = [r["image_id"] for r in chunk_rows]
        paths = [PROJECT_ROOT / r["path"] for r in chunk_rows]
        emb = encoder.encode_paths(paths, batch_size=batch_size, return_numpy=True)
        emb = np.asarray(emb, dtype=np.float32)
        stop_if(emb.shape != (len(chunk_rows), 512), f"chunk shape {emb.shape}")
        stop_if(np.isnan(emb).any() or np.isinf(emb).any(), f"NaN/Inf in chunk {ci}")
        np.savez_compressed(out_p, image_ids=np.array(ids), embeddings=emb)
        done_images += len(chunk_rows)
        elapsed = time.perf_counter() - t0
        print(
            f"wrote chunk {ci+1}/{n_chunks} n={len(chunk_rows)} "
            f"total={done_images}/{len(rows)} elapsed={elapsed/60:.1f}m"
        )

    # Consolidate
    all_ids = []
    all_emb = []
    for ci in range(n_chunks):
        data = np.load(chunk_path(ci), allow_pickle=False)
        all_ids.extend(data["image_ids"].tolist())
        all_emb.append(data["embeddings"].astype(np.float32))
    embeddings = np.concatenate(all_emb, axis=0)
    image_ids = np.array(all_ids)
    stop_if(embeddings.shape != (EXPECTED_N, 512), f"final shape {embeddings.shape}")
    stop_if(len(set(image_ids.tolist())) != EXPECTED_N, "duplicate image_ids in embeddings")

    # Align rows to sorted usable order (already sorted)
    id_to_row = {r["image_id"]: i for i, r in enumerate(rows)}
    order = np.array([id_to_row[i] for i in image_ids], dtype=np.int64)
    # Reorder embeddings to match sorted rows order
    inv = np.empty_like(order)
    # Actually chunks were written in sorted rows order, so image_ids should already match
    stop_if(list(image_ids) != [r["image_id"] for r in rows], "image_id order mismatch vs split")

    norms = np.linalg.norm(embeddings, axis=1)
    sha = hashlib.sha256(embeddings.tobytes()).hexdigest()
    wall = time.perf_counter() - t0

    np.savez_compressed(
        OUT_NPZ,
        image_ids=image_ids,
        embeddings=embeddings,
        embedding_dim=np.array([512]),
        model_name=np.array(["ViT-B-16-quickgelu"]),
        pretrained=np.array(["openai"]),
        sha256=np.array([sha]),
    )

    # Manifest
    fields = [
        "image_id",
        "binary_label",
        "source_dataset",
        "real_domain",
        "generator_id",
        "prompt_group",
        "embedding_row",
        "fold_1_role",
        "fold_2_role",
        "fold_3_role",
        "fold_4_role",
        "path",
    ]
    with OUT_MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(
                {
                    "image_id": r["image_id"],
                    "binary_label": r["binary_label"],
                    "source_dataset": r["source_dataset"],
                    "real_domain": r.get("real_domain", ""),
                    "generator_id": r.get("canonical_generator_id", r.get("generator", "")),
                    "prompt_group": r.get("prompt_group", ""),
                    "embedding_row": i,
                    "fold_1_role": r["fold_1_role"],
                    "fold_2_role": r["fold_2_role"],
                    "fold_3_role": r["fold_3_role"],
                    "fold_4_role": r["fold_4_role"],
                    "path": r["path"],
                }
            )

    report = "\n".join(
        [
            "V2-5 CLIP embedding extraction report",
            f"N={embeddings.shape[0]} D={embeddings.shape[1]} dtype={embeddings.dtype}",
            f"device={device} batch_size={batch_size} chunk_size={chunk_size}",
            f"wall_clock_sec={wall:.1f} wall_clock_min={wall/60:.2f}",
            f"nan={int(np.isnan(embeddings).sum())} inf={int(np.isinf(embeddings).sum())}",
            f"norm_mean={float(norms.mean()):.6f} min={float(norms.min()):.6f} max={float(norms.max()):.6f}",
            f"artifact={OUT_NPZ} bytes={OUT_NPZ.stat().st_size}",
            f"sha256_embeddings_bytes={sha}",
            f"model={encoder.meta.model_name} pretrained={encoder.meta.pretrained_tag}",
            "CLIP frozen=YES fine_tune=NO NTIRE=NO",
        ]
    )
    OUT_REPORT.write_text(report + "\n")
    print(report)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = p.parse_args()
    extract(device=args.device, batch_size=args.batch_size, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()

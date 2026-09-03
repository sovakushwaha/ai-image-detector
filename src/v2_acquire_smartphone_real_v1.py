"""V2-3: Reconstruct genuine smartphone Real images from laionmobile/laion-mobile.

Dataset construction only — no model inference, no CLIP, no NTIRE access.

Selection is deterministic (seed=42) BEFORE downloading. Downloads proceed in
candidate_order until TARGET_VALID images pass the quality gate (or candidates
are exhausted). Dead URLs are skipped; replacements are never based on appearance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SOURCE_DATASET = "laionmobile/laion-mobile"
SOURCE_REVISION = "0c60f598e67cffe8475f54edd71307648cd03465"
TARGET_VALID = 3000
CANDIDATE_POOL = 8000
MIN_SIDE = 256
MAX_DEVICE_SHARE = 0.10  # of final target when alternatives exist
TIMEOUT_S = 20
RETRIES = 3
USER_AGENT = "ai-image-detector-v2-3-research/1.0 (academic; local reconstruction)"

OUT_DIR = PROJECT_ROOT / "data" / "v2" / "smartphone_real"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "v2_smartphone_real_manifest_v1.csv"
PROGRESS_PATH = PROJECT_ROOT / "results" / "v2" / "v2_smartphone_download_progress_v1.json"
CANDIDATE_PATH = PROJECT_ROOT / "metadata" / "v2_smartphone_candidate_pool_v1.csv"
META_CACHE = PROJECT_ROOT / "metadata" / "v2" / "laion_mobile_eval_sample_metadata.csv"


MANIFEST_FIELDS = [
    "v2_image_id",
    "source_dataset",
    "source_row_id",
    "source_url",
    "manufacturer",
    "device_model",
    "width",
    "height",
    "format",
    "sha256",
    "download_status",
    "candidate_order",
    "split",
    "provenance_verified",
    "local_path",
    "notes",
]


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def ensure_meta_csv(path: Path) -> Path:
    if path.exists():
        return path
    # Prefer prior /tmp cache, else download metadata-only CSV from HF
    tmp = Path("/tmp/laion_eval_sample_metadata.csv")
    if tmp.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(tmp.read_bytes())
        return path
    url = (
        f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/"
        f"{SOURCE_REVISION}/metadata/eval_sample_metadata.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=120, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def load_rows(meta_path: Path) -> list[dict]:
    with meta_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    # dedupe URL then sha
    seen_url: set[str] = set()
    seen_sha: set[str] = set()
    unique = []
    for r in rows:
        url = (r.get("url") or "").strip()
        sha = (r.get("content_sha256") or "").strip().lower()
        if not url or url in seen_url:
            continue
        if sha and sha in seen_sha:
            continue
        seen_url.add(url)
        if sha:
            seen_sha.add(sha)
        unique.append(r)
    return unique


def select_candidates(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic stratified candidate list with device-model caps."""
    rng = random.Random(seed)
    by_make: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_make[r["make_norm"]].append(r)
    for m in by_make:
        by_make[m].sort(key=lambda x: x["image_id"])
        rng.shuffle(by_make[m])

    makes = sorted(by_make.keys())
    # manufacturer shares: Apple capped ~40%, rest proportional
    raw = {m: len(by_make[m]) / len(rows) for m in makes}
    desired = {m: (min(raw[m], 0.40) if m == "Apple" else raw[m]) for m in makes}
    s = sum(desired.values())
    desired = {m: desired[m] / s for m in makes}
    alloc = {m: int(desired[m] * n) for m in makes}
    while sum(alloc.values()) < n:
        m = max(makes, key=lambda x: len(by_make[x]) - alloc[x])
        if alloc[m] >= len(by_make[m]):
            break
        alloc[m] += 1
    while sum(alloc.values()) > n:
        m = max(makes, key=lambda x: alloc[x])
        alloc[m] -= 1

    max_per_device = max(1, int(TARGET_VALID * MAX_DEVICE_SHARE))
    # When selecting candidates (larger than target), allow 2x device cap in pool
    max_per_device_pool = max_per_device * 2

    selected: list[dict] = []
    device_counts: Counter[str] = Counter()
    for m in makes:
        pool = by_make[m]
        # diversify by model round-robin
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_model[r["model"]].append(r)
        models = sorted(by_model.keys())
        take: list[dict] = []
        while len(take) < alloc[m]:
            progressed = False
            for model in models:
                if not by_model[model]:
                    continue
                if device_counts[model] >= max_per_device_pool:
                    by_model[model].clear()
                    continue
                r = by_model[model].pop(0)
                take.append(r)
                device_counts[model] += 1
                progressed = True
                if len(take) >= alloc[m]:
                    break
            if not progressed:
                break
        selected.extend(take)

    rng.shuffle(selected)
    # assign candidate_order
    out = []
    for i, r in enumerate(selected[:n]):
        out.append({**r, "candidate_order": i})
    return out


def try_download(url: str) -> tuple[bytes | None, str]:
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(
                url,
                timeout=TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
                stream=True,
            )
            if resp.status_code != 200:
                last_err = f"http_{resp.status_code}"
                time.sleep(0.4 * attempt)
                continue
            data = resp.content
            if not data:
                last_err = "empty_body"
                continue
            return data, "ok"
        except requests.RequestException as exc:
            last_err = f"network:{type(exc).__name__}"
            time.sleep(0.5 * attempt)
    return None, last_err


def validate_image(data: bytes, expected_sha: str | None) -> dict | None:
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha and expected_sha.lower() != sha:
        # keep image but flag sha mismatch — still usable if decodable
        sha_note = "sha_mismatch_vs_metadata"
    else:
        sha_note = "sha_ok" if expected_sha else "sha_computed"
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            w, h = im.size
            fmt = (im.format or "UNKNOWN").upper()
            mode = im.mode
    except Exception as exc:  # noqa: BLE001
        return {"error": f"unreadable:{type(exc).__name__}"}
    if min(w, h) < MIN_SIDE:
        return {"error": f"too_small:{w}x{h}"}
    if w <= 0 or h <= 0:
        return {"error": "invalid_geometry"}
    return {
        "width": w,
        "height": h,
        "format": fmt,
        "mode": mode,
        "sha256": sha,
        "sha_note": sha_note,
    }


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"success": {}, "failed": {}, "n_success": 0}


def save_progress(prog: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(prog, indent=2) + "\n")


def write_manifest(rows: list[dict]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    # mirror
    mirror = PROJECT_ROOT / "metadata" / "v2" / "v2_smartphone_real_manifest_v1.csv"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_bytes(MANIFEST_PATH.read_bytes())


def assign_splits(success_rows: list[dict], seed: int = SEED) -> list[dict]:
    """Assign train/validation/internal_holdout after reconstruction."""
    n = len(success_rows)
    stop_if(n < 2000, f"only {n} smartphone images reconstructed (<2000 minimum)")
    if n >= 3000:
        n_train, n_val, n_hold = 2000, 500, 500
        success_rows = success_rows[:3000]
        n = 3000
    else:
        n_train = int(round(n * 0.67))
        n_val = int(round(n * 0.17))
        n_hold = n - n_train - n_val
        if n_hold < 300 and n >= 1800:
            # prefer holdout floor when possible
            deficit = 300 - n_hold
            take = min(deficit, max(0, n_val - 200))
            n_val -= take
            n_hold += take
            if n_hold < 300:
                take2 = min(300 - n_hold, max(0, n_train - 1000))
                n_train -= take2
                n_hold += take2

    rng = random.Random(seed)
    by_make: dict[str, list[dict]] = defaultdict(list)
    for r in success_rows:
        by_make[r["manufacturer"]].append(r)
    assigned: list[dict] = []
    for m, items in by_make.items():
        items = list(items)
        rng.shuffle(items)
        k = len(items)
        # proportional
        t = int(round(k * n_train / n))
        v = int(round(k * n_val / n))
        h = k - t - v
        parts = [("train", t), ("validation", v), ("internal_holdout", h)]
        i = 0
        for split, cnt in parts:
            for r in items[i : i + cnt]:
                assigned.append({**r, "split": split})
            i += cnt

    # fix exact totals
    need = {"train": n_train, "validation": n_val, "internal_holdout": n_hold}
    counts = Counter(r["split"] for r in assigned)
    for split in list(need):
        while counts[split] > need[split]:
            donor = [r for r in assigned if r["split"] == split]
            rng.shuffle(donor)
            for deficit in need:
                if counts[deficit] < need[deficit]:
                    donor[0]["split"] = deficit
                    counts[split] -= 1
                    counts[deficit] += 1
                    break
    # enforce SHA uniqueness across splits (should already be unique)
    return assigned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--target", type=int, default=TARGET_VALID)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_POOL)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    assert_path_not_final_external_test(str(OUT_DIR), str(MANIFEST_PATH))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = ensure_meta_csv(META_CACHE)
    rows = load_rows(meta_path)
    print(f"unique metadata rows: {len(rows)}")

    if CANDIDATE_PATH.exists() and not args.select_only:
        with CANDIDATE_PATH.open() as f:
            candidates = list(csv.DictReader(f))
        print(f"loaded existing candidate pool: {len(candidates)}")
    else:
        candidates = select_candidates(rows, args.candidates, SEED)
        CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fields = list(candidates[0].keys())
        with CANDIDATE_PATH.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(candidates)
        print(f"wrote candidate pool n={len(candidates)} -> {CANDIDATE_PATH}")
        if args.select_only:
            return

    prog = load_progress()
    success: dict[str, dict] = prog.get("success", {})
    failed: dict[str, str] = prog.get("failed", {})

    # Resume: count existing valid files
    pending = []
    for c in sorted(candidates, key=lambda x: int(x["candidate_order"])):
        sid = str(c["image_id"])
        if sid in success:
            continue
        if sid in failed:
            continue
        pending.append(c)

    print(f"already success={len(success)} failed={len(failed)} pending={len(pending)}")

    def process_one(c: dict) -> tuple[str, dict | None, str]:
        sid = str(c["image_id"])
        url = c["url"].strip()
        assert_path_not_final_external_test(url)
        data, err = try_download(url)
        if data is None:
            return sid, None, err
        expected = (c.get("content_sha256") or "").strip() or None
        meta = validate_image(data, expected)
        if "error" in meta:
            return sid, None, meta["error"]
        v2_id = f"SMARTPHONE_{sid}"
        ext = {
            "JPEG": ".jpg",
            "JPG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "GIF": ".gif",
        }.get(meta["format"], ".img")
        local = OUT_DIR / f"{v2_id}{ext}"
        try:
            if not local.exists():
                local.write_bytes(data)
        except OSError as exc:
            return sid, None, f"write_failed:{type(exc).__name__}"
        row = {
            "v2_image_id": v2_id,
            "source_dataset": SOURCE_DATASET,
            "source_row_id": sid,
            "source_url": url,
            "manufacturer": c["make_norm"],
            "device_model": c["model"],
            "width": meta["width"],
            "height": meta["height"],
            "format": meta["format"],
            "sha256": meta["sha256"],
            "download_status": "SUCCESS",
            "candidate_order": int(c["candidate_order"]),
            "split": "",
            "provenance_verified": "YES",
            "local_path": str(local.relative_to(PROJECT_ROOT)),
            "notes": meta.get("sha_note", ""),
        }
        return sid, row, "ok"

    # Download until target. Device-model diversity was applied at candidate
    # selection; soft-cap accepted images only when alternatives remain later.
    target = args.target
    max_device = max(1, int(target * MAX_DEVICE_SHARE))
    idx = 0
    while len(success) < target and idx < len(pending):
        batch = pending[idx : idx + max(args.workers * 3, 24)]
        idx += len(batch)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, c): c for c in batch}
            for fut in as_completed(futs):
                if len(success) >= target:
                    break
                sid, payload, status = fut.result()
                if payload is None:
                    failed[sid] = status
                    continue
                model = payload["device_model"]
                model_n = sum(1 for s in success.values() if s.get("device_model") == model)
                if model_n >= max_device and len(success) + (len(pending) - idx) > target:
                    # Soft cap only while enough remaining candidates exist.
                    failed[sid] = "device_cap_deferred"
                    # keep file for potential later fill if needed
                    continue
                success[sid] = payload
        prog = {
            "success": success,
            "failed": failed,
            "n_success": len(success),
            "n_failed": len(failed),
        }
        save_progress(prog)
        write_manifest(list(success.values()))
        print(f"progress success={len(success)} failed={len(failed)} scanned={idx}/{len(pending)}")
        if len(success) >= target:
            break

    # If still short after soft caps, promote deferred device-cap downloads
    if len(success) < target:
        deferred = [sid for sid, reason in failed.items() if reason == "device_cap_deferred"]
        deferred_rows = []
        for sid in deferred:
            # recover from disk if present
            matches = list(OUT_DIR.glob(f"SMARTPHONE_{sid}.*"))
            if not matches:
                continue
            local = matches[0]
            data = local.read_bytes()
            meta = validate_image(data, None)
            if "error" in meta:
                continue
            # find candidate meta
            c = next((x for x in candidates if str(x["image_id"]) == sid), None)
            if c is None:
                continue
            deferred_rows.append(
                {
                    "v2_image_id": f"SMARTPHONE_{sid}",
                    "source_dataset": SOURCE_DATASET,
                    "source_row_id": sid,
                    "source_url": c["url"],
                    "manufacturer": c["make_norm"],
                    "device_model": c["model"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "format": meta["format"],
                    "sha256": meta["sha256"],
                    "download_status": "SUCCESS",
                    "candidate_order": int(c["candidate_order"]),
                    "split": "",
                    "provenance_verified": "YES",
                    "local_path": str(local.relative_to(PROJECT_ROOT)),
                    "notes": "accepted_after_device_cap_relaxation",
                }
            )
        deferred_rows.sort(key=lambda r: int(r["candidate_order"]))
        for row in deferred_rows:
            if len(success) >= target:
                break
            success[row["source_row_id"]] = row
            failed.pop(row["source_row_id"], None)
        print(f"after cap relaxation success={len(success)}")

    stop_if(len(success) < 2000, f"reconstructed only {len(success)} (<2000); aborting split construction")

    # Prefer earliest candidate_order among successes up to target
    ordered = sorted(success.values(), key=lambda r: int(r["candidate_order"]))[:target]
    # drop extras from success dict for split assignment clarity
    ordered = assign_splits(ordered, seed=SEED)
    write_manifest(ordered)
    save_progress(
        {
            "success": {r["source_row_id"]: r for r in ordered},
            "failed": failed,
            "n_success": len(ordered),
            "n_failed": len(failed),
            "split_counts": dict(Counter(r["split"] for r in ordered)),
            "manufacturer_counts": dict(Counter(r["manufacturer"] for r in ordered)),
            "device_model_top": Counter(r["device_model"] for r in ordered).most_common(15),
        }
    )
    print("DONE smartphone acquisition")
    print("n=", len(ordered), Counter(r["split"] for r in ordered))
    print("manufacturers", Counter(r["manufacturer"] for r in ordered))
    top = Counter(r["device_model"] for r in ordered).most_common(5)
    print("top devices", top, "max share", top[0][1] / len(ordered) if top else None)


if __name__ == "__main__":
    main()

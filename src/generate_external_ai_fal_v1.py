#!/usr/bin/env python3
"""
SUPERSEDED — DO NOT RUN FOR STAGE 27A
Replaced by Stage 27A V2 public-dataset protocol before external detector inference.

Stage 27A protocol v1.1 — fal.ai AI image acquisition (resumable).

Generates 400 AI images (4 generators × 100 locked prompts) via FAL_KEY.
Does NOT load the detector or run inference.
Never prints FAL_KEY.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

EXT = ROOT / "data" / "external_v1"
META = EXT / "metadata"
RESULTS = ROOT / "results"
PROMPTS_CSV = ROOT / "metadata" / "external_prompt_set_v1.csv"
MANIFEST_CSV = META / "external_ai_generation_manifest_v1.csv"
SMOKE_JSON = RESULTS / "external_fal_smoke_test_v1.json"
PROGRESS_JSON = RESULTS / "external_fal_acquisition_progress_v1.json"

GENERATORS = {
    "gpt_image_2": {
        "group_id": "G1",
        "name": "GPT Image 2",
        "vendor": "OpenAI",
        "requested_endpoint": "openai/gpt-image-2",
        "dir": EXT / "native" / "ai" / "gpt_image_2",
        "file_prefix": "GPT2_",
        "id_prefix": "EXT_GPT2_",
    },
    "gemini_3_1_flash_image": {
        "group_id": "G2",
        "name": "Gemini 3.1 Flash Image / Nano Banana 2",
        "vendor": "Google",
        "requested_endpoint": "fal-ai/gemini-3.1-flash-image-preview",
        "dir": EXT / "native" / "ai" / "gemini_3_1_flash_image",
        "file_prefix": "GEM31_",
        "id_prefix": "EXT_GEM31_",
    },
    "stable_diffusion_3_5_large": {
        "group_id": "G3",
        "name": "Stable Diffusion 3.5 Large",
        "vendor": "Stability AI",
        "requested_endpoint": "fal-ai/stable-diffusion-v35-large",
        "dir": EXT / "native" / "ai" / "stable_diffusion_3_5_large",
        "file_prefix": "SD35_",
        "id_prefix": "EXT_SD35_",
    },
    "seedream_5_pro": {
        "group_id": "G4",
        "name": "Seedream 5.0 Pro",
        "vendor": "ByteDance",
        "requested_endpoint": "bytedance/seedream/v5/pro/text-to-image",
        "dir": EXT / "native" / "ai" / "seedream_5_pro",
        "file_prefix": "SEED5P_",
        "id_prefix": "EXT_SEED5P_",
    },
}

MANIFEST_FIELDS = [
    "external_image_id",
    "prompt_id",
    "prompt_index",
    "prompt",
    "provider",
    "underlying_generator_vendor",
    "generator_family",
    "generator_key",
    "requested_model_endpoint",
    "actual_model_endpoint",
    "compatibility_adjustment",
    "request_id",
    "returned_url",
    "generation_timestamp",
    "generation_seed",
    "seed_supported",
    "requested_settings_json",
    "actual_width",
    "actual_height",
    "content_type",
    "native_extension",
    "native_path",
    "file_size_bytes",
    "sha256",
    "generation_status",
    "retry_count",
    "provenance_verified",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompts() -> list[dict]:
    rows = list(csv.DictReader(PROMPTS_CSV.open()))
    if len(rows) != 100:
        raise RuntimeError(f"Expected 100 prompts, found {len(rows)}")
    return rows


def load_manifest() -> dict[tuple[str, str], dict]:
    if not MANIFEST_CSV.exists():
        return {}
    out = {}
    for row in csv.DictReader(MANIFEST_CSV.open()):
        out[(row["generator_key"], row["prompt_id"])] = row
    return out


def save_manifest(rows_by_key: dict[tuple[str, str], dict]) -> None:
    import fcntl

    META.mkdir(parents=True, exist_ok=True)
    ordered = []
    for gkey in GENERATORS:
        for i in range(1, 101):
            pid = f"P{i:04d}"
            row = rows_by_key.get((gkey, pid))
            if row:
                ordered.append(row)
    lock_path = META / "external_ai_generation_manifest_v1.lock"
    with lock_path.open("w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        # reload latest before write to merge parallel generator processes
        latest = load_manifest()
        latest.update(rows_by_key)
        ordered = []
        for gkey in GENERATORS:
            for i in range(1, 101):
                pid = f"P{i:04d}"
                row = latest.get((gkey, pid))
                if row:
                    ordered.append(row)
        with MANIFEST_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            for row in ordered:
                w.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})
        fcntl.flock(lockf, fcntl.LOCK_UN)


def prompt_index(prompt_id: str) -> int:
    return int(prompt_id.replace("P", ""))


def build_arguments(gkey: str, prompt: str, pidx: int) -> tuple[dict, str, str]:
    """Return (arguments, actual_endpoint, compatibility_note)."""
    requested = GENERATORS[gkey]["requested_endpoint"]
    actual = requested
    note = ""
    if gkey == "gpt_image_2":
        args = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": "square",
            "quality": "high",
            "output_format": "png",
        }
        note = "image_size=square (validated in P0001 smoke test)"
    elif gkey == "gemini_3_1_flash_image":
        args = {
            "prompt": prompt,
            "num_images": 1,
            "aspect_ratio": "1:1",
            "resolution": "1K",
            "output_format": "png",
            "enable_web_search": False,
        }
    elif gkey == "stable_diffusion_3_5_large":
        args = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": "square_hd",
            "output_format": "png",
            "seed": 420000 + pidx,
            "enable_safety_checker": True,
        }
    elif gkey == "seedream_5_pro":
        args = {
            "prompt": prompt,
            "num_images": 1,
            "image_size": "square_hd",
            "output_format": "png",
            "enable_safety_checker": True,
        }
        note = "image_size=square_hd for square output (auto_1K returned non-square on P0001)"
    else:
        raise KeyError(gkey)
    return args, actual, note


def extract_image_info(result: dict) -> tuple[str, str | None, int | None]:
    """Return url, content_type, seed."""
    images = result.get("images") or []
    if not images:
        # some endpoints use image
        if result.get("image"):
            images = [result["image"]]
    if not images:
        raise RuntimeError(f"No images in result keys={list(result.keys())}")
    img = images[0]
    if isinstance(img, str):
        return img, None, result.get("seed")
    url = img.get("url")
    if not url:
        raise RuntimeError(f"Image missing url: {img}")
    return url, img.get("content_type"), result.get("seed")


def download_image(url: str, dest_no_ext: Path) -> Path:
    req = urllib.request.Request(url, headers={"User-Agent": "ai-image-detector-external-v1"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
        ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    ext = ".png"
    if "jpeg" in ctype or "jpg" in ctype:
        ext = ".jpg"
    elif "webp" in ctype:
        ext = ".webp"
    elif url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
        ext = ".jpg"
    elif url.lower().endswith(".webp"):
        ext = ".webp"
    dest = dest_no_ext.with_suffix(ext)
    dest.write_bytes(data)
    # validate openable
    with Image.open(dest) as im:
        im.load()
    return dest


class BalanceExhaustedError(RuntimeError):
    """fal.ai account locked / exhausted balance — stop acquisition immediately."""


def fal_queue_generate(endpoint: str, arguments: dict, timeout_s: float = 300.0) -> tuple[dict, str]:
    """Submit to fal queue and poll until complete. Returns (result, request_id)."""
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError("FAL_KEY absent")
    submit_url = f"https://queue.fal.run/{endpoint}"
    payload = json.dumps(arguments).encode()
    req = urllib.request.Request(
        submit_url,
        data=payload,
        headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            submitted = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            body = ""
        low = body.lower()
        if exc.code == 403 and ("exhausted balance" in low or "user is locked" in low):
            raise BalanceExhaustedError(body[:300] or str(exc)) from exc
        raise RuntimeError(f"HTTPError: HTTP Error {exc.code}: {exc.reason}; body={body[:300]}") from exc
    request_id = submitted.get("request_id") or ""
    status_url = submitted.get("status_url") or f"https://queue.fal.run/{endpoint}/requests/{request_id}/status"
    response_url = submitted.get("response_url") or f"https://queue.fal.run/{endpoint}/requests/{request_id}"
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"fal queue timeout after {timeout_s}s request_id={request_id}")
        sreq = urllib.request.Request(
            status_url,
            headers={"Authorization": f"Key {key}"},
            method="GET",
        )
        with urllib.request.urlopen(sreq, timeout=60) as resp:
            status = json.loads(resp.read().decode())
        st = status.get("status")
        if st == "COMPLETED":
            rreq = urllib.request.Request(
                response_url,
                headers={"Authorization": f"Key {key}"},
                method="GET",
            )
            with urllib.request.urlopen(rreq, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            # some responses wrap under 'payload' or return images at top-level
            if isinstance(result, dict) and "images" not in result and "image" not in result:
                if "payload" in result and isinstance(result["payload"], dict):
                    result = result["payload"]
            return result, request_id
        if st in {"FAILED", "CANCELLED", "ERROR"}:
            raise RuntimeError(f"fal queue failed status={st} body={status}")
        time.sleep(2.0)


def generate_one(
    gkey: str,
    prompt_row: dict,
    retry_budget: int = 3,
) -> dict:
    g = GENERATORS[gkey]
    pid = prompt_row["prompt_id"]
    pidx = prompt_index(pid)
    prompt = prompt_row["generation_prompt"]
    args, actual_endpoint, compat = build_arguments(gkey, prompt, pidx)
    g["dir"].mkdir(parents=True, exist_ok=True)
    dest_stem = g["dir"] / f"{g['file_prefix']}{pid}"

    last_err = ""
    for attempt in range(retry_budget + 1):
        try:
            result, request_id = fal_queue_generate(actual_endpoint, args, timeout_s=360.0)
            if not isinstance(result, dict):
                result = dict(result)
            url, content_type, seed = extract_image_info(result)
            path = download_image(url, dest_stem)
            with Image.open(path) as im:
                w, h = im.size
            row = {
                "external_image_id": f"{g['id_prefix']}{pidx:04d}",
                "prompt_id": pid,
                "prompt_index": str(pidx),
                "prompt": prompt,
                "provider": "fal.ai",
                "underlying_generator_vendor": g["vendor"],
                "generator_family": g["name"],
                "generator_key": gkey,
                "requested_model_endpoint": g["requested_endpoint"],
                "actual_model_endpoint": actual_endpoint,
                "compatibility_adjustment": compat,
                "request_id": request_id,
                "returned_url": url.split("?")[0][:300],
                "generation_timestamp": utc_now(),
                "generation_seed": "" if seed is None else str(seed),
                "seed_supported": "true" if seed is not None else ("true" if "seed" in args else "false"),
                "requested_settings_json": json.dumps(args, sort_keys=True),
                "actual_width": str(w),
                "actual_height": str(h),
                "content_type": content_type or "",
                "native_extension": path.suffix.lower(),
                "native_path": str(path.relative_to(ROOT)),
                "file_size_bytes": str(path.stat().st_size),
                "sha256": sha256_file(path),
                "generation_status": "success",
                "retry_count": str(attempt),
                "provenance_verified": "true",
                "notes": "queue_http_poller",
            }
            if "seed" in args and seed is None:
                row["generation_seed"] = str(args["seed"])
                row["seed_supported"] = "true"
                row["notes"] = "queue_http_poller; requested seed recorded; return seed absent"
            return row
        except BalanceExhaustedError:
            raise
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            # Compatibility: GPT image_size variants
            if gkey == "gpt_image_2" and attempt == 0 and "image_size" in str(exc).lower():
                args["image_size"] = "square"
                compat = "switched image_size → square after schema error"
                continue
            if gkey == "gpt_image_2" and attempt == 1:
                args["image_size"] = {"width": 1024, "height": 1024}
                compat = "switched image_size to {width:1024,height:1024}"
                continue
            if gkey == "seedream_5_pro" and attempt == 0 and "image_size" in str(exc).lower():
                args["image_size"] = "square_hd"
                compat = "switched image_size → square_hd"
                continue
            if gkey == "gemini_3_1_flash_image" and attempt == 0:
                if "resolution" in args:
                    args.pop("resolution", None)
                    args["image_size"] = "1K"
                    compat = "dropped resolution; used image_size=1K"
                    continue
            # Do not burn retries on account lock / exhausted balance
            if "exhausted balance" in last_err.lower() or "user is locked" in last_err.lower():
                raise BalanceExhaustedError(last_err) from exc
            time.sleep(min(2 ** attempt, 8))
    return {
        "external_image_id": f"{g['id_prefix']}{pidx:04d}",
        "prompt_id": pid,
        "prompt_index": str(pidx),
        "prompt": prompt,
        "provider": "fal.ai",
        "underlying_generator_vendor": g["vendor"],
        "generator_family": g["name"],
        "generator_key": gkey,
        "requested_model_endpoint": g["requested_endpoint"],
        "actual_model_endpoint": actual_endpoint,
        "compatibility_adjustment": compat,
        "request_id": "",
        "returned_url": "",
        "generation_timestamp": utc_now(),
        "generation_seed": "",
        "seed_supported": "false",
        "requested_settings_json": json.dumps(args, sort_keys=True),
        "actual_width": "",
        "actual_height": "",
        "content_type": "",
        "native_extension": "",
        "native_path": "",
        "file_size_bytes": "",
        "sha256": "",
        "generation_status": "failed",
        "retry_count": str(retry_budget),
        "provenance_verified": "false",
        "notes": last_err[:500],
    }


def is_successful_existing(row: dict | None) -> bool:
    if not row or row.get("generation_status") != "success":
        return False
    path = ROOT / row.get("native_path", "")
    if not path.exists() or path.stat().st_size == 0:
        return False
    if row.get("sha256") and sha256_file(path) != row["sha256"]:
        return False
    return True


def write_progress(manifest: dict) -> None:
    counts = {}
    fails = []
    for gkey in GENERATORS:
        ok = sum(
            1
            for i in range(1, 101)
            if is_successful_existing(manifest.get((gkey, f"P{i:04d}")))
        )
        counts[gkey] = ok
        for i in range(1, 101):
            row = manifest.get((gkey, f"P{i:04d}"))
            if row and row.get("generation_status") == "failed":
                fails.append({"generator": gkey, "prompt_id": f"P{i:04d}", "notes": row.get("notes", "")})
    payload = {
        "updated_at": utc_now(),
        "fal_key_present": bool(os.environ.get("FAL_KEY")),
        "counts": counts,
        "total_ai_success": sum(counts.values()),
        "failed": fails,
        "endpoints": {k: v["requested_endpoint"] for k, v in GENERATORS.items()},
    }
    PROGRESS_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def run_smoke(prompts: list[dict], manifest: dict) -> dict:
    smoke = {"fal_key_present": bool(os.environ.get("FAL_KEY")), "timestamp": utc_now(), "results": {}}
    p0001 = prompts[0]
    assert p0001["prompt_id"] == "P0001"
    print("\n=== PRE-FLIGHT SMOKE TESTS (P0001 counts as official sample) ===")
    print("Endpoints:")
    for gkey, g in GENERATORS.items():
        print(f"  {gkey}: {g['requested_endpoint']}")
    print("Pricing: consult fal.ai model pages for current per-image rates before full run.")
    print("Cost safety: max 100 successful images per generator; num_images=1.\n")

    for gkey in GENERATORS:
        existing = manifest.get((gkey, "P0001"))
        if is_successful_existing(existing):
            print(f"[smoke skip] {gkey} P0001 already successful")
            smoke["results"][gkey] = {
                "status": "already_present",
                "endpoint": GENERATORS[gkey]["requested_endpoint"],
                "path": existing.get("native_path"),
            }
            continue
        print(f"[smoke] {gkey} P0001 ...")
        row = generate_one(gkey, p0001)
        manifest[(gkey, "P0001")] = row
        save_manifest(manifest)
        write_progress(manifest)
        ok = row["generation_status"] == "success"
        print(
            f"  → {'OK' if ok else 'FAIL'} endpoint={row['actual_model_endpoint']} "
            f"compat={row['compatibility_adjustment'] or 'none'} "
            f"size={row.get('actual_width')}x{row.get('actual_height')} "
            f"retries={row['retry_count']}"
        )
        if not ok:
            print(f"  notes: {row['notes'][:200]}")
        smoke["results"][gkey] = {
            "status": row["generation_status"],
            "requested_endpoint": row["requested_model_endpoint"],
            "actual_endpoint": row["actual_model_endpoint"],
            "compatibility_adjustment": row["compatibility_adjustment"],
            "width": row.get("actual_width"),
            "height": row.get("actual_height"),
            "path": row.get("native_path"),
            "notes": row.get("notes", "")[:300],
        }
    SMOKE_JSON.write_text(json.dumps(smoke, indent=2) + "\n")
    return smoke


def run_full(prompts: list[dict], manifest: dict, only: str | None = None) -> None:
    keys = [only] if only else list(GENERATORS)
    for gkey in keys:
        print(f"\n=== Generating {gkey} ===")
        for pr in prompts:
            pid = pr["prompt_id"]
            existing = manifest.get((gkey, pid))
            if is_successful_existing(existing):
                continue
            print(f"[{gkey}] {pid} ...", flush=True)
            try:
                row = generate_one(gkey, pr)
            except BalanceExhaustedError as exc:
                print("STOP: fal.ai balance exhausted / account locked.", flush=True)
                print(f"  detail: {str(exc)[:240]}", flush=True)
                print("Top up at fal.ai/dashboard/billing, then re-run --full (resumable).", flush=True)
                write_progress(manifest)
                raise SystemExit(3) from exc
            manifest[(gkey, pid)] = row
            save_manifest(manifest)
            write_progress(manifest)
            status = row["generation_status"]
            print(f"  → {status} {row.get('native_path','')} retries={row['retry_count']}", flush=True)
            if status != "success":
                print(f"  FAIL: {row['notes'][:200]}", flush=True)
                notes = (row.get("notes") or "").lower()
                if "exhausted balance" in notes or "user is locked" in notes:
                    print("STOP: consecutive balance lock detected in failure notes.", flush=True)
                    raise SystemExit(3)


def readiness_from_manifest(manifest: dict) -> dict:
    real_n = len(list((EXT / "native" / "real" / "coco2017").glob("*.jpg")))
    counts = {"real": real_n}
    for gkey in GENERATORS:
        counts[gkey] = sum(
            1 for i in range(1, 101) if is_successful_existing(manifest.get((gkey, f"P{i:04d}")))
        )
    counts["total"] = counts["real"] + sum(counts[g] for g in GENERATORS)
    required = {
        "real": 400,
        "gpt_image_2": 100,
        "gemini_3_1_flash_image": 100,
        "stable_diffusion_3_5_large": 100,
        "seedream_5_pro": 100,
        "total": 800,
    }
    complete = all(counts[k] >= required[k] for k in required)
    return {"counts": counts, "required": required, "complete": complete}


def print_gate(report: dict) -> None:
    c = report["counts"]
    print("\n" + "=" * 50)
    if report["complete"]:
        print("STAGE 27A — DATA READINESS GATE PASSED")
    else:
        print("STAGE 27A — DATA READINESS INCOMPLETE")
    print("=" * 50)
    print(f"Real:                       {c['real']} / 400")
    print(f"GPT Image 2:                {c['gpt_image_2']} / 100")
    print(f"Gemini 3.1 Flash Image:     {c['gemini_3_1_flash_image']} / 100")
    print(f"Stable Diffusion 3.5 Large: {c['stable_diffusion_3_5_large']} / 100")
    print(f"Seedream 5.0 Pro:           {c['seedream_5_pro']} / 100")
    print(f"Total:                      {c['total']} / 800")
    print(f"MODEL INFERENCE: {'AUTHORISED' if report['complete'] else 'NOT STARTED'}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--only", choices=list(GENERATORS), default=None)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()

    print("FAL_KEY detected:", "YES" if bool(os.environ.get("FAL_KEY")) else "NO")
    if not os.environ.get("FAL_KEY"):
        print("STOP: FAL_KEY absent")
        return 2

    for g in GENERATORS.values():
        g["dir"].mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts()
    manifest = load_manifest()

    if args.gate_only:
        report = readiness_from_manifest(manifest)
        print_gate(report)
        return 0 if report["complete"] else 1

    smoke = run_smoke(prompts, manifest)
    failed_smoke = [k for k, v in smoke["results"].items() if v["status"] not in ("success", "already_present")]
    if failed_smoke:
        print("\nSMOKE FAILURES:", failed_smoke)
        print("STOP before full generation for failed endpoints.")
        if args.smoke_only:
            return 1
        # Continue other generators only if --full and some passed
        if not args.full:
            return 1

    if args.smoke_only:
        report = readiness_from_manifest(manifest)
        print_gate(report)
        return 0

    if args.full or not args.smoke_only:
        # default path after smoke: full generation
        run_full(prompts, manifest, only=args.only)

    report = readiness_from_manifest(manifest)
    # write readiness report
    lines = [
        "STAGE 27A — DATA READINESS REPORT (protocol v1.1 fal.ai)",
        f"Updated: {utc_now()}",
        f"FAL_KEY present: {bool(os.environ.get('FAL_KEY'))}",
        "",
        f"Real: {report['counts']['real']} / 400",
        f"GPT Image 2: {report['counts']['gpt_image_2']} / 100",
        f"Gemini 3.1 Flash Image: {report['counts']['gemini_3_1_flash_image']} / 100",
        f"Stable Diffusion 3.5 Large: {report['counts']['stable_diffusion_3_5_large']} / 100",
        f"Seedream 5.0 Pro: {report['counts']['seedream_5_pro']} / 100",
        f"Total: {report['counts']['total']} / 800",
        f"Complete: {report['complete']}",
        f"Model inference: {'AUTHORISED' if report['complete'] else 'NOT STARTED'}",
        "",
        "Midjourney required: NO (SUPERSEDED_BY_PROTOCOL_V1_1)",
    ]
    (RESULTS / "external_data_readiness_report_v1.txt").write_text("\n".join(lines) + "\n")
    (RESULTS / "external_data_readiness_v1.json").write_text(
        json.dumps(
            {
                **report,
                "fal_key_present": bool(os.environ.get("FAL_KEY")),
                "protocol_version": "1.1",
                "midjourney_required": False,
            },
            indent=2,
        )
        + "\n"
    )
    print_gate(report)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())

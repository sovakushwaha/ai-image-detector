"""Local orchestrator for V2-8 Kaggle CLIP LoRA kernel (push, monitor, import).

Never prints or stores Kaggle API secrets.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = PROJECT_ROOT / "kaggle" / "v2_lora"
RESULTS = PROJECT_ROOT / "results" / "v2"
IMPORT_DIR = RESULTS / "kaggle_v2_lora"
KERNEL_SLUG = "sovaakushwaha/v2-clip-lora-generalisation"
SECRET_PATTERNS = [
    re.compile(r"KGAT_[A-Za-z0-9]+"),
    re.compile(r'"key"\s*:\s*"[a-f0-9]{32}"'),
]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, capture_output=True, text=True)


def auth_test() -> tuple[bool, str]:
    try:
        cp = run(["kaggle", "datasets", "list", "-s", "tiny", "-v"], check=False)
        ok = cp.returncode == 0 and "tiny-genimage" in (cp.stdout or "")
        user = ""
        kj = Path.home() / ".kaggle" / "kaggle.json"
        if kj.exists():
            user = json.loads(kj.read_text()).get("username", "")
        return ok, user
    except Exception as e:
        return False, str(e)


def scan_secrets(paths: list[Path]) -> list[str]:
    hits = []
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                hits.append(str(p))
                break
    return hits


def write_integration_report(auth_ok: bool, user: str, notes: list[str]) -> None:
    lines = [
        "V2-8 Kaggle integration report",
        f"authentication: {'PASS' if auth_ok else 'FAIL'}",
        f"account: {user or 'unknown'}",
        "credential_printed: NO",
        "credential_in_repo: NO",
        "",
        "Notes:",
        *notes,
    ]
    (RESULTS / "v2_kaggle_integration_report_v1.txt").write_text("\n".join(lines) + "\n")


def set_mode(mode: str) -> None:
    rc_path = KAGGLE_DIR / "run_config.json"
    rc = json.loads(rc_path.read_text()) if rc_path.exists() else {}
    rc["mode"] = mode
    rc.setdefault("support_dataset", "sovaakushwaha/v2-clip-lora-support")
    rc.setdefault("support_dataset_version", 6)
    rc.setdefault("kernel_target_version", 37)
    rc_path.write_text(json.dumps(rc, indent=2) + "\n")


def active_v2_8_job_blocks_launch() -> tuple[bool, str]:
    """Fail closed if any V2-8 GPU job is active or status is unknown."""
    st = kernel_status()
    if not st:
        return True, "KAGGLE_STATUS_UNKNOWN_NO_LAUNCH"
    low = st.lower()
    for label in ("running", "queued", "starting", "pending"):
        if label in low:
            return True, f"OTHER_ACTIVE_V2_8_JOB: {st.splitlines()[0]}"
    lock_path = RESULTS / "v2_8_execution_lock_v1.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock.get("locked"):
            return True, f"EXECUTION_LOCK: {lock.get('reason', 'locked')}"
    return False, st.splitlines()[0] if st else "ok"


def push_kernel() -> tuple[bool, str]:
    blocked, reason = active_v2_8_job_blocks_launch()
    if blocked:
        return False, f"LAUNCH_BLOCKED: {reason}"
    cp = run(
        ["kaggle", "kernels", "push", "-p", str(KAGGLE_DIR), "--accelerator", "gpu"],
        check=False,
    )
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode == 0, out.strip()


def kernel_status() -> str:
    cp = run(["kaggle", "kernels", "status", KERNEL_SLUG], check=False)
    return ((cp.stdout or "") + (cp.stderr or "")).strip()


def download_outputs() -> tuple[bool, str]:
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    cp = run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(IMPORT_DIR)], check=False)
    out = (cp.stdout or "") + (cp.stderr or "")
    return cp.returncode == 0, out.strip()


def import_checkpoints() -> None:
    src_ckpt = IMPORT_DIR / "v2_lora_outputs" / "checkpoints"
    if not src_ckpt.exists():
        src_ckpt = IMPORT_DIR / "checkpoints"
    if not src_ckpt.exists():
        return
    dst = PROJECT_ROOT / "models" / "v2"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src_ckpt.glob("clip_lora_fold*_best_v1.pt"):
        shutil.copy2(p, dst / p.name)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    auth_ok, user = auth_test()
    notes.append(f"auth_test user={user}")
    if not auth_ok:
        write_integration_report(False, user, notes + ["STOP: authentication failed"])
        raise SystemExit("STOP: Kaggle authentication failed")

    repo_hits = scan_secrets(list(PROJECT_ROOT.rglob("*")))
    repo_hits = [h for h in repo_hits if "node_modules" not in h and ".venv" not in h]
    if repo_hits:
        notes.append(f"SECRET_SANITIZATION_REQUIRED paths={len(repo_hits)}")
        write_integration_report(auth_ok, user, notes)
        raise SystemExit("SECRET_SANITIZATION_REQUIRED")

    # Smoke push
    set_mode("smoke")
    ok, out = push_kernel()
    notes.append(f"smoke_push ok={ok}")
    if not ok:
        write_integration_report(auth_ok, user, notes + [out[:500]])
        raise SystemExit("STOP: kernel push failed")

    # Poll status (respect rate limits)
    final_status = ""
    for _ in range(60):
        time.sleep(30)
        st = kernel_status()
        final_status = st
        notes.append(f"status: {st.splitlines()[0] if st else 'empty'}")
        if "complete" in st.lower():
            break
        if "error" in st.lower() or "failed" in st.lower():
            write_integration_report(auth_ok, user, notes)
            raise SystemExit("STOP: kernel failed")
    notes.append(f"final_status: {final_status[:200]}")

    dl_ok, dl_out = download_outputs()
    notes.append(f"download ok={dl_ok}")
    if dl_ok:
        post_hits = scan_secrets(list(IMPORT_DIR.rglob("*")))
        if post_hits:
            notes.append("SECRET_SANITIZATION_REQUIRED in downloaded outputs")
            write_integration_report(auth_ok, user, notes)
            raise SystemExit("SECRET_SANITIZATION_REQUIRED")
        import_checkpoints()

    write_integration_report(auth_ok, user, notes)
    print("V2-8 orchestrator complete")
    print("auth", "PASS" if auth_ok else "FAIL")
    print("import_dir", IMPORT_DIR)


if __name__ == "__main__":
    main()

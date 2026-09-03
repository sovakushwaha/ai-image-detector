"""Paginate Kaggle kernel output with backoff; download scientific artifacts only."""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "v2" / "kaggle_v2_lora_v37"
KERNEL = ("sovaakushwaha", "v2-clip-lora-generalisation")
SCIENTIFIC = re.compile(
    r"(v2_lora_outputs\.zip|v2_lora_outputs/|__results__\.html|"
    r"\.(pt|json|csv)$|heartbeat|smoke|materialization|summary|predict|"
    r"metric|checkpoint|environment|param_audit|integrity|one_batch|config_v1)",
    re.I,
)


def list_all_files(api: KaggleApi, page_size: int = 200, delay_s: float = 3.0):
    owner, slug = KERNEL
    token = None
    files = []
    page = 0
    with api.build_kaggle_client() as client:
        while True:
            page += 1
            req = ApiListKernelSessionOutputRequest()
            req.user_name = owner
            req.kernel_slug = slug
            req.page_size = page_size
            if token:
                req.page_token = token
            for attempt in range(8):
                try:
                    resp = client.kernels.kernels_api_client.list_kernel_session_output(req)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 7:
                        wait = delay_s * (2**attempt)
                        print(f"429 page {page} attempt {attempt+1}, sleep {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    raise
            batch = resp.files or []
            files.extend(batch)
            print(f"page {page}: +{len(batch)} total {len(files)}")
            token = resp.next_page_token
            if not token:
                return files, resp.log
            time.sleep(delay_s)


def download_matches(files, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for item in files:
        if not SCIENTIFIC.search(item.file_name):
            continue
        out = dest / item.file_name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            saved.append(out)
            continue
        for attempt in range(5):
            try:
                r = requests.get(item.url, timeout=120)
                r.raise_for_status()
                out.write_bytes(r.content)
                saved.append(out)
                print("saved", item.file_name, out.stat().st_size)
                break
            except Exception as e:
                if attempt < 4:
                    time.sleep(5 * (attempt + 1))
                else:
                    print("FAIL", item.file_name, e)
    return saved


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    files, log = list_all_files(api)
    print("listed", len(files), "files")
    names = [f.file_name for f in files if SCIENTIFIC.search(f.file_name)]
    print("scientific matches:", len(names))
    for n in sorted(names)[:80]:
        print(" ", n)
    saved = download_matches(files, OUT_DIR)
    if log:
        log_path = OUT_DIR / "v2-clip-lora-generalisation.log"
        if not log_path.exists():
            log_path.write_text(log)
    print("downloaded", len(saved), "files to", OUT_DIR)


if __name__ == "__main__":
    main()

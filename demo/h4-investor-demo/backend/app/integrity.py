from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MANIFEST_PATH, PROJECT_ROOT


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class IntegrityResult:
    verified: bool
    status: str
    manifest: str
    n_checked: int
    artifacts: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "status": self.status,
            "manifest": self.manifest,
            "n_checked": self.n_checked,
            "artifacts": self.artifacts,
        }


def verify_freeze_manifest() -> IntegrityResult:
    """Validate LoRA + H4 artifacts against the authoritative freeze manifest."""
    if not MANIFEST_PATH.is_file():
        return IntegrityResult(
            verified=False,
            status="MANIFEST_MISSING",
            manifest=str(MANIFEST_PATH),
            n_checked=0,
            artifacts=[],
        )

    man = json.loads(MANIFEST_PATH.read_text())
    rows: list[dict[str, Any]] = []
    ok = True
    for a in man.get("artifacts", []):
        if a.get("role") not in {"H4_HEAD", "LORA_CHECKPOINT"}:
            continue
        rel = a["relative_path"]
        p = PROJECT_ROOT / rel
        expected = a["sha256"]
        if not p.is_file():
            rows.append(
                {
                    "path": rel,
                    "role": a.get("role"),
                    "fold": a.get("fold"),
                    "expected_sha256": expected,
                    "actual_sha256": None,
                    "match": False,
                    "status": "MISSING",
                }
            )
            ok = False
            continue
        actual = sha256_file(p)
        match = actual == expected
        ok = ok and match
        rows.append(
            {
                "path": rel,
                "role": a.get("role"),
                "fold": a.get("fold"),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": match,
                "status": "OK" if match else "MISMATCH",
            }
        )

    return IntegrityResult(
        verified=ok and len(rows) == 8,
        status="VERIFIED" if (ok and len(rows) == 8) else "INTEGRITY_FAILED",
        manifest=str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        n_checked=len(rows),
        artifacts=rows,
    )

"""Permanent fal.ai disable guard for ai-image-detector.

Historical fal-generated images may remain archived; no paid fal API calls
are permitted from this repository.
"""

from __future__ import annotations

import os
import sys

FAL_ENV_NAMES = ("FAL_KEY", "FAL_API_KEY")
FAL_BLOCKED_RC = 3
FAL_BLOCKED_MESSAGE = (
    "fal.ai is permanently disabled in this project. "
    "Use Stage 27A V2 public-dataset acquisition only. "
    "Revoke any fal.ai API key in the fal dashboard."
)


def strip_fal_env() -> list[str]:
    """Remove fal credential env vars from the current process (values never logged)."""
    removed: list[str] = []
    for name in FAL_ENV_NAMES:
        if name in os.environ:
            os.environ.pop(name, None)
            removed.append(name)
    return removed


def raise_fal_blocked(caller: str = "") -> None:
    strip_fal_env()
    detail = f" ({caller})" if caller else ""
    raise RuntimeError(f"FAL_BLOCKED{detail}: {FAL_BLOCKED_MESSAGE}")


def block_fal_usage(caller: str = "") -> None:
    """Exit immediately if fal-backed workflow is invoked."""
    strip_fal_env()
    label = f" ({caller})" if caller else ""
    print(f"STOP: fal.ai disabled{label}. {FAL_BLOCKED_MESSAGE}", file=sys.stderr)
    sys.exit(FAL_BLOCKED_RC)


def assert_provider_not_fal(provider: str) -> None:
    p = (provider or "").strip().lower()
    if p in {"fal", "fal.ai", "fal-ai", "@fal-ai/client", "fal_client"}:
        raise_fal_blocked(f"provider={provider}")

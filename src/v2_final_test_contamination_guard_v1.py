"""Fail loudly if NTIRE reserved final-test paths are used before Stage V2-11.

Import and call ``assert_path_not_final_external_test(path)`` from any V2
data-loading or training entrypoint that accepts filesystem paths or dataset IDs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

FORBIDDEN_MARKERS = (
    "NTIRE-RobustAIGenDetection-val",
    "V2_FINAL_EXTERNAL_TEST_V1",
    "deepfakesMSU/NTIRE",
    "deepfakesMSU\\NTIRE",
    "val_images.zip",
    "val_images_hard.zip",
    "val_labels.csv",
    "val_hard_labels.csv",
)

AUTHORIZED_STAGE_TOKEN = "STAGE_V2_11_AUTHORIZED"


class FinalExternalTestContaminationError(RuntimeError):
    """Raised when reserved NTIRE final-test data is touched before V2-11."""


def _haystack(parts: Iterable[str]) -> str:
    return " | ".join(parts).lower()


def path_looks_like_final_external_test(*parts: str) -> bool:
    text = _haystack(str(p) for p in parts)
    return any(marker.lower() in text for marker in FORBIDDEN_MARKERS)


def assert_path_not_final_external_test(
    *parts: str | Path,
    authorization_token: str | None = None,
) -> None:
    """Raise if any argument appears to reference the reserved NTIRE final test.

    Pass ``authorization_token=AUTHORIZED_STAGE_TOKEN`` only from an explicit
    Stage V2-11 entrypoint after human authorisation.
    """
    if authorization_token == AUTHORIZED_STAGE_TOKEN:
        return
    if path_looks_like_final_external_test(*(str(p) for p in parts)):
        raise FinalExternalTestContaminationError(
            "REFUSING to use reserved V2 final external test data "
            "(deepfakesMSU/NTIRE-RobustAIGenDetection-val / "
            "V2_FINAL_EXTERNAL_TEST_V1) before STAGE V2-11. "
            "This dataset must not enter training, validation, calibration, "
            "threshold selection, or architecture selection. "
            "See results/v2_final_test_contamination_guard_v1.json."
        )


if __name__ == "__main__":
    # Self-check: benign path OK; NTIRE path must fail.
    assert_path_not_final_external_test("data/v2/development/example.jpg")
    try:
        assert_path_not_final_external_test(
            "data/v2/NTIRE-RobustAIGenDetection-val/val_labels.csv"
        )
    except FinalExternalTestContaminationError as exc:
        print("PASS contamination guard:", exc)
    else:
        raise SystemExit("FAIL: expected contamination error")

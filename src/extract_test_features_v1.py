"""Extract handcrafted_features_v1 from locked test splits (Stage 11, Task 1).

Why this file exists
--------------------
This is the first authorised access to known_test and unseen_test images.
It reuses the exact feature definitions from extract_handcrafted_features.py.
No model training, scaling, or threshold selection occurs here.

How to run
----------
    source .venv/bin/activate
    python src/extract_test_features_v1.py

What to expect
--------------
    metadata/test_features_v1.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from extract_handcrafted_features import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    extract_features,
    load_controlled_rgb,
    stop_if,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
TEST_FEATURES_PATH = PROJECT_ROOT / "metadata" / "test_features_v1.csv"

TEST_SPLITS = {"known_test", "unseen_test"}
KNOWN_AI_GENERATORS = {"ADM", "BigGAN", "GLIDE", "SD15"}
UNSEEN_AI_GENERATORS = {"Midjourney", "VQDM", "Wukong"}


def load_feature_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    stop_if(len(names) != 13, f"expected 13 feature names, found {len(names)}")
    stop_if(names != FEATURE_COLUMNS, "feature list does not match training feature order")
    return names


def validate_test_table(features: pd.DataFrame, feature_names: list[str]) -> None:
    stop_if(len(features) != 2168, f"expected 2168 test rows, found {len(features)}")

    known = features[features["split"] == "known_test"]
    unseen = features[features["split"] == "unseen_test"]
    stop_if(len(known) != 456, f"known_test rows = {len(known)}, expected 456")
    stop_if(len(unseen) != 1712, f"unseen_test rows = {len(unseen)}, expected 1712")

    stop_if(int((known["label"] == 0).sum()) != 228, "known_test Real count != 228")
    stop_if(int((known["label"] == 1).sum()) != 228, "known_test AI count != 228")
    stop_if(int((unseen["label"] == 0).sum()) != 856, "unseen_test Real count != 856")
    stop_if(int((unseen["label"] == 1).sum()) != 856, "unseen_test AI count != 856")

    for generator in KNOWN_AI_GENERATORS:
        count = int(((known["generator"] == generator) & (known["label"] == 1)).sum())
        stop_if(count != 57, f"known_test {generator} AI count = {count}, expected 57")

    unseen_ai_counts = {
        "Midjourney": 286,
        "VQDM": 285,
        "Wukong": 285,
    }
    for generator, expected in unseen_ai_counts.items():
        count = int(((unseen["generator"] == generator) & (unseen["label"] == 1)).sum())
        stop_if(count != expected, f"unseen_test {generator} AI count = {count}, expected {expected}")

    stop_if(set(features["split"].unique()) - TEST_SPLITS, "unexpected splits in test table")
    stop_if(features["image_id"].isna().any(), "missing image IDs")
    stop_if(features["image_id"].duplicated().any(), "duplicate image IDs")

    block = features[feature_names]
    stop_if(block.isna().any().any(), "NaN values in features")
    stop_if(~np.isfinite(block.to_numpy(dtype=float)).all(), "non-finite feature values")
    stop_if(list(feature_names) != FEATURE_COLUMNS, "feature names/order mismatch")


def main() -> None:
    feature_names = load_feature_names(FEATURE_LIST_PATH)

    meta = pd.read_csv(SPLIT_META_PATH)
    test_rows = meta[meta["split"].isin(TEST_SPLITS)].copy()
    stop_if(len(test_rows) != 2168, f"expected 2168 test metadata rows, found {len(test_rows)}")

    rows = []
    for _, row in tqdm(test_rows.iterrows(), total=len(test_rows), desc="Extracting test features"):
        stop_if(row["split"] not in TEST_SPLITS, f"unexpected split {row['split']}")
        rgb_u8 = load_controlled_rgb(PROJECT_ROOT / row["processed_path"])
        values = extract_features(rgb_u8)
        record = {
            "image_id": row["image_id"],
            "processed_path": row["processed_path"],
            "label": int(row["label"]),
            "generator": row["generator"],
            "split": row["split"],
            "feature_version": FEATURE_VERSION,
        }
        record.update(values)
        rows.append(record)

    features = pd.DataFrame(rows)
    validate_test_table(features, feature_names)
    TEST_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(TEST_FEATURES_PATH, index=False)

    print("Test features extracted")
    print(features["split"].value_counts().to_string())
    print(f"Wrote {TEST_FEATURES_PATH}")


if __name__ == "__main__":
    main()

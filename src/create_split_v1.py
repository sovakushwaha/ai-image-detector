"""Create locked generator-aware splits for protocol v1.

Why this file exists
--------------------
The research question needs two kinds of evaluation:
- known generators (seen during training)
- unseen generators (never seen during training)

This script assigns every image to exactly one split. The assignment is
based on stable image_id values, so RAW and controlled_v1 can reuse the
same split file later.

This script does not extract features or train a model.
known_test and unseen_test must not be used for development decisions.

How to run
----------
    source .venv/bin/activate
    python src/create_split_v1.py

What to expect
--------------
    metadata/split_assignments_v1.csv
    metadata/controlled_v1_split_metadata.csv
    results/split_v1_report.txt
    figures/split_distribution_v1.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- named constants ---
RANDOM_SEED = 42
SPLIT_PROTOCOL = "generator_protocol_v1"

KNOWN_GENERATORS = [
    "ADM",
    "BigGAN",
    "GLIDE",
    "SD15",
]
UNSEEN_GENERATORS = [
    "Midjourney",
    "VQDM",
    "Wukong",
]

KNOWN_AI_COUNTS = {"train": 172, "validation": 57, "known_test": 57}
REAL_COUNTS = {
    "train": 688,
    "validation": 228,
    "known_test": 228,
    "unseen_test": 856,
}

EXPECTED_SPLIT_TOTALS = {
    "train": {"real": 688, "ai": 688, "total": 1376},
    "validation": {"real": 228, "ai": 228, "total": 456},
    "known_test": {"real": 228, "ai": 228, "total": 456},
    "unseen_test": {"real": 856, "ai": 856, "total": 1712},
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLED_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_metadata.csv"
SPLIT_PATH = PROJECT_ROOT / "metadata" / "split_assignments_v1.csv"
MERGED_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "split_v1_report.txt"
FIGURE_PATH = PROJECT_ROOT / "figures" / "split_distribution_v1.png"

SPLIT_ORDER = ["train", "validation", "known_test", "unseen_test"]
GENERATOR_ORDER = ["Real"] + KNOWN_GENERATORS + UNSEEN_GENERATORS


def assign_known_ai(group: pd.DataFrame) -> pd.Series:
    """Shuffle one known generator and cut into train/val/known_test."""
    if len(group) != 286:
        raise SystemExit(
            f"STOP: expected 286 images for {group['generator'].iloc[0]}, found {len(group)}"
        )
    rng = np.random.RandomState(RANDOM_SEED)
    ordered = group.sort_values("image_id").reset_index(drop=True)
    perm = rng.permutation(len(ordered))
    shuffled = ordered.iloc[perm].reset_index(drop=True)
    n_train = KNOWN_AI_COUNTS["train"]
    n_val = KNOWN_AI_COUNTS["validation"]
    splits = (
        ["train"] * n_train
        + ["validation"] * n_val
        + ["known_test"] * KNOWN_AI_COUNTS["known_test"]
    )
    return pd.Series(splits, index=shuffled.index, name="split"), shuffled


def assign_real(group: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if len(group) != 2000:
        raise SystemExit(f"STOP: expected 2000 Real images, found {len(group)}")
    rng = np.random.RandomState(RANDOM_SEED)
    ordered = group.sort_values("image_id").reset_index(drop=True)
    perm = rng.permutation(len(ordered))
    shuffled = ordered.iloc[perm].reset_index(drop=True)
    n_train = REAL_COUNTS["train"]
    n_val = REAL_COUNTS["validation"]
    n_known = REAL_COUNTS["known_test"]
    splits = (
        ["train"] * n_train
        + ["validation"] * n_val
        + ["known_test"] * n_known
        + ["unseen_test"] * REAL_COUNTS["unseen_test"]
    )
    return pd.Series(splits, name="split"), shuffled


def build_assignments(meta: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for generator in KNOWN_GENERATORS:
        group = meta[meta["generator"] == generator].copy()
        splits, shuffled = assign_known_ai(group)
        shuffled = shuffled.copy()
        shuffled["split"] = splits.to_numpy()
        parts.append(shuffled)

    unseen = meta[meta["generator"].isin(UNSEEN_GENERATORS)].copy()
    unseen["split"] = "unseen_test"
    parts.append(unseen)

    real = meta[meta["generator"] == "Real"].copy()
    splits, shuffled = assign_real(real)
    shuffled = shuffled.copy()
    shuffled["split"] = splits.to_numpy()
    parts.append(shuffled)

    assigned = pd.concat(parts, ignore_index=True)
    assigned["random_seed"] = RANDOM_SEED
    assigned["split_protocol"] = SPLIT_PROTOCOL
    return assigned.sort_values("image_id").reset_index(drop=True)


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def validate_assignments(assigned: pd.DataFrame, meta: pd.DataFrame) -> list[str]:
    """Return passed assertion names. Stop on the first failure."""
    passed = []

    stop_if(len(assigned) != 4000, f"expected 4000 assigned records, found {len(assigned)}")
    passed.append("exactly 4000 assigned records")

    stop_if(assigned["image_id"].duplicated().any(), "duplicated image IDs in assignments")
    passed.append("no duplicated image IDs")

    stop_if(assigned["split"].isna().any(), "some images have a missing split")
    stop_if(set(assigned["split"]) - set(SPLIT_ORDER), "unexpected split names")
    passed.append("every image has exactly one split")

    for split_name, expected in EXPECTED_SPLIT_TOTALS.items():
        subset = assigned[assigned["split"] == split_name]
        n_real = int((subset["label"] == 0).sum())
        n_ai = int((subset["label"] == 1).sum())
        stop_if(
            n_real != expected["real"] or n_ai != expected["ai"] or len(subset) != expected["total"],
            f"{split_name} counts are Real={n_real}, AI={n_ai}, total={len(subset)}",
        )
    passed.append("class counts match expectations")

    for generator in KNOWN_GENERATORS:
        counts = assigned.loc[assigned["generator"] == generator, "split"].value_counts()
        stop_if(counts.get("train", 0) != 172, f"{generator} train != 172")
        stop_if(counts.get("validation", 0) != 57, f"{generator} validation != 57")
        stop_if(counts.get("known_test", 0) != 57, f"{generator} known_test != 57")
        stop_if("unseen_test" in counts, f"{generator} leaked into unseen_test")
    unseen_expected = {"Midjourney": 286, "VQDM": 285, "Wukong": 285}
    for generator, n_expected in unseen_expected.items():
        subset = assigned[assigned["generator"] == generator]
        stop_if(len(subset) != n_expected, f"{generator} count {len(subset)} != {n_expected}")
        stop_if(set(subset["split"]) != {"unseen_test"}, f"{generator} is not only in unseen_test")
    passed.append("generator counts match expectations")

    unseen_rows = assigned[assigned["generator"].isin(UNSEEN_GENERATORS)]
    stop_if(set(unseen_rows["split"]) != {"unseen_test"}, "unseen generators appear outside unseen_test")
    passed.append("unseen generators occur ONLY in unseen_test")

    known_rows = assigned[assigned["generator"].isin(KNOWN_GENERATORS)]
    stop_if(
        set(known_rows["split"]) - {"train", "validation", "known_test"},
        "known generators appear outside train/validation/known_test",
    )
    passed.append("known generators occur ONLY in train/validation/known_test")

    real_rows = assigned[assigned["label"] == 0]
    ai_rows = assigned[assigned["label"] == 1]
    stop_if((real_rows["generator"] != "Real").any(), "a Real record has a non-Real generator")
    stop_if((ai_rows["generator"] == "Real").any(), "an AI record is marked generator=Real")
    passed.append("no AI generator appears in Real records")

    missing_processed = 0
    for rel in assigned["processed_path"]:
        if not (PROJECT_ROOT / rel).is_file():
            missing_processed += 1
    stop_if(missing_processed > 0, f"{missing_processed} controlled_v1 paths are missing")
    passed.append("no missing paths in controlled_v1")

    hash_split = assigned.groupby("processed_sha256")["split"].nunique()
    overlapping = hash_split[hash_split > 1]
    stop_if(len(overlapping) > 0, f"{len(overlapping)} processed hashes appear in more than one split")
    passed.append("no overlap in processed image hashes between splits")

    stop_if(set(assigned["image_id"]) != set(meta["image_id"]), "assignment image_ids do not match metadata")
    return passed


def save_figure(assigned: pd.DataFrame) -> None:
    table = (
        assigned.groupby(["generator", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=GENERATOR_ORDER, columns=SPLIT_ORDER)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(table.to_numpy(), cmap="Blues")
    ax.set_xticks(range(len(SPLIT_ORDER)))
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_yticks(range(len(GENERATOR_ORDER)))
    ax.set_yticklabels(GENERATOR_ORDER)
    ax.set_title("Generator counts by split (protocol v1)")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = int(table.iloc[i, j])
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Image count")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)


def write_report(assigned: pd.DataFrame, passed: list[str]) -> str:
    lines = [
        "Split protocol v1 report",
        "========================",
        "",
        f"random_seed: {RANDOM_SEED}",
        f"split_protocol: {SPLIT_PROTOCOL}",
        f"known generators: {', '.join(KNOWN_GENERATORS)}",
        f"unseen generators: {', '.join(UNSEEN_GENERATORS)}",
        "",
        "Class counts",
        "------------",
    ]
    for split_name in SPLIT_ORDER:
        subset = assigned[assigned["split"] == split_name]
        n_real = int((subset["label"] == 0).sum())
        n_ai = int((subset["label"] == 1).sum())
        lines.append(f"{split_name}: Real={n_real}, AI={n_ai}, total={len(subset)}")
    lines.append("")
    lines.append("Generator counts per split")
    lines.append("--------------------------")
    table = (
        assigned.groupby(["split", "generator"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SPLIT_ORDER, columns=GENERATOR_ORDER)
    )
    lines.append(table.to_string())
    lines.append("")
    lines.append("SCIENTIFIC PURPOSE")
    lines.append("------------------")
    lines.append("Train: learn model parameters.")
    lines.append(
        "Validation: model/feature/hyperparameter/threshold development. "
        "Do not use test sets for these decisions."
    )
    lines.append(
        "Known test: evaluate held-out images from familiar generators "
        "(ADM, BigGAN, GLIDE, SD15)."
    )
    lines.append(
        "Unseen test: evaluate cross-generator generalisation using generators "
        "never exposed during model development (Midjourney, VQDM, Wukong)."
    )
    lines.append("")
    lines.append("SOURCE LIMITATION")
    lines.append("-----------------")
    lines.append("Source-level independence cannot be guaranteed from official metadata.")
    lines.append("Tiny-GenImage does not provide true source IDs.")
    lines.append("Filename-derived ImageNet-like tokens are not used as official source IDs.")
    lines.append("SHA-256 duplicate screening found no exact duplicates.")
    lines.append("pHash screening found no confirmed near-duplicate copies.")
    lines.append(
        "These checks reduce leakage risk but do not prove complete source independence."
    )
    lines.append("")
    lines.append("RAW vs controlled_v1")
    lines.append("--------------------")
    lines.append(
        "split_assignments_v1.csv is keyed by image_id (and original_filename / raw_path)."
    )
    lines.append(
        "The same assignments can be applied to RAW and controlled_v1. "
        "Those two experiments must differ by preprocessing only, not by sample identity."
    )
    lines.append("")
    lines.append("Validation assertions passed")
    lines.append("----------------------------")
    for item in passed:
        lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> None:
    meta = pd.read_csv(CONTROLLED_META_PATH)
    assigned_full = build_assignments(meta)

    split_table = assigned_full[
        [
            "image_id",
            "original_filename",
            "raw_path",
            "processed_path",
            "label",
            "generator",
            "split",
            "random_seed",
            "split_protocol",
        ]
    ].copy()

    passed = validate_assignments(assigned_full, meta)
    split_table.to_csv(SPLIT_PATH, index=False)

    merged = meta.merge(
        split_table[["image_id", "split", "random_seed", "split_protocol"]],
        on="image_id",
        how="inner",
        validate="one_to_one",
    )
    stop_if(len(merged) != 4000, f"merged metadata has {len(merged)} rows, expected 4000")
    merged.to_csv(MERGED_PATH, index=False)

    save_figure(assigned_full)
    report = write_report(assigned_full, passed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Split protocol v1 locked")
    print(assigned_full.groupby("split")["label"].value_counts().sort_index().to_string())
    print("Unseen generators only in:", sorted(assigned_full.loc[assigned_full["generator"].isin(UNSEEN_GENERATORS), "split"].unique()))
    print("Assertions passed:")
    for item in passed:
        print(" -", item)
    print(f"Wrote {SPLIT_PATH}")
    print(f"Wrote {MERGED_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()

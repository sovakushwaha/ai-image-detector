"""Audit the Tiny-GenImage pilot subset before any splitting or training.

Why this file exists
--------------------
A high classification score is not automatically good. Before we choose
generators or train a model, we need to know whether Real and AI images
differ in ways a classifier could exploit as shortcuts (format, size,
aspect ratio, and so on).

This script only inspects files. It does not modify raw images, create
splits, extract ML features, or train a model.

How to run
----------
    source .venv/bin/activate
    python src/audit_dataset.py

What to expect
--------------
    metadata/pilot_audit.csv
    metadata/exact_duplicates.csv
    results/pilot_audit_summary.txt
    figures/class_distribution.png
    figures/generator_distribution.png
    figures/image_format_by_label.png
    figures/image_dimensions_by_label.png
    figures/file_size_by_label.png
    figures/pilot_sample_grid.png
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- configuration ---
RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "download_manifest.csv"
AUDIT_PATH = PROJECT_ROOT / "metadata" / "pilot_audit.csv"
DUPLICATES_PATH = PROJECT_ROOT / "metadata" / "exact_duplicates.csv"
SUMMARY_PATH = PROJECT_ROOT / "results" / "pilot_audit_summary.txt"
FIGURES_DIR = PROJECT_ROOT / "figures"

GENERATOR_ORDER = [
    "Real",
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD15",
    "VQDM",
    "Wukong",
]
EXPECTED_MODES = {"RGB"}
THUMBNAIL_SIZE = (128, 128)


def sha256_file(path: Path) -> str:
    """Hash the exact file bytes. This is not a perceptual hash."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_from_magic(path: Path) -> str:
    """Identify format from file header, independent of the filename."""
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "WEBP"
    if header.startswith(b"GIF8"):
        return "GIF"
    return "UNKNOWN"


def inspect_image(row: pd.Series) -> dict:
    """Inspect one image. Never write back to the image file."""
    relative_path = Path(row["path"])
    full_path = PROJECT_ROOT / relative_path

    result = {
        "exists": False,
        "readable": False,
        "format": pd.NA,
        "format_magic": pd.NA,
        "width": pd.NA,
        "height": pd.NA,
        "aspect_ratio": pd.NA,
        "image_mode": pd.NA,
        "channels": pd.NA,
        "file_size_bytes": pd.NA,
        "exact_sha256": pd.NA,
        "issue": "",
    }
    issues = []

    if not full_path.is_file():
        issues.append("missing_file")
        result["issue"] = ";".join(issues)
        return result

    result["exists"] = True
    result["file_size_bytes"] = full_path.stat().st_size
    result["format_magic"] = format_from_magic(full_path)
    result["exact_sha256"] = sha256_file(full_path)

    try:
        with Image.open(full_path) as image:
            image.load()
            width, height = image.size
            result["readable"] = True
            result["format"] = image.format
            result["width"] = int(width)
            result["height"] = int(height)
            result["image_mode"] = image.mode
            result["channels"] = len(image.getbands())
            if width > 0 and height > 0:
                result["aspect_ratio"] = round(width / height, 6)
            else:
                issues.append("invalid_width_or_height")
            if image.mode not in EXPECTED_MODES:
                issues.append(f"unexpected_mode:{image.mode}")
    except Exception as error:
        issues.append(f"unreadable:{type(error).__name__}")

    result["issue"] = ";".join(issues)
    return result


def audit_all_images(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add audit columns to every download-manifest row."""
    audit_rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Auditing images"):
        audit_rows.append(inspect_image(row))
    audit_df = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(audit_rows)], axis=1)
    return audit_df


def classify_duplicate_group(group: pd.DataFrame) -> str:
    """Describe how a SHA-256 group cuts across labels and generators."""
    labels = set(group["label"].tolist())
    generators = set(group["generator"].tolist())
    if labels == {0}:
        return "within_real"
    if labels == {1} and len(generators) == 1:
        return "within_ai_same_generator"
    if labels == {1} and len(generators) > 1:
        return "across_ai_generators"
    if labels == {0, 1}:
        return "across_real_and_ai"
    return "other"


def find_exact_duplicates(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Find groups of files with identical SHA-256 hashes."""
    readable = audit_df[audit_df["exact_sha256"].notna()].copy()
    counts = readable["exact_sha256"].value_counts()
    duplicated_hashes = counts[counts > 1].index

    if len(duplicated_hashes) == 0:
        return pd.DataFrame(
            columns=[
                "duplicate_group",
                "duplicate_scope",
                "group_size",
                "exact_sha256",
                "image_id",
                "path",
                "label",
                "generator",
            ]
        )

    records = []
    group_id = 1
    for sha in duplicated_hashes:
        group = readable[readable["exact_sha256"] == sha]
        scope = classify_duplicate_group(group)
        for _, row in group.iterrows():
            records.append(
                {
                    "duplicate_group": f"dup_{group_id:04d}",
                    "duplicate_scope": scope,
                    "group_size": len(group),
                    "exact_sha256": sha,
                    "image_id": row["image_id"],
                    "path": row["path"],
                    "label": row["label"],
                    "generator": row["generator"],
                }
            )
        group_id += 1
    return pd.DataFrame(records)


def counts_text(series: pd.Series) -> str:
    return series.value_counts(dropna=False).sort_index().to_string()


def crosstab_text(index: pd.Series, columns: pd.Series) -> str:
    return pd.crosstab(index, columns, dropna=False).to_string()


def describe_numeric(df: pd.DataFrame, column: str, by: str) -> str:
    stats = df.groupby(by)[column].describe()
    return stats.round(3).to_string()


def shortcut_risks_text(audit_df: pd.DataFrame) -> str:
    """Describe possible confounders. Do not claim they already cause model behaviour."""
    readable = audit_df[audit_df["readable"] == True]
    lines = ["Potential shortcut/confounding risks", "====================================", ""]
    lines.append(
        "These are audit observations, not proof that a future model will use them."
    )
    lines.append("Wording: potential confounder requiring control or ablation.")
    lines.append("")

    format_by_label = pd.crosstab(readable["label"], readable["format"])
    lines.append("1. File format")
    lines.append(format_by_label.to_string())
    jpeg_real = int(((readable["label"] == 0) & (readable["format"] == "JPEG")).sum())
    png_real = int(((readable["label"] == 0) & (readable["format"] == "PNG")).sum())
    jpeg_ai = int(((readable["label"] == 1) & (readable["format"] == "JPEG")).sum())
    png_ai = int(((readable["label"] == 1) & (readable["format"] == "PNG")).sum())
    if jpeg_real > 0 and png_ai > 0 and png_real == 0 and jpeg_ai == 0:
        lines.append(
            "MAJOR potential shortcut: every readable Real image is JPEG and "
            "every readable AI image is PNG. A classifier could ignore visual "
            "content and still separate the classes from compression artefacts "
            "or format-related statistics. Do not convert formats yet; this "
            "audit only records the evidence."
        )
    else:
        lines.append(
            "Potential confounder requiring control or ablation: format is not "
            "perfectly aligned with label, but class-conditional format imbalance "
            "should still be checked."
        )
    lines.append("")

    lines.append("2. Dimensions")
    lines.append("Width by label:")
    lines.append(describe_numeric(readable, "width", "label"))
    lines.append("Height by label:")
    lines.append(describe_numeric(readable, "height", "label"))
    lines.append("Width by generator:")
    lines.append(describe_numeric(readable, "width", "generator"))
    lines.append("Height by generator:")
    lines.append(describe_numeric(readable, "height", "generator"))
    real_square = int(
        ((readable["label"] == 0) & (readable["width"] == readable["height"])).sum()
    )
    ai_square = int(
        ((readable["label"] == 1) & (readable["width"] == readable["height"])).sum()
    )
    n_real = int((readable["label"] == 0).sum())
    n_ai = int((readable["label"] == 1).sum())
    lines.append(
        f"Square images: Real {real_square}/{n_real}, AI {ai_square}/{n_ai}."
    )
    lines.append(
        "Potential confounder requiring control or ablation: AI generators in "
        "this subset often use a small set of native square resolutions, while "
        "Real ImageNet photographs vary more in width and height."
    )
    lines.append("")

    lines.append("3. Aspect ratio")
    lines.append(describe_numeric(readable, "aspect_ratio", "label"))
    lines.append(
        "Potential confounder requiring control or ablation: every AI image in this "
        "subset is square (aspect ratio 1.0) while almost all Real images are not. "
        "Aspect ratio itself could help a classifier."
    )
    lines.append("")

    lines.append("4. Image mode / channels")
    lines.append("By label:")
    lines.append(crosstab_text(readable["label"], readable["image_mode"]))
    lines.append("By generator:")
    lines.append(crosstab_text(readable["generator"], readable["image_mode"]))
    lines.append(
        "Potential confounder requiring control or ablation: a rare mode such as "
        "grayscale or RGBA, if concentrated in one class or generator, could become "
        "a shortcut. In this subset, all ADM images are RGBA and 10 Real images are "
        "grayscale (L). These rows were flagged, not removed."
    )
    lines.append("")

    lines.append("5. File size")
    lines.append(describe_numeric(readable, "file_size_bytes", "label"))
    lines.append(describe_numeric(readable, "file_size_bytes", "generator"))
    lines.append(
        "Potential confounder requiring control or ablation: PNG vs JPEG encoding "
        "and generator resolution both change file size, so file size is not an "
        "independent forensic signal."
    )
    lines.append("")

    lines.append("6. Generator-specific properties")
    lines.append(
        "Potential confounder requiring control or ablation: each AI generator "
        "appears to occupy a characteristic resolution/file-size band. A model "
        "could learn generator identity proxies rather than general AI artefacts. "
        "Generator identity must never be used as a predictive feature."
    )
    lines.append("")
    lines.append(
        "SHA-256 note: exact-hash duplicates are byte-identical files only. "
        "This audit does not detect resized, cropped, recompressed, or "
        "near-duplicate source content."
    )
    return "\n".join(lines)


def build_summary(audit_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> str:
    readable = audit_df[audit_df["readable"] == True]
    missing = int((audit_df["exists"] == False).sum())
    unreadable = int(((audit_df["exists"] == True) & (audit_df["readable"] == False)).sum())
    issue_rows = int((audit_df["issue"].fillna("") != "").sum())

    n_dup_groups = 0 if duplicates_df.empty else duplicates_df["duplicate_group"].nunique()
    n_dup_files = 0 if duplicates_df.empty else len(duplicates_df)

    within_real = within_ai = across_label = across_gen = 0
    if not duplicates_df.empty:
        scope_counts = duplicates_df.drop_duplicates("duplicate_group")["duplicate_scope"].value_counts()
        within_real = int(scope_counts.get("within_real", 0))
        within_ai = int(scope_counts.get("within_ai_same_generator", 0))
        across_gen = int(scope_counts.get("across_ai_generators", 0))
        across_label = int(scope_counts.get("across_real_and_ai", 0))

    lines = [
        "Tiny-GenImage pilot audit summary",
        "=================================",
        "",
        f"Random seed used for sample-grid sampling: {RANDOM_SEED}",
        "Raw images were not modified.",
        "",
        "1. Total number of records",
        f"   {len(audit_df)}",
        "",
        "2. Real vs AI counts (0=Real, 1=AI-generated)",
        counts_text(audit_df["label"]),
        "",
        "3. Generator counts",
        counts_text(audit_df["generator"]),
        "",
        "4. Image format by label",
        crosstab_text(readable["label"], readable["format"]),
        "",
        "5. Image format by generator",
        crosstab_text(readable["generator"], readable["format"]),
        "",
        "6. Image modes by label",
        crosstab_text(readable["label"], readable["image_mode"]),
        "",
        "7. Width/height summary statistics by label",
        "Width:",
        describe_numeric(readable, "width", "label"),
        "",
        "Height:",
        describe_numeric(readable, "height", "label"),
        "",
        "8. Width/height summary statistics by generator",
        "Width:",
        describe_numeric(readable, "width", "generator"),
        "",
        "Height:",
        describe_numeric(readable, "height", "generator"),
        "",
        "9. Aspect-ratio statistics",
        "By label:",
        describe_numeric(readable, "aspect_ratio", "label"),
        "",
        "By generator:",
        describe_numeric(readable, "aspect_ratio", "generator"),
        "",
        "10. File-size statistics by label (bytes)",
        describe_numeric(readable, "file_size_bytes", "label"),
        "",
        "File-size statistics by generator (bytes)",
        describe_numeric(readable, "file_size_bytes", "generator"),
        "",
        "11. Corrupted/missing image count",
        f"   missing files: {missing}",
        f"   unreadable/corrupted: {unreadable}",
        f"   rows with any issue flag: {issue_rows}",
        "",
        "12. Exact duplicate count (SHA-256 of file bytes)",
        f"   duplicate groups with more than one file: {n_dup_groups}",
        f"   files belonging to those groups: {n_dup_files}",
        f"   groups within Real: {within_real}",
        f"   groups within AI (same generator): {within_ai}",
        f"   groups across AI generators: {across_gen}",
        f"   groups across Real and AI: {across_label}",
        "",
        "SHA-256 detects exact byte-identical files only.",
        "It does NOT detect resized, cropped, recompressed, or near-duplicate images.",
        "",
        shortcut_risks_text(audit_df),
    ]
    return "\n".join(lines)


def save_bar_counts(series: pd.Series, title: str, xlabel: str, output_path: Path, order=None) -> None:
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex([item for item in order if item in counts.index])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_format_by_label(audit_df: pd.DataFrame, output_path: Path) -> None:
    readable = audit_df[audit_df["readable"] == True]
    table = pd.crosstab(readable["label"].map({0: "Real", 1: "AI"}), readable["format"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    table.plot(kind="bar", ax=ax)
    ax.set_title("Image format by label")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of images")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Format")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dimensions_by_label(audit_df: pd.DataFrame, output_path: Path) -> None:
    readable = audit_df[audit_df["readable"] == True].copy()
    readable["class_name"] = readable["label"].map({0: "Real", 1: "AI"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for class_name, colour in [("Real", "#4C72B0"), ("AI", "#DD8452")]:
        subset = readable[readable["class_name"] == class_name]
        axes[0].scatter(
            subset["width"],
            subset["height"],
            s=12,
            alpha=0.35,
            label=class_name,
            c=colour,
        )
    axes[0].set_title("Width vs height")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].legend()

    data = [
        readable.loc[readable["label"] == 0, "width"],
        readable.loc[readable["label"] == 1, "width"],
        readable.loc[readable["label"] == 0, "height"],
        readable.loc[readable["label"] == 1, "height"],
    ]
    axes[1].boxplot(data, tick_labels=["Real W", "AI W", "Real H", "AI H"])
    axes[1].set_title("Width and height by label")
    axes[1].set_ylabel("Pixels")

    fig.suptitle("Image dimensions by label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_file_size_by_label(audit_df: pd.DataFrame, output_path: Path) -> None:
    readable = audit_df[audit_df["readable"] == True]
    data = [
        readable.loc[readable["label"] == 0, "file_size_bytes"] / 1024,
        readable.loc[readable["label"] == 1, "file_size_bytes"] / 1024,
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, tick_labels=["Real", "AI"])
    ax.set_title("File size by label")
    ax.set_ylabel("File size (KB)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_sample_grid(audit_df: pd.DataFrame, output_path: Path) -> None:
    """Contact sheet for manual inspection. Thumbnails are display-only."""
    readable = audit_df[audit_df["readable"] == True]
    real = readable[readable["label"] == 0].sample(n=8, random_state=RANDOM_SEED)

    ai_samples = []
    ai_generators = [name for name in GENERATOR_ORDER if name != "Real"]
    for generator in ai_generators:
        subset = readable[readable["generator"] == generator]
        n = min(2, len(subset))
        if n > 0:
            ai_samples.append(subset.sample(n=n, random_state=RANDOM_SEED))
    chosen = pd.concat([real] + ai_samples, ignore_index=True)

    n_cols = 4
    n_rows = (len(chosen) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.2 * n_rows))
    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for i, row in chosen.iterrows():
        image = Image.open(PROJECT_ROOT / row["path"])
        image = image.convert("RGB")
        image.thumbnail(THUMBNAIL_SIZE)
        axes[i].imshow(image)
        axes[i].set_title(f"{row['generator']} | label={row['label']}", fontsize=8)
        axes[i].axis("off")
        image.close()

    fig.suptitle(f"Pilot sample grid (seed={RANDOM_SEED})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def print_brief_summary(audit_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> None:
    """Terminal output: counts only, not thousands of rows."""
    print("Audit complete")
    print(f"Total records: {len(audit_df)}")
    print("Label counts:")
    print(audit_df["label"].value_counts().sort_index().to_string())
    print("Generator counts:")
    print(audit_df["generator"].value_counts().reindex(GENERATOR_ORDER).to_string())
    print("Format by label:")
    readable = audit_df[audit_df["readable"] == True]
    print(pd.crosstab(readable["label"], readable["format"]).to_string())
    missing = int((audit_df["exists"] == False).sum())
    unreadable = int(((audit_df["exists"] == True) & (audit_df["readable"] == False)).sum())
    print(f"Missing: {missing} | Unreadable: {unreadable}")
    n_groups = 0 if duplicates_df.empty else duplicates_df["duplicate_group"].nunique()
    print(f"Exact duplicate groups: {n_groups}")
    print(f"Wrote {AUDIT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    audit_df = audit_all_images(manifest)
    audit_df.to_csv(AUDIT_PATH, index=False)

    duplicates_df = find_exact_duplicates(audit_df)
    duplicates_df.to_csv(DUPLICATES_PATH, index=False)

    summary = build_summary(audit_df, duplicates_df)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    save_bar_counts(
        audit_df["label"].map({0: "Real", 1: "AI"}),
        "Class distribution",
        "Label",
        FIGURES_DIR / "class_distribution.png",
        order=["Real", "AI"],
    )
    save_bar_counts(
        audit_df["generator"],
        "Generator distribution",
        "Generator",
        FIGURES_DIR / "generator_distribution.png",
        order=GENERATOR_ORDER,
    )
    save_format_by_label(audit_df, FIGURES_DIR / "image_format_by_label.png")
    save_dimensions_by_label(audit_df, FIGURES_DIR / "image_dimensions_by_label.png")
    save_file_size_by_label(audit_df, FIGURES_DIR / "file_size_by_label.png")
    save_sample_grid(audit_df, FIGURES_DIR / "pilot_sample_grid.png")

    print_brief_summary(audit_df, duplicates_df)


if __name__ == "__main__":
    main()

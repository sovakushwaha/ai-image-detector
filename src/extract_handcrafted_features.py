"""Extract Baseline Feature Set V1 from controlled_v1 development images.

Why this file exists
--------------------
The first classical ML experiment uses 13 handcrafted numerical features,
not a neural network. This script turns each development image into one
row of numbers.

known_test and unseen_test are locked. This script never opens those
images. It also does not train a model, fit a scaler, or select features.

How to run
----------
    source .venv/bin/activate
    python src/extract_handcrafted_features.py

What to expect
--------------
    metadata/handcrafted_feature_columns_v1.txt
    metadata/development_features_v1.csv
    results/handcrafted_features_v1_report.txt
    figures/handcrafted_feature_distributions_train_v1.png
    figures/handcrafted_feature_correlation_train_v1.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- named constants ---
FEATURE_VERSION = "handcrafted_features_v1"
CANNY_LOW = 100
CANNY_HIGH = 200
LOW_FREQUENCY_RADIUS_FRACTION = 0.10
EXPECTED_SIZE = (224, 224)

FEATURE_COLUMNS = [
    "mean_brightness",
    "std_brightness",
    "mean_red",
    "mean_green",
    "mean_blue",
    "std_red",
    "std_green",
    "std_blue",
    "mean_saturation",
    "edge_density",
    "sharpness",
    "entropy",
    "high_frequency_ratio",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_META_PATH = PROJECT_ROOT / "metadata" / "controlled_v1_split_metadata.csv"
FEATURE_LIST_PATH = PROJECT_ROOT / "metadata" / "handcrafted_feature_columns_v1.txt"
FEATURES_PATH = PROJECT_ROOT / "metadata" / "development_features_v1.csv"
REPORT_PATH = PROJECT_ROOT / "results" / "handcrafted_features_v1_report.txt"
DIST_FIG_PATH = PROJECT_ROOT / "figures" / "handcrafted_feature_distributions_train_v1.png"
CORR_FIG_PATH = PROJECT_ROOT / "figures" / "handcrafted_feature_correlation_train_v1.png"


def stop_if(condition: bool, message: str) -> None:
    if condition:
        raise SystemExit(f"STOP: {message}")


def load_controlled_rgb(path: Path) -> np.ndarray:
    """Load a controlled_v1 image and check it is RGB JPEG 224×224."""
    with Image.open(path) as image:
        image.load()
        stop_if(image.format != "JPEG", f"{path} is {image.format}, expected JPEG")
        stop_if(image.mode != "RGB", f"{path} is mode {image.mode}, expected RGB")
        stop_if(image.size != EXPECTED_SIZE, f"{path} is {image.size}, expected {EXPECTED_SIZE}")
        rgb_u8 = np.asarray(image, dtype=np.uint8)
    return rgb_u8


def extract_features(rgb_u8: np.ndarray) -> dict[str, float]:
    """Compute the 13 Baseline Feature Set V1 values from one RGB image.

    Pixel values are scaled to 0.0–1.0 before colour/brightness statistics.
    Saturation uses OpenCV HSV (8-bit S channel / 255).
    Edges use OpenCV Canny with fixed thresholds 100 and 200.
    Sharpness is variance of the Laplacian of 8-bit grayscale.
    Entropy is Shannon entropy of the 256-bin grayscale histogram.
    High-frequency ratio uses a centred FFT power-spectrum mask.
    """
    rgb = rgb_u8.astype(np.float64) / 255.0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    # Rec. 601 luminance. This is pixel brightness, not file size.
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    gray_u8 = np.clip(np.round(luminance * 255.0), 0, 255).astype(np.uint8)

    # OpenCV HSV: H 0–179, S 0–255, V 0–255 for 8-bit images.
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float64) / 255.0

    edges = cv2.Canny(gray_u8, CANNY_LOW, CANNY_HIGH)
    edge_density = float((edges > 0).sum() / edges.size)

    laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
    sharpness = float(laplacian.var())

    hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256))
    probabilities = hist.astype(np.float64) / hist.sum()
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    centred = luminance - luminance.mean()
    spectrum = np.fft.fftshift(np.fft.fft2(centred))
    power = np.abs(spectrum) ** 2
    height, width = luminance.shape
    centre_y = height // 2
    centre_x = width // 2
    radius = LOW_FREQUENCY_RADIUS_FRACTION * min(height, width)
    grid_y, grid_x = np.ogrid[:height, :width]
    low_mask = (grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2 <= radius**2
    total_energy = float(power.sum())
    if total_energy <= 0:
        high_frequency_ratio = 0.0
    else:
        high_energy = float(power[~low_mask].sum())
        high_frequency_ratio = high_energy / total_energy

    return {
        "mean_brightness": float(luminance.mean()),
        "std_brightness": float(luminance.std(ddof=0)),
        "mean_red": float(red.mean()),
        "mean_green": float(green.mean()),
        "mean_blue": float(blue.mean()),
        "std_red": float(red.std(ddof=0)),
        "std_green": float(green.std(ddof=0)),
        "std_blue": float(blue.std(ddof=0)),
        "mean_saturation": float(saturation.mean()),
        "edge_density": edge_density,
        "sharpness": sharpness,
        "entropy": entropy,
        "high_frequency_ratio": high_frequency_ratio,
    }


def cohens_d(real_values: np.ndarray, ai_values: np.ndarray) -> float:
    """Standardised mean difference. Diagnostic only, not a trained model."""
    n_real = len(real_values)
    n_ai = len(ai_values)
    var_real = real_values.var(ddof=1)
    var_ai = ai_values.var(ddof=1)
    pooled = np.sqrt(((n_real - 1) * var_real + (n_ai - 1) * var_ai) / (n_real + n_ai - 2))
    if pooled == 0:
        return 0.0
    return float((ai_values.mean() - real_values.mean()) / pooled)


def validate_feature_table(features: pd.DataFrame) -> list[str]:
    passed = []
    stop_if(len(features) != 1832, f"expected 1832 rows, found {len(features)}")
    passed.append("exactly 1832 rows")

    n_train = int((features["split"] == "train").sum())
    n_val = int((features["split"] == "validation").sum())
    stop_if(n_train != 1376, f"expected 1376 train rows, found {n_train}")
    stop_if(n_val != 456, f"expected 456 validation rows, found {n_val}")
    passed.append("exactly 1376 train rows")
    passed.append("exactly 456 validation rows")

    missing = [name for name in FEATURE_COLUMNS if name not in features.columns]
    stop_if(len(FEATURE_COLUMNS) != 13, "feature list is not 13 names")
    stop_if(missing, f"missing feature columns: {missing}")
    passed.append("exactly 13 predictive feature columns")

    block = features[FEATURE_COLUMNS]
    stop_if(block.isna().any().any(), "NaN values in features")
    passed.append("no NaN feature values")
    stop_if(~np.isfinite(block.to_numpy(dtype=float)).all(), "non-finite feature values")
    passed.append("no infinite feature values")

    stop_if(features["image_id"].isna().any(), "missing image IDs")
    passed.append("no missing image IDs")
    stop_if(features["image_id"].duplicated().any(), "duplicate image IDs")
    passed.append("no duplicate image IDs")

    stop_if((features["split"] == "known_test").any(), "known_test rows were extracted")
    stop_if((features["split"] == "unseen_test").any(), "unseen_test rows were extracted")
    passed.append("no known_test rows")
    passed.append("no unseen_test rows")

    stop_if(set(features["label"].unique()) - {0, 1}, "labels are not only 0/1")
    passed.append("labels remain 0/1")

    train = features[features["split"] == "train"]
    val = features[features["split"] == "validation"]
    stop_if(
        int((train["label"] == 0).sum()) != 688 or int((train["label"] == 1).sum()) != 688,
        "train class counts are not 688/688",
    )
    stop_if(
        int((val["label"] == 0).sum()) != 228 or int((val["label"] == 1).sum()) != 228,
        "validation class counts are not 228/228",
    )
    passed.append("expected class counts remain correct")
    return passed


def write_report(features: pd.DataFrame, passed: list[str]) -> str:
    lines = [
        "Handcrafted features v1 report",
        "==============================",
        "",
        f"feature_version: {FEATURE_VERSION}",
        f"CANNY_LOW = {CANNY_LOW}",
        f"CANNY_HIGH = {CANNY_HIGH}",
        f"LOW_FREQUENCY_RADIUS_FRACTION = {LOW_FREQUENCY_RADIUS_FRACTION}",
        "",
        "Saturation: OpenCV RGB→HSV, mean of S/255.",
        "Sharpness: variance of Laplacian(grayscale uint8).",
        "Entropy: Shannon entropy of 256-bin grayscale histogram, log2, zero bins ignored.",
        "High-frequency ratio: mean-centred luminance FFT power outside a centred",
        "circle of radius 0.10 × min(H, W).",
        "",
        f"rows: {len(features)}",
        f"train: {int((features['split'] == 'train').sum())}",
        f"validation: {int((features['split'] == 'validation').sum())}",
        "",
        "Assertions passed",
        "-----------------",
    ]
    for item in passed:
        lines.append(f"- {item}")

    for split_name in ["train", "validation"]:
        subset = features[features["split"] == split_name]
        lines.append("")
        lines.append(f"{split_name.upper()} descriptive statistics")
        lines.append("-" * 40)
        stats = subset[FEATURE_COLUMNS].agg(["min", "max", "mean", "std", "median"]).T
        lines.append(stats.round(6).to_string())

    lines.append("")
    lines.append("Constant / nearly-constant check (TRAIN)")
    lines.append("----------------------------------------")
    train = features[features["split"] == "train"]
    for name in FEATURE_COLUMNS:
        values = train[name].to_numpy(dtype=float)
        n_unique = np.unique(values).size
        std = float(values.std(ddof=0))
        span = float(values.max() - values.min())
        nearly = n_unique == 1 or (span > 0 and std / span < 1e-6) or span == 0
        flag = "NEARLY CONSTANT" if nearly else "varies"
        lines.append(f"{name}: unique={n_unique}, std={std:.6g}, range={span:.6g} [{flag}]")

    lines.append("")
    lines.append("TRAIN Real vs AI separation (diagnostic, not a trained model)")
    lines.append("-------------------------------------------------------------")
    real = train[train["label"] == 0]
    ai = train[train["label"] == 1]
    for name in FEATURE_COLUMNS:
        real_v = real[name].to_numpy(dtype=float)
        ai_v = ai[name].to_numpy(dtype=float)
        d = cohens_d(real_v, ai_v)
        real_q = np.quantile(real_v, [0.05, 0.95])
        ai_q = np.quantile(ai_v, [0.05, 0.95])
        disjoint = real_q[1] < ai_q[0] or ai_q[1] < real_q[0]
        note = ""
        if abs(d) >= 1.0 or disjoint:
            note = "  Potentially strong class-correlated feature"
        lines.append(
            f"{name}: Real mean={real_v.mean():.4g}, AI mean={ai_v.mean():.4g}, "
            f"Cohen_d={d:.3f}{note}"
        )
    lines.append("")
    lines.append(
        "A large Real/AI difference on controlled_v1 is not proof of an AI "
        "forensic signature. It may still reflect residual JPEG traces, "
        "generator artefacts, content differences, or resampling artefacts."
    )
    return "\n".join(lines)


def save_distribution_figure(train: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    axes = axes.flatten()
    real = train[train["label"] == 0]
    ai = train[train["label"] == 1]
    for i, name in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        ax.hist(real[name], bins=30, alpha=0.55, label="Real", color="#4C72B0")
        ax.hist(ai[name], bins=30, alpha=0.55, label="AI", color="#DD8452")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    for ax in axes[len(FEATURE_COLUMNS) :]:
        ax.axis("off")
    fig.suptitle("Train feature distributions: Real vs AI (handcrafted_features_v1)")
    fig.tight_layout()
    DIST_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DIST_FIG_PATH, dpi=150)
    plt.close(fig)


def save_correlation_figure(train: pd.DataFrame) -> None:
    corr = train[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=75, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURE_COLUMNS, fontsize=8)
    ax.set_title("Train feature correlation (diagnostic, no selection)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CORR_FIG_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    FEATURE_LIST_PATH.write_text("\n".join(FEATURE_COLUMNS) + "\n", encoding="utf-8")

    meta = pd.read_csv(SPLIT_META_PATH)
    development = meta[meta["split"].isin(["train", "validation"])].copy()
    stop_if(
        development["split"].isin(["known_test", "unseen_test"]).any(),
        "test rows leaked into the development table before extraction",
    )
    stop_if(len(development) != 1832, f"expected 1832 development rows, found {len(development)}")

    rows = []
    for _, row in tqdm(development.iterrows(), total=len(development), desc="Extracting features"):
        stop_if(row["split"] in {"known_test", "unseen_test"}, "attempted to open a locked test image")
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
    passed = validate_feature_table(features)
    features.to_csv(FEATURES_PATH, index=False)

    train = features[features["split"] == "train"]
    save_distribution_figure(train)
    save_correlation_figure(train)

    report = write_report(features, passed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Handcrafted features v1 extracted")
    print(f"rows: {len(features)}")
    print(features["split"].value_counts().to_string())
    print("Assertions passed:")
    for item in passed:
        print(" -", item)
    print(f"Wrote {FEATURES_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

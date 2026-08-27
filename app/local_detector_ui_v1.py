"""Stage 28B/28C — local research upload UI for FINAL_RESEARCH_MODEL_V1.

Reuses Stage 28A FinalImageDetectorV1 as the single inference source of truth.
Stage 28C adds plain-language display interpretation only.

No training, recalibration, APIs, or automatic upload persistence.
UI interpretation bands (0.45 / 0.55) do NOT alter selective decisions.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from final_inference_v1 import FinalImageDetectorV1, InferenceResultV1  # noqa: E402

# UI-only directional explanation bands inside the existing UNCERTAIN class.
# These do NOT alter model decision, selective policy, or scientific results.
UI_LEAN_REAL_MAX = 0.45
UI_LEAN_AI_MIN = 0.55

ACCEPT_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]

PRIVACY_NOTE = (
    "Images are processed locally on this machine. The interface does not upload "
    "images to external APIs or automatically save them."
)


@dataclass(frozen=True)
class DisplayInterpretationV1:
    """Human-facing labels for an unchanged scientific selective decision."""

    selective_prediction: str
    display_label: str
    display_code: str
    explanation: str
    score_meaning: str


def interpret_display(
    selective_prediction: str,
    calibrated_p: float,
    real_boundary: float,
    ai_boundary: float,
) -> DisplayInterpretationV1:
    """Map frozen selective decision + score to UI-only wording.

    UI-only directional explanation; does not alter model decision.
    """
    if selective_prediction == "REAL":
        return DisplayInterpretationV1(
            selective_prediction="REAL",
            display_label="Likely Real",
            display_code="LIKELY_REAL",
            explanation=(
                "The model's score is below the Real decision boundary. This means the image "
                "appears more consistent with the Real examples learned by the detector."
            ),
            score_meaning="The model found stronger evidence consistent with a real image.",
        )
    if selective_prediction == "AI-GENERATED":
        return DisplayInterpretationV1(
            selective_prediction="AI-GENERATED",
            display_label="Likely AI-generated",
            display_code="LIKELY_AI_GENERATED",
            explanation=(
                "The model's score is above the AI-generated decision boundary. This means "
                "the image contains patterns the detector associates more strongly with "
                "AI-generated imagery."
            ),
            score_meaning=(
                "The model found strong enough evidence to classify the image as likely "
                "AI-generated."
            ),
        )

    # Underlying decision remains UNCERTAIN. Direction bands are display-only.
    if calibrated_p < UI_LEAN_REAL_MAX:
        return DisplayInterpretationV1(
            selective_prediction="UNCERTAIN",
            display_label="Uncertain — leaning Real",
            display_code="UNCERTAIN_LEANING_REAL",
            explanation=(
                "The detector is not confident enough to make a final classification, although "
                "its score leans toward Real."
            ),
            score_meaning="The result is uncertain, but the score leans toward Real.",
        )
    if calibrated_p > UI_LEAN_AI_MIN:
        return DisplayInterpretationV1(
            selective_prediction="UNCERTAIN",
            display_label="Uncertain — leaning AI-generated",
            display_code="UNCERTAIN_LEANING_AI",
            explanation=(
                "The detector is not confident enough to make a final classification, although "
                "its score leans toward AI-generated."
            ),
            score_meaning=(
                "The result is uncertain, but the score leans toward AI-generated."
            ),
        )
    return DisplayInterpretationV1(
        selective_prediction="UNCERTAIN",
        display_label="Uncertain — no clear indication",
        display_code="UNCERTAIN_NEUTRAL",
        explanation=(
            "The detector does not have enough evidence to favour either Real or "
            "AI-generated."
        ),
        score_meaning="The model found no clear direction. The result is uncertain.",
    )


@st.cache_resource(show_spinner="Loading frozen research detector…")
def load_detector(device: str = "auto") -> FinalImageDetectorV1:
    return FinalImageDetectorV1(project_root=ROOT, device=device)


def predict_uploaded_bytes(
    detector: FinalImageDetectorV1,
    data: bytes,
    suffix: str,
) -> InferenceResultV1:
    """Write bytes to a temporary file, run FinalImageDetectorV1, then delete."""
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        return detector.predict(tmp_path, research_controlled_v1=False)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def render_likelihood_scale(
    calibrated_p: float,
    lower: float,
    upper: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 1.55))
    ax.barh([0], [lower], left=0.0, height=0.5, color="#5B8C5A", label="Likely Real")
    ax.barh([0], [upper - lower], left=lower, height=0.5, color="#D4A017", label="Uncertain")
    ax.barh([0], [1.0 - upper], left=upper, height=0.5, color="#C45C26", label="Likely AI-generated")
    ax.axvline(lower, color="#333", lw=1.0, ls="--")
    ax.axvline(upper, color="#333", lw=1.0, ls="--")
    ax.axvline(calibrated_p, color="#111", lw=2.4)
    ax.scatter([calibrated_p], [0], color="#111", s=70, zorder=5)
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xticks([0.0, lower, 0.5, upper, 1.0])
    ax.set_xticklabels(
        [
            "0%",
            f"{100 * lower:.1f}%",
            "50%",
            f"{100 * upper:.1f}%",
            "100%",
        ]
    )
    ax.set_xlabel("AI likelihood score")
    ax.set_title(f"Current score: {100.0 * calibrated_p:.1f}%")
    ax.legend(loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.35))
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def result_json(
    result: InferenceResultV1,
    filename: str,
    display: DisplayInterpretationV1,
) -> str:
    payload = {
        "filename": filename,
        "selective_prediction": result.selective_decision,
        "display_interpretation": display.display_code,
        "display_label": display.display_label,
        "calibrated_probability": result.calibrated_probability,
        "raw_probability": result.raw_probability,
        "model_id": result.model_id,
        "note": (
            "display_interpretation is UI metadata only; selective_prediction is the "
            "authoritative frozen scientific decision."
        ),
        "research_warning": (
            "Research prototype — use this result as an indication, not proof."
        ),
    }
    return json.dumps(payload, indent=2)


def render_primary_card(result: InferenceResultV1, display: DisplayInterpretationV1) -> None:
    pct = 100.0 * result.calibrated_probability
    if display.display_code == "LIKELY_REAL":
        st.success(f"**{display.display_label}**")
    elif display.display_code == "LIKELY_AI_GENERATED":
        # Restrained orange/red tone — not an emergency alert.
        st.markdown(
            f'<div style="padding:0.75rem 1rem;border-radius:0.4rem;'
            f'background:#F8E8E0;border-left:4px solid #C45C26;color:#3A2A22;">'
            f"<strong>{display.display_label}</strong></div>",
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"**{display.display_label}**")

    st.markdown(f"### AI likelihood score: {pct:.1f}%")
    st.caption(
        "This is the model's calibrated AI score, not a guaranteed probability that "
        "the image is AI-generated."
    )
    st.write(display.explanation)
    st.markdown("**What does this mean?**")
    st.write(display.score_meaning)
    st.caption(f"Final model decision: `{result.selective_decision}`")


def main() -> None:
    st.set_page_config(
        page_title="AI-Generated Image Detector — Research Prototype",
        layout="centered",
    )

    st.title("AI-Generated Image Detector — Research Prototype")
    st.caption("Resource-aware AI image detection with uncertainty-aware prediction")

    st.warning("Research prototype — use this result as an indication, not proof.")
    st.caption(
        "Independent testing showed that the detector can perform poorly on newer AI "
        "generators and heavily transformed images."
    )
    st.info(PRIVACY_NOTE)
    st.caption("Image quality matters: strong blur is a known failure mode for this model.")

    with st.expander("Why?"):
        st.write(
            "- Independent modern-generator external ROC-AUC ≈ 0.516 (near chance).\n"
            "- Modern generators remain difficult for this research model.\n"
            "- Severe blur is a documented failure mode.\n"
            "- Wrong high-confidence predictions can still occur."
        )

    with st.expander("How to read this result"):
        st.write(
            "The detector produces an AI likelihood score from 0% to 100%. Scores below "
            "26.4% are reported as Likely Real. Scores above 73.6% are reported as Likely "
            "AI-generated. Scores between these boundaries are marked Uncertain because the "
            "model was designed to avoid forcing a decision when its evidence is not strong "
            "enough."
        )
        st.write(
            "Within the Uncertain region, the interface may show whether the score leans "
            "toward Real or AI-generated. This does not change the underlying model "
            "decision."
        )

    with st.expander("About this research model"):
        st.write("**Model:** MobileNetV3-Small")
        st.write("**Parameters:** 1.52 million")
        st.write("**Purpose:** Research into robust and trustworthy AI-image detection")
        st.write("**Pilot unseen AUC:** approximately 0.850")
        st.write("**Independent modern-generator external AUC:** approximately 0.516")
        st.write(
            "The difference shows why the interface includes an Uncertain result and "
            "strong limitations."
        )

    device_choice = st.selectbox(
        "Device",
        options=["auto", "cpu", "mps", "cuda"],
        index=0,
        help="auto selects CUDA → MPS → CPU",
    )

    try:
        detector = load_detector(device_choice)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model integrity / load failure: {exc}")
        return

    uploaded = st.file_uploader(
        "Choose an image to analyse",
        type=ACCEPT_TYPES,
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.write("Upload a single local image to begin.")
        return

    try:
        image = Image.open(uploaded)
        image.load()
        preview = image.copy()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unreadable or unsupported image: {exc}")
        return

    st.image(preview, caption=uploaded.name, use_container_width=True)
    st.write(
        f"**Filename:** `{uploaded.name}`  \n"
        f"**Original dimensions:** {preview.size[0]} × {preview.size[1]}  \n"
        f"**Format:** {preview.format or Path(uploaded.name).suffix.lstrip('.').upper() or 'unknown'}"
    )
    st.caption("Metadata above is display-only and is not used for classification.")

    col_a, col_b = st.columns(2)
    analyse = col_a.button("Analyse image", type="primary")
    clear = col_b.button("Clear result")

    if clear:
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_filename", None)

    if analyse:
        try:
            data = uploaded.getvalue()
            if not data:
                st.error("Empty upload — please choose a valid image file.")
                return
            suffix = Path(uploaded.name).suffix.lower() or ".jpg"
            result = predict_uploaded_bytes(detector, data, suffix)
            st.session_state["last_result"] = result.to_dict()
            st.session_state["last_filename"] = uploaded.name
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
            return

    if "last_result" not in st.session_state:
        return

    payload = st.session_state["last_result"]
    result = InferenceResultV1(**payload)
    filename = st.session_state.get("last_filename", uploaded.name)
    display = interpret_display(
        result.selective_decision,
        result.calibrated_probability,
        result.real_boundary,
        result.ai_boundary,
    )

    render_primary_card(result, display)
    render_likelihood_scale(
        result.calibrated_probability,
        result.real_boundary,
        result.ai_boundary,
    )

    with st.expander("Technical details"):
        st.write(
            f"**Underlying decision:** `{result.selective_decision}`  \n"
            f"**Display interpretation:** `{display.display_label}` "
            f"(`{display.display_code}` — UI metadata only)"
        )
        st.write(
            "“Uncertain — leaning …” wording is a human-facing explanation of an "
            "underlying UNCERTAIN prediction and does not change the model decision."
        )
        st.write(f"**Model:** {result.model_id}")
        st.write("**Architecture:** MobileNetV3-Small")
        st.write(f"**Parameters:** {detector.n_parameters:,}")
        st.write(f"**Device:** {result.device}")
        st.write(f"**Raw model P(AI):** {result.raw_probability:.6f}")
        st.write(f"**Calibrated model P(AI):** {result.calibrated_probability:.6f}")
        st.write(f"**Temperature:** {result.temperature}")
        st.write(f"**Real boundary:** {result.real_boundary}")
        st.write(f"**AI boundary:** {result.ai_boundary}")
        st.write(
            f"**Historical binary diagnostic threshold:** {result.historical_binary_threshold}"
        )
        st.write(f"**Historical binary diagnostic:** {result.historical_binary_diagnostic}")

    st.download_button(
        "Download result as JSON",
        data=result_json(result, filename, display),
        file_name="local_detector_result_v1.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()

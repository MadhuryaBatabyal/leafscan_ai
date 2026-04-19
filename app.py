"""
LeafScan AI - Leaf Health Detection App
========================================
A Streamlit app that analyzes leaf images through a 4-stage pipeline:
Stage 1: Quality Check (is it a valid leaf?)
Stage 2: Dryness / Water Stress Detection
Stage 3: Disease Detection
Stage 4: Pest Damage Detection

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect Streamlit Cloud
"""

import streamlit as st
from PIL import Image
import io
import numpy as np
from predict import run_pipeline  # Our prediction module

# ── Page configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a clean look ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Header styling */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .app-header h1 { font-size: 2.4rem; font-weight: 700; color: #2e7d32; }
    .app-header p  { color: #555; font-size: 1.05rem; }

    /* Result cards */
    .result-card {
        background: #f9fbe7;
        border-left: 5px solid #8bc34a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .result-card.warning { background: #fff8e1; border-left-color: #ffa000; }
    .result-card.danger  { background: #fce4ec; border-left-color: #e53935; }
    .result-card.info    { background: #e3f2fd; border-left-color: #1e88e5; }

    /* Stage label badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-pass  { background: #c8e6c9; color: #1b5e20; }
    .badge-fail  { background: #ffcdd2; color: #b71c1c; }
    .badge-warn  { background: #fff9c4; color: #7c6500; }
    .badge-info  { background: #e3f2fd; color: #0d47a1; }

    /* Confidence bar container */
    .conf-label { font-size: 0.85rem; color: #555; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🌿 LeafScan AI</h1>
    <p>Upload or capture a leaf photo — get instant health diagnostics powered by deep learning.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Input Section ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📷 Input Image")

    # Choose input method
    input_method = st.radio(
        "How do you want to provide the leaf image?",
        ["Upload a photo", "Use camera (mobile/webcam)"],
        horizontal=True,
    )

    image = None  # Will hold the PIL Image

    if input_method == "Upload a photo":
        uploaded_file = st.file_uploader(
            "Choose a leaf image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            help="For best results: clear, well-lit photo with a single leaf.",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

    else:  # Camera input
        camera_file = st.camera_input(
            "Point at a leaf and take a photo",
            help="Works on mobile and desktops with webcams.",
        )
        if camera_file:
            image = Image.open(camera_file).convert("RGB")

    if image:
        st.image(image, caption="Input leaf image", use_column_width=True)

# ── Analysis Section ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("🔬 Analysis Results")

    if image is None:
        st.info("Waiting for an image… upload or capture a leaf photo on the left.")
        st.markdown("""
        **How it works:**
        1. 🔍 **Stage 1** — checks if the image is a valid leaf
        2. 💧 **Stage 2** — detects dryness / water stress
        3. 🦠 **Stage 3** — identifies leaf diseases
        4. 🐛 **Stage 4** — detects pest damage
        """)

    else:
        # Run the 4-stage pipeline from predict.py
        with st.spinner("Running 4-stage analysis…"):
            results = run_pipeline(image)

        # ── Stage 1: Quality Check ──────────────────────────────────────
        st.markdown("##### Stage 1 — Quality Check")
        if not results["is_valid_leaf"]:
            st.markdown("""
            <div class="result-card danger">
                <span class="badge badge-fail">FAILED</span>
                <strong>Not a valid leaf image.</strong><br>
                Please upload a clear, close-up photo of a single leaf.
            </div>
            """, unsafe_allow_html=True)
            st.stop()  # Stop here — no point running further stages

        st.markdown("""
        <div class="result-card">
            <span class="badge badge-pass">PASS</span>
            Valid leaf image detected.
        </div>
        """, unsafe_allow_html=True)

        # ── Stage 2: Dryness ───────────────────────────────────────────
        st.markdown("##### Stage 2 — Dryness / Water Stress")
        _render_result_card(results["dryness"], label_name="Dryness")

        # ── Stage 3: Disease ───────────────────────────────────────────
        st.markdown("##### Stage 3 — Disease Detection")
        _render_result_card(results["disease"], label_name="Disease")

        # ── Stage 4: Pest Damage ───────────────────────────────────────
        st.markdown("##### Stage 4 — Pest Damage")
        _render_result_card(results["pest"], label_name="Pest")

        # ── Overall Recommendation ─────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 💡 Recommended Action")
        _render_recommendation(results)


# ── Helper UI functions ──────────────────────────────────────────────────────

def _render_result_card(stage_result: dict, label_name: str):
    """
    Renders a result card for a single detection stage.
    stage_result keys: label, confidence, is_uncertain, action
    """
    label      = stage_result["label"]
    confidence = stage_result["confidence"]          # 0.0 – 1.0
    uncertain  = stage_result["is_uncertain"]        # True if conf < threshold
    action     = stage_result["action"]

    # Choose card style based on result
    if uncertain:
        card_class = "info"
        badge_class = "badge-info"
        badge_text = "UNCERTAIN"
    elif label.lower() == "healthy":
        card_class = "result-card"
        badge_class = "badge-pass"
        badge_text = "HEALTHY"
    else:
        card_class = "warning"
        badge_class = "badge-warn"
        badge_text = label.upper()

    # Confidence percentage
    pct = int(confidence * 100)
    conf_color = "#4caf50" if confidence > 0.75 else ("#ff9800" if confidence > 0.50 else "#f44336")

    # Confidence bar using Streamlit's progress
    st.markdown(
        f'<span class="conf-label"><span class="badge {badge_class}">{badge_text}</span>'
        f'Confidence: <strong>{pct}%</strong></span>',
        unsafe_allow_html=True,
    )
    st.progress(confidence)

    if uncertain:
        st.warning("⚠️ Low confidence — please recapture the image in better lighting.")
    else:
        st.markdown(f"**Suggested action:** {action}", unsafe_allow_html=False)

    st.markdown("")  # Spacer


def _render_recommendation(results: dict):
    """Aggregates all stage results into a top-level recommendation."""
    issues = []
    if results["dryness"]["label"] != "healthy" and not results["dryness"]["is_uncertain"]:
        issues.append(("💧 Water stress", results["dryness"]["action"]))
    if results["disease"]["label"] != "healthy" and not results["disease"]["is_uncertain"]:
        issues.append(("🦠 Disease", results["disease"]["action"]))
    if results["pest"]["label"] != "healthy" and not results["pest"]["is_uncertain"]:
        issues.append(("🐛 Pest damage", results["pest"]["action"]))

    if not issues:
        st.success("✅ Leaf appears healthy! Continue regular watering and monitoring.")
    else:
        for issue, action in issues:
            st.markdown(f"**{issue}:** {action}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.82rem;'>"
    "LeafScan AI · Built with Streamlit + PyTorch · College Project Demo"
    "</p>",
    unsafe_allow_html=True,
)

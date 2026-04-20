"""
app.py — LeafScan AI Streamlit Web App
========================================
Run locally:   streamlit run app.py
Deploy:        Push to GitHub → connect at share.streamlit.io

Folder structure required:
  leafscan/
  ├── app.py          ← this file
  ├── predict.py      ← pipeline logic
  └── models/
      ├── stage1_quality.pth
      ├── stage2_dryness.pth
      ├── stage3_disease.pth
      └── stage4_pest.pth
"""

import streamlit as st
from PIL import Image
from predict import run_pipeline

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LeafScan AI",
    page_icon="🌿",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stage-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 1.2rem 0 0.3rem;
}
.action-box {
    background: #f1f8e9;
    border-left: 4px solid #7cb342;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.93rem;
    margin-top: 0.3rem;
}
.action-box.warn  { background:#fff8e1; border-left-color:#f9a825; }
.action-box.alert { background:#fce4ec; border-left-color:#e53935; }
.action-box.grey  { background:#f5f5f5; border-left-color:#bdbdbd; }
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 6px;
    vertical-align: middle;
}
.pill-green  { background:#c8e6c9; color:#1b5e20; }
.pill-yellow { background:#fff9c4; color:#7c6500; }
.pill-red    { background:#ffcdd2; color:#b71c1c; }
.pill-grey   { background:#e0e0e0; color:#424242; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("# 🌿 LeafScan AI")
st.markdown("Upload or photograph a leaf — get an instant 4-stage health report.")
st.divider()

# ── Input ──────────────────────────────────────────────────────────────────
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📷 Leaf Image")
    mode = st.radio("Input method", ["Upload photo", "Camera"], horizontal=True)

    image = None
    if mode == "Upload photo":
        f = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
        if f:
            image = Image.open(f).convert("RGB")
    else:
        f = st.camera_input("Take a photo of a leaf")
        if f:
            image = Image.open(f).convert("RGB")

    if image:
        st.image(image, use_column_width=True, caption="Input image")

# ── Results ────────────────────────────────────────────────────────────────
with col_results:
    st.subheader("🔬 Diagnosis")

    if image is None:
        st.info("Waiting for an image…")
        st.markdown("""
        The pipeline runs four checks in order:
        1. **Stage 1** — Is this a valid leaf photo?
        2. **Stage 2** — Any water stress / dryness?
        3. **Stage 3** — Any disease?
        4. **Stage 4** — Any pest damage?
        """)

    else:
        with st.spinner("Analysing…"):
            results = run_pipeline(image)

        # ── Stage 1 result ────────────────────────────────────────────────
        st.markdown('<p class="stage-header">Stage 1 — Quality check</p>', unsafe_allow_html=True)

        if not results["is_valid_leaf"]:
            leaf_pct = int(results["leaf_confidence"] * 100)
            st.markdown(
                f'<span class="pill pill-red">REJECTED</span> '
                f'Not recognised as a leaf (leaf confidence: {leaf_pct}%)',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="action-box alert">'
                'Please retake the photo: fill the frame with a single leaf, '
                'good lighting, camera in focus.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        leaf_pct = int(results["leaf_confidence"] * 100)
        st.markdown(
            f'<span class="pill pill-green">PASS</span> Valid leaf ({leaf_pct}% confidence)',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Stages 2–4 ────────────────────────────────────────────────────
        for stage_key, stage_title, stage_icon in [
            ("dryness", "Stage 2 — Dryness / water stress", "💧"),
            ("disease", "Stage 3 — Disease detection",      "🦠"),
            ("pest",    "Stage 4 — Pest damage",            "🐛"),
        ]:
            r = results[stage_key]
            label      = r["label"]
            confidence = r["confidence"]
            uncertain  = r["is_uncertain"]
            action     = r["action"]
            pct        = int(confidence * 100)

            st.markdown(
                f'<p class="stage-header">{stage_icon} {stage_title}</p>',
                unsafe_allow_html=True,
            )

            if uncertain:
                st.markdown(
                    f'<span class="pill pill-grey">UNCERTAIN</span> '
                    f'Confidence too low ({pct}%) — please recapture',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="action-box grey">'
                    'Retake in better lighting with the leaf filling the frame.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            elif label == "healthy":
                st.markdown(
                    f'<span class="pill pill-green">HEALTHY</span> {pct}% confidence',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="action-box">{action}</div>',
                    unsafe_allow_html=True,
                )
            else:
                # Determine severity for colour — severe/late = red, else yellow
                is_severe = any(word in label for word in ("severe", "late", "mosaic"))
                pill_cls  = "pill-red"    if is_severe else "pill-yellow"
                box_cls   = "alert"       if is_severe else "warn"
                label_display = label.replace("_", " ").title()
                st.markdown(
                    f'<span class="pill {pill_cls}">{label_display.upper()}</span> '
                    f'{pct}% confidence',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="action-box {box_cls}">{action}</div>',
                    unsafe_allow_html=True,
                )

            # Confidence bar
            st.progress(confidence)

        # ── Overall summary ────────────────────────────────────────────────
        st.divider()
        issues = [
            k for k in ("dryness", "disease", "pest")
            if results[k]["label"] != "healthy" and not results[k]["is_uncertain"]
        ]
        if not issues:
            st.success("✅ Leaf looks healthy across all checks. Keep monitoring regularly.")
        else:
            issue_str = ", ".join(i.title() for i in issues)
            st.warning(f"⚠️ Issues detected: **{issue_str}**. Follow the recommended actions above.")

# ── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:0.8rem;'>"
    "LeafScan AI · College Project · Built with Streamlit + PyTorch"
    "</p>",
    unsafe_allow_html=True,
)

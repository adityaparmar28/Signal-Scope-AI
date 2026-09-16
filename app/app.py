"""
SignalScope Web Interface (Bonus Module F)
Interactive Media Forensics Dashboard built with Streamlit.
Features:
- Single image drag-and-drop inspection
- Layer-CAM anomaly heatmap visualization (Headline Bonus A)
- Provenance & EXIF Metadata Inspector (Bonus D)
- On-the-fly Robustness & Degradation stress-test (Bonus C)
- Batch scanning with CSV export
"""

import os
import sys
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure model and src are on path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import importlib
import model.predict
importlib.reload(model.predict)
from model.predict import predict
from src.degradation_benchmark import apply_degradation


st.set_page_config(
    page_title="SignalScope — Media Forensics & Authenticity",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1.05rem; color: #94A3B8; margin-bottom: 1.5rem; }
    .verdict-card { padding: 1.2rem; border-radius: 10px; margin-bottom: 1rem; border-left: 6px solid #CBD5E1; }
    .verdict-ai { background-color: rgba(239, 68, 68, 0.15); border-left-color: #EF4444; }
    .verdict-real { background-color: rgba(34, 197, 94, 0.15); border-left-color: #22C55E; }
    .verdict-inconclusive { background-color: rgba(245, 158, 11, 0.15); border-left-color: #F59E0B; }
</style>
""", unsafe_allow_html=True)

st.sidebar.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg", width=50)
st.sidebar.title("SignalScope")
st.sidebar.caption("Telling Real From Synthetic in the Age of Generative Media")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Navigation", [
    "Single Image Forensics",
    "Batch Image Scanner",
    "Degradation Stress-Test",
    "System Architecture & Specs"
])

st.sidebar.markdown("---")
st.sidebar.info("""
**SIH 2026 Internal Hackathon**  
*Problem Statement 2: SignalScope*  
• Dual-Branch Spatial + Spectral Model  
• Layer-CAM Grounded Explainability  
• C2PA & EXIF Provenance Analysis  
""")


# ==============================================================================
# MODE 1: SINGLE IMAGE FORENSICS
# ==============================================================================
if mode == "Single Image Forensics":
    st.markdown('<div class="main-title">Check If Image Is Real Or AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload any photo to check if it was made by AI or taken by a real camera.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Drop image here (JPEG, PNG, WebP)...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Image")
            pil_img = Image.open(tmp_path)
            st.image(pil_img, use_container_width=True, caption=f"Dimensions: {pil_img.size[0]} × {pil_img.size[1]} px")

        with st.spinner("Checking if this image is real or AI-made..."):
            results = predict(tmp_path, explain=True)

        with col2:
            st.subheader("Result")

            verdict = results["verdict"]
            css_class = "verdict-inconclusive"
            badge_icon = "⚠️"
            if verdict == "Likely AI-Generated":
                css_class = "verdict-ai"
                badge_icon = "🤖"
            elif verdict == "Likely Real Photo":
                css_class = "verdict-real"
                badge_icon = "📷"

            conf_pct = results["calibrated_confidence"] * 100
            st.markdown(f"""
            <div class="verdict-card {css_class}">
                <h3 style="margin:0; padding-bottom: 0.3rem;">{badge_icon} {verdict}</h3>
                <p style="margin:0; font-size: 1.1rem;"><strong>Confidence Score:</strong> {conf_pct:.1f}%</p>
                <p style="margin:0.4rem 0 0 0; font-size: 0.95rem;">{results['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(results["calibrated_confidence"])

            st.markdown("#### File Info & Metadata")
            st.info(results["metadata_signal"])

        st.markdown("---")
        st.subheader("Why We Think This")
        st.caption("The heatmap shows which parts of the image look AI-generated. Red/yellow areas = suspicious regions.")

        if "explanation" in results:
            exp = results["explanation"]
            exp_col1, exp_col2 = st.columns([1, 1])

            with exp_col1:
                st.image(exp["heatmap_path"], use_container_width=True, caption="Heatmap — Red/Yellow areas look AI-generated")

            with exp_col2:
                st.markdown("**What looks suspicious:**")
                for cue in exp["cues"]:
                    st.markdown(f"• **{cue}**")

                st.markdown("**Explanation:**")
                st.write(exp["summary"])
                st.metric(label="Suspicious Regions Found", value=exp["anomalous_regions_count"])

        try:
            os.remove(tmp_path)
        except Exception:
            pass


# ==============================================================================
# MODE 2: BATCH IMAGE SCANNER
# ==============================================================================
elif mode == "Batch Image Scanner":
    st.markdown('<div class="main-title">Scan Multiple Images</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload multiple images to check them all at once.</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"Loaded **{len(uploaded_files)}** images.")
        if st.button("Check All Images"):
            batch_results = []
            progress_bar = st.progress(0)

            for i, uf in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uf.name)[1]) as tmp:
                    tmp.write(uf.getvalue())
                    t_path = tmp.name

                res = predict(t_path, explain=False)
                batch_results.append({
                    "File": uf.name,
                    "Result": res["verdict"],
                    "Confidence": f"{res['calibrated_confidence'] * 100:.1f}%",
                    "AI Score": f"{res['fused_probability_ai'] * 100:.1f}%",
                    "Extra Info": res["metadata_signal"]
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
                try:
                    os.remove(t_path)
                except Exception:
                    pass

            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results (CSV)", csv, "signalscope_results.csv", "text/csv")


# ==============================================================================
# MODE 3: DEGRADATION STRESS-TEST
# ==============================================================================
elif mode == "Degradation Stress-Test":
    st.markdown('<div class="main-title">Robustness & Degradation Stress-Test (Bonus C)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Simulate social media re-compression, screenshotting, and blurring on the fly.</div>', unsafe_allow_html=True)

    up_file = st.file_uploader("Upload an image to stress-test...", type=["jpg", "jpeg", "png"])
    if up_file:
        pil_img = Image.open(up_file).convert("RGB")

        deg_type = st.selectbox("Select Degradation Attack", ["jpeg", "blur", "downscale", "screenshot"])
        severity = st.slider("Attack Severity", min_value=1.0, max_value=3.0, value=1.5, step=0.5)

        degraded_img = apply_degradation(pil_img, deg_type, severity)

        c1, c2 = st.columns(2)
        with c1:
            st.image(pil_img, caption="Original Image", use_container_width=True)
        with c2:
            st.image(degraded_img, caption=f"Degraded Image ({deg_type.upper()}, severity={severity})", use_container_width=True)

        if st.button("Compare Stability"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1, tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
                pil_img.save(f1.name)
                degraded_img.save(f2.name)

                res_orig = predict(f1.name, explain=False)
                res_deg = predict(f2.name, explain=False)

                try:
                    os.remove(f1.name)
                    os.remove(f2.name)
                except Exception:
                    pass

            delta = abs(res_orig["raw_probability_ai"] - res_deg["raw_probability_ai"])
            st.markdown("### Stability Report:")
            st.write(f"• **Original Score:** {res_orig['raw_probability_ai'] * 100:.1f}% ({res_orig['verdict']})")
            st.write(f"• **Degraded Score:** {res_deg['raw_probability_ai'] * 100:.1f}% ({res_deg['verdict']})")
            st.write(f"• **Absolute Δ Confidence:** {delta:.3f}")
            if delta < 0.08:
                st.success("Verdict is Highly Stable under degradation (meets Bonus C criteria)!")
            else:
                st.warning("Moderate degradation variance observed.")


# ==============================================================================
# MODE 4: SYSTEM ARCHITECTURE & SPECS
# ==============================================================================
elif mode == "System Architecture & Specs":
    st.markdown('<div class="main-title">System Architecture & Forensics Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
    SignalScope addresses the unseen-generator generalization challenge via a **Dual-Branch Hybrid Pipeline**:

    1. **Spatial Semantic Branch (Pretrained EfficientNet-B0)**: Analyzes high-level composition, geometric consistency, and lighting vectors.
    2. **Frequency/Residual Branch (SRM High-Pass Filters)**: Analyzes Discrete Cosine Transform (DCT) upsampling grid artifacts that remain constant across generative diffusion backbones.
    3. **Temperature Calibration**: Guarantees that confidence outputs represent true empirical accuracy.
    4. **Layer-CAM Grounded Explainer**: Isolates pixel-level anomalous regions without LLM hallucinations.
    5. **Provenance Fusion Engine**: Cross-references C2PA Content Credentials manifests and physical camera EXIF tags.
    """)
    st.info("Evaluation protocol matches SIH PS-2 Section 4: Tested against held-out unseen generative models.")

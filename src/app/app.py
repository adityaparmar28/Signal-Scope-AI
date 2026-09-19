import streamlit as st
import os
import torch
from src.app.ui.theme import apply_theme

# Page Config must be the first command
st.set_page_config(
    page_title="SignalScope AI Forensics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom Forensic Light theme
apply_theme()

from src.app.utils import ensure_model_loaded
ensure_model_loaded()

# Main Page Content
st.title("Verify Digital Authenticity")
st.markdown("<p class='subtitle'>Analyze images for synthetic generation signals, model evidence, metadata anomalies, and robustness under common transformations.</p>", unsafe_allow_html=True)

st.divider()

st.subheader("System Status")

col1, col2 = st.columns(2)
with col1:
    if st.session_state.model_loaded:
        st.success("✓ Core Model (EfficientNet-B0 + SRM Fusion) Online")
        st.success("✓ Grad-CAM Explainability Engine Online")
        st.success("✓ Calibration Weights Loaded")
    else:
        st.error("⚠ Model checkpoint unavailable. Inference disabled.")
        st.info(f"Please ensure `best_model.pth` exists in: `{MODEL_DIR}`")

with col2:
    st.markdown("""
        **Available Modules:**
        * **Module A**: Faithful Explanation (Grad-CAM + GenAI Text)
        * **Module C**: Robustness to Degradation
        * **Module D**: Provenance & Metadata (EXIF/C2PA extraction)
        * **Module F**: Deployable Interface (This Dashboard)
        * **Module G**: Active Defence Analysis (FGSM Adversarial)
    """)

st.markdown("👈 **Select a tool from the sidebar to begin analysis.**")

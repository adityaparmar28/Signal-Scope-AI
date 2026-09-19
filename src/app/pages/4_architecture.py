import streamlit as st
import os
from src.app.ui.theme import apply_theme

st.set_page_config(page_title="Architecture & Status - SignalScope", layout="wide")
apply_theme()

st.title("System Architecture & Status")
st.markdown("<p class='subtitle'>Technical specifications, evaluation metrics, and system health status for SignalScope.</p>", unsafe_allow_html=True)

# Status Check
st.subheader("Live System Health")
model_loaded = st.session_state.get('model_loaded', False)

if model_loaded:
    st.success("🟢 Core Model Online")
    st.info("✓ Checkpoint loaded successfully from: `model/weights/best_model.pth`")
    st.info("✓ Temperature calibration parameters loaded.")
else:
    st.error("🔴 Core Model Offline")
    st.warning("⚠️ Could not locate or load `best_model.pth`. Inference is disabled.")

st.divider()

# Architecture Diagram
st.subheader("Dual-Branch Forensic Architecture")
st.markdown("""
```mermaid
graph TD
    A[Input Image] --> B(Spatial Branch: EfficientNet-B0)
    A --> C(Frequency Branch: SRM Filters)
    
    B -->|Spatial Semantics<br>1280 features| D[Feature Concatenation]
    C -->|High-Frequency Noise Residuals<br>128 features| D
    
    D --> E[Fusion Block: Dense Layers + BatchNorm]
    E --> F[Classifier]
    F -->|Logits| G[Temperature Calibration]
    G --> H((Forensic Verdict & Confidence))
```
""")

st.divider()

# Metrics
st.subheader("Official Evaluation Metrics")
st.markdown("""
These metrics are based on the held-out test evaluation against unseen generators:

* **Overall ROC-AUC:** 0.9732
* **Unseen-Generator ROC-AUC:** 0.9685
* **Macro-F1:** 0.9015
* **Accuracy:** 90.75%
* **FPR (False Positive Rate) @ 0.50 Threshold:** 4.48%

*Note: The model utilizes a dynamically loaded `LegacyDualBranchNet` adapter to faithfully reconstruct the exact EfficientNet-B0 training graph for inference.*
""")

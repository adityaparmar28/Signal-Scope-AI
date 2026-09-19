import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import cv2
from src.app.ui.theme import apply_theme
from src.model.predict import get_transforms

st.set_page_config(page_title="Robustness Lab - SignalScope", layout="wide")
apply_theme()

st.title("Robustness Lab")
st.markdown("<p class='subtitle'>Test the resilience of the AI detector against common real-world image degradations like JPEG compression and blurring.</p>", unsafe_allow_html=True)

from src.app.utils import ensure_model_loaded
ensure_model_loaded()

if not st.session_state.get('model_loaded', False):
    st.error("Model checkpoint could not be found.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image to stress-test", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
    # 1. Load Image
    image_bytes = uploaded_file.getvalue()
    original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    st.sidebar.header("Degradation Settings")
    degradation_type = st.sidebar.selectbox("Transformation", ["JPEG Compression", "Gaussian Blur", "Downscale"])
    
    severity = 0
    if degradation_type == "JPEG Compression":
        severity = st.sidebar.slider("Quality (Lower is worse)", 10, 100, 70)
    elif degradation_type == "Gaussian Blur":
        severity = st.sidebar.slider("Blur Kernel Size", 3, 31, 7, step=2)
    elif degradation_type == "Downscale":
        severity = st.sidebar.slider("Scale Factor (%)", 10, 100, 50)
        
    if st.button("Run Stress Test", type="primary"):
        # Apply degradation
        img_np = np.array(original_image)
        degraded_np = img_np.copy()
        
        if degradation_type == "JPEG Compression":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), severity]
            _, encimg = cv2.imencode('.jpg', img_np, encode_param)
            degraded_np = cv2.imdecode(encimg, 1)
            degraded_np = cv2.cvtColor(degraded_np, cv2.COLOR_BGR2RGB)
        elif degradation_type == "Gaussian Blur":
            degraded_np = cv2.GaussianBlur(img_np, (severity, severity), 0)
        elif degradation_type == "Downscale":
            h, w = img_np.shape[:2]
            new_h, new_w = int(h * severity / 100), int(w * severity / 100)
            degraded_np = cv2.resize(img_np, (new_w, new_h))
            degraded_np = cv2.resize(degraded_np, (w, h)) # scale back up for model input
            
        degraded_image = Image.fromarray(degraded_np)
        
        # Inference function
        def get_prediction(img):
            model = st.session_state.model
            scaler = st.session_state.scaler
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            transform = get_transforms()
            tensor = transform(image=np.array(img))['image'].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
                logits = out['logits']
                calibrated_logits = scaler.scale(logits)
                probs = torch.softmax(calibrated_logits, dim=1)[0]
            prob_fake = probs[1].item()
            is_fake = prob_fake > 0.5
            confidence = prob_fake if is_fake else (1.0 - prob_fake)
            return is_fake, confidence * 100
            
        orig_is_fake, orig_conf = get_prediction(original_image)
        deg_is_fake, deg_conf = get_prediction(degraded_image)
        
        delta = deg_conf - orig_conf
        
        # Display Results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(original_image, use_container_width=True)
            verdict_str = "AI-Generated" if orig_is_fake else "Real"
            st.metric("Original Verdict", verdict_str, f"{orig_conf:.1f}% confidence", delta_color="off")
            
        with col2:
            st.subheader(f"Degraded ({degradation_type})")
            st.image(degraded_image, use_container_width=True)
            deg_verdict_str = "AI-Generated" if deg_is_fake else "Real"
            
            # Delta logic
            delta_str = f"{delta:+.1f}%"
            if orig_is_fake != deg_is_fake:
                st.error(f"⚠️ VERDICT FLIPPED to {deg_verdict_str}")
                st.metric("Degraded Verdict", deg_verdict_str, delta_str, delta_color="inverse")
            else:
                st.success("✓ VERDICT STABLE")
                st.metric("Degraded Verdict", deg_verdict_str, delta_str, delta_color="normal" if delta > 0 else "inverse")

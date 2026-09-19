import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title='SignalScope', page_icon='🔍', layout='wide')

# Try importing model modules
try:
    import torch
    from src.model.dual_branch_net import DualBranchNet, TemperatureScaler
    from src.model.predict import predict_from_pil, get_transforms
    from src.explain.gradcam import SignalScopeGradCAM
    from src.explain.explainer import generate_full_report
    from src.model.metadata import analyze_metadata
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    import_error = str(e)


@st.cache_resource
def load_model_cached(model_dir):
    """Load and cache the model."""
    device = torch.device('cpu')
    model_path = os.path.join(model_dir, 'best_model.pth')

    from src.model.legacy_model import LegacyDualBranchNet
    model = LegacyDualBranchNet()
    scaler = TemperatureScaler()
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
    else:
        return None, None, None

    model.to(device)
    model.eval()

    scaler = TemperatureScaler()
    temp_path = os.path.join(model_dir, 'temperature.json')
    scaler.load(temp_path)

    gradcam = SignalScopeGradCAM(model, device=device)

    return model, scaler, gradcam


def get_confidence_color(confidence, label):
    if confidence > 0.8:
        return "🔴" if label == 'ai_generated' else "🟢"
    elif confidence > 0.6:
        return "🟠"
    else:
        return "🟡"


# ---- HEADER ----
st.title('🔍 SignalScope')
st.markdown('### AI-Generated Image Detector — Telling Real from Synthetic')
st.markdown('---')

# ---- SIDEBAR ----
st.sidebar.title("ℹ️ About SignalScope")
st.sidebar.markdown("""
**SignalScope** detects AI-generated images using a dual-branch architecture:
- **Spatial Branch**: EfficientNet-B4 for visual features
- **Frequency Branch**: SRM filters for spectral artifacts
- **Explainability**: Grad-CAM heatmaps + faithful text explanations
""")
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Disclaimer**: This tool provides probabilistic assessments, not definitive determinations. Outputs are framed as likelihoods.")
st.sidebar.markdown("---")
st.sidebar.markdown("**Built for SIH 2026 Internal Hackathon**")
st.sidebar.markdown("L.J. Institute of Engineering and Technology")

# ---- CHECK MODEL ----
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model', 'weights')

if not MODULES_AVAILABLE:
    st.error(f"Required modules not available: {import_error}")
    st.info("Please install requirements: `pip install -r requirements.txt`")
    st.stop()

model, scaler, gradcam = load_model_cached(MODEL_DIR)

if model is None:
    st.warning("⚠️ Model weights not found. Running in **Demo Mode** with untrained model.")
    st.info(f"To use a trained model, place `best_model.pth` in `{MODEL_DIR}`")
    # Load untrained model for demo
    device = torch.device('cpu')
    model = DualBranchNet(num_classes=2, backbone='efficientnet_b4', pretrained=True)
    model.to(device)
    model.eval()
    scaler = TemperatureScaler()
    gradcam = SignalScopeGradCAM(model, device=device)
    demo_mode = True
else:
    demo_mode = False

# ---- FILE UPLOAD ----
st.markdown("### 📤 Upload an Image for Analysis")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Supported formats: JPG, PNG, WEBP"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Run analysis
    with st.spinner('🔍 Analyzing image...'):
        device = torch.device('cpu')

        # Get prediction
        prediction = predict_from_pil(image, model, scaler, device=device)

        # Generate Grad-CAM
        image_tensor = prediction['tensor']
        heatmap_result = gradcam.generate(image_tensor)
        heatmap = heatmap_result['heatmap']

        # Create overlay
        img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        overlay = gradcam.overlay(img_np, heatmap)

        # Generate explanation
        report = generate_full_report(prediction, heatmap)
        
        # Analyze metadata
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_img.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        metadata_analysis = analyze_metadata(temp_path)

    # ---- RESULTS ----
    st.markdown("---")

    if demo_mode:
        st.info("🔔 **Demo Mode**: Results are from an untrained model. Train the model for accurate predictions.")

    # Verdict banner
    label = prediction['label']
    confidence = prediction['confidence']
    emoji = get_confidence_color(confidence, label)

    if label == 'ai_generated':
        st.error(f"{emoji} **Verdict: Likely AI-Generated** — Confidence: {confidence*100:.1f}%")
    else:
        st.success(f"{emoji} **Verdict: Likely Real** — Confidence: {confidence*100:.1f}%")

    # Columns layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📸 Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("#### 🔥 Grad-CAM Heatmap")
        st.image(overlay, use_container_width=True, clamp=True)

    # Explanation & Metadata section
    st.markdown("---")
    
    col_exp, col_meta = st.columns([2, 1])
    
    with col_exp:
        st.markdown("### 📝 Explanation")
        st.markdown(report['explanation'])
        
    with col_meta:
        st.markdown("### 🔍 Provenance & Metadata")
        if metadata_analysis['ai_signature_found']:
            st.error("⚠️ **AI Software Signature Found!**")
            st.write(f"Software: `{metadata_analysis['software']}`")
        elif metadata_analysis['suspicious']:
            st.warning("⚠️ **Suspicious Metadata**\nMissing standard camera EXIF data.")
        else:
            st.success("✅ **Camera EXIF Present**")
            if metadata_analysis['camera_make'] or metadata_analysis['camera_model']:
                st.write(f"Camera: `{metadata_analysis.get('camera_make', '')} {metadata_analysis.get('camera_model', '')}`")
        
        with st.expander("View Raw Metadata"):
            if metadata_analysis['raw_metadata']:
                st.json(metadata_analysis['raw_metadata'])
            else:
                st.write("No standard EXIF data found.")

    # Confidence bar
    st.markdown("### 📊 Confidence Score")
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.progress(confidence)
    with col_b:
        st.metric("Confidence", f"{confidence*100:.1f}%")

    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("**Raw Model Output:**")
        st.json({
            'label': prediction['label'],
            'confidence': prediction['confidence'],
            'fake_probability': prediction['fake_probability'],
            'raw_logits': prediction['raw_logits']
        })
        st.markdown("**Heatmap Analysis:**")
        st.json(report['heatmap_analysis'])

    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ " + report['disclaimer'])

else:
    # Show instructions when no image uploaded
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: #f0f2f6; border-radius: 10px; margin: 20px 0;">
        <h3>👆 Upload an image to get started</h3>
        <p>Drop in a suspicious image and SignalScope will analyze it for AI-generated artifacts.</p>
        <p style="color: #666;">Supports JPG, PNG, WEBP formats</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "SignalScope — SIH 2026 | Dual-Branch Architecture (EfficientNet-B4 + SRM Filters) | "
    "Grad-CAM Explanations | Temperature-Calibrated Confidence"
    "</div>",
    unsafe_allow_html=True
)

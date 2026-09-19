import streamlit as st
import os
import torch
from src.model.legacy_model import LegacyDualBranchNet
from src.model.dual_branch_net import TemperatureScaler
from src.explain.gradcam import SignalScopeGradCAM

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model', 'weights')

@st.cache_resource(show_spinner=False)
def init_model():
    """Loads and caches the model components."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(MODEL_DIR, 'best_model.pth')
    temp_path = os.path.join(MODEL_DIR, 'temperature.json')
    
    if not os.path.exists(model_path):
        return None, None, None
        
    model = LegacyDualBranchNet()
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    
    scaler = TemperatureScaler()
    if os.path.exists(temp_path):
        scaler.load(temp_path)
        
    gradcam = SignalScopeGradCAM(model, device=device)
    
    return model, scaler, gradcam

def ensure_model_loaded():
    if 'model_loaded' not in st.session_state or not st.session_state.model_loaded:
        with st.spinner("Initializing AI Core..."):
            model, scaler, gradcam = init_model()
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.gradcam = gradcam
            st.session_state.model_loaded = model is not None

import streamlit as st
import pandas as pd
import torch
import numpy as np
from PIL import Image
import io
from src.app.ui.theme import apply_theme
from src.model.predict import get_transforms

st.set_page_config(page_title="Batch Scanner - SignalScope", layout="wide")
apply_theme()

st.title("Batch Image Scanner")
st.markdown("<p class='subtitle'>Upload multiple images to rapidly scan them for AI-generation signals.</p>", unsafe_allow_html=True)

from src.app.utils import ensure_model_loaded
ensure_model_loaded()

if not st.session_state.get('model_loaded', False):
    st.error("Model checkpoint could not be found.")
    st.stop()

uploaded_files = st.file_uploader("Upload multiple images", type=['jpg', 'jpeg', 'png', 'webp'], accept_multiple_files=True)

if uploaded_files:
    if st.button("Start Batch Scan", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        model = st.session_state.model
        scaler = st.session_state.scaler
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform = get_transforms()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Scanning {file.name} ({i+1}/{len(uploaded_files)})...")
            
            image = Image.open(io.BytesIO(file.getvalue())).convert('RGB')
            img_np = np.array(image)
            img_tensor = transform(image=img_np)['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                out = model(img_tensor)
                logits = out['logits']
                calibrated_logits = scaler.scale(logits)
                probs = torch.softmax(calibrated_logits, dim=1)[0]
                
            prob_fake = probs[1].item()
            is_fake = prob_fake > 0.5
            confidence = prob_fake if is_fake else (1.0 - prob_fake)
            
            results.append({
                "Filename": file.name,
                "Verdict": "Likely AI-Generated" if is_fake else "Likely Real",
                "Confidence (%)": round(confidence * 100, 2),
                "Evidence": "High Frequency Artefacts" if is_fake else "Natural Frequencies",
                "Status": "⚠️ Flagged" if is_fake else "✓ Clean"
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Batch scan complete!")
        
        # Display Results
        df = pd.DataFrame(results)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scanned", len(results))
        col2.metric("AI Detected", len(df[df['Verdict'] == 'Likely AI-Generated']))
        col3.metric("Real Detected", len(df[df['Verdict'] == 'Likely Real']))
        
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forensic Report (CSV)", csv, "signalscope_batch_report.csv", "text/csv")

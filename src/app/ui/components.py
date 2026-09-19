import streamlit as st

def render_verdict_banner(is_fake, confidence):
    """Renders the top verdict banner based on prediction."""
    if is_fake:
        st.markdown(f"""
            <div class="verdict-banner verdict-fake">
                <h1>LIKELY AI-GENERATED</h1>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="verdict-banner verdict-real">
                <h1>LIKELY REAL</h1>
                <p>Confidence: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

def render_evidence_card(title, value, subtitle, status_class="status-neutral"):
    """Renders a metric/evidence card."""
    st.markdown(f"""
        <div class="evidence-card">
            <h4 style="margin: 0; color: #64748B; font-size: 0.875rem; text-transform: uppercase;">{title}</h4>
            <div style="font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">{value}</div>
            <span class="status-pill {status_class}">{subtitle}</span>
        </div>
    """, unsafe_allow_html=True)

def render_disclaimer():
    st.markdown("""
        <div class="disclaimer-box">
            <strong>⚠️ Disclaimer:</strong> Highlighted regions indicate areas that influenced the model's prediction. They are supporting evidence, not proof of manipulation. Missing metadata is NOT proof that an image is AI-generated.
        </div>
    """, unsafe_allow_html=True)

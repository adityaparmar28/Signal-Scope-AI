import streamlit as st

def apply_theme():
    """Applies the Forensic Light theme to the Streamlit app."""
    st.markdown("""
        <style>
        /* Forensic Light Colors */
        :root {
            --bg-color: #F7F9FC;
            --surface-color: #FFFFFF;
            --primary-color: #1D4ED8;
            --primary-hover: #1E40AF;
            --text-main: #0F172A;
            --text-secondary: #64748B;
            --border-color: #E2E8F0;
            --success-color: #16A34A;
            --warning-color: #D97706;
            --danger-color: #DC2626;
        }

        /* App Background */
        .stApp {
            background-color: var(--bg-color);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: var(--text-main);
        }
        
        .subtitle {
            color: var(--text-secondary) !important;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }

        /* Cards */
        .evidence-card {
            background-color: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Verdict Banners */
        .verdict-banner {
            padding: 2rem;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        }
        .verdict-fake {
            background-color: #FEF2F2;
            border: 1px solid #FECACA;
            color: var(--danger-color);
        }
        .verdict-real {
            background-color: #F0FDF4;
            border: 1px solid #BBF7D0;
            color: var(--success-color);
        }
        .verdict-banner h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: 1px;
            color: inherit;
        }
        .verdict-banner p {
            margin-top: 0.5rem;
            font-size: 1.25rem;
            color: inherit;
            opacity: 0.9;
        }

        /* Disclaimer */
        .disclaimer-box {
            background-color: #FFFBEB;
            border-left: 4px solid var(--warning-color);
            padding: 1rem;
            border-radius: 4px;
            margin-top: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: #92400E;
        }

        /* Status Pills */
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-success { background-color: #DCFCE7; color: #166534; }
        .status-warning { background-color: #FEF3C7; color: #92400E; }
        .status-danger { background-color: #FEE2E2; color: #991B1B; }
        .status-neutral { background-color: #F1F5F9; color: #475569; }
        
        </style>
    """, unsafe_allow_html=True)

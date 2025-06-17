# utils/st_style.py

import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        ul.st-emotion-cache-1gczx66.edtmxes2 {
            font-size: 26px !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #D5F5E3;
        }

        .stApp {
            background-color: #f9f9f9;
        }

        .stMarkdown p {
    font-size: 24px;
        }

        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.markdown("## 🔬 FunVFPred")


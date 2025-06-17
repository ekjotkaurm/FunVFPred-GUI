# Home.py
import streamlit as st
st.set_page_config(page_title="FunVFPred - Fungal Virulence Factor Predictor", layout="wide")
from utils.st_style import apply_custom_css
apply_custom_css()

st.title("FunVFPred: Predicting fungal virulence factors using a unified representation learning model")

st.markdown("""
FunVFPred fills a critical gap in fungal pathogenicity research by being the first computational approach designed specifically to predict fungal VFs, based on Random Forest classifier. The model's predictive performance was significantly enhanced through the incorporation of UniRep embeddings, either individually or in combination with traditional sequence-based features like AAC and DDE.
""")

# Add image here
st.image("assets/home.jpg", caption="FunVFPred - Unified Model for Fungal VF Prediction", use_container_width=True)

st.markdown("""
Fig. Key virulence factors of Candida albicans contributing to its growth, survival, and infection in the human host
""")



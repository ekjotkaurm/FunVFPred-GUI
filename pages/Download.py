# pages/Download.py

import streamlit as st
st.set_page_config(page_title="Download - FunVFPred", layout="centered")



from utils.st_style import apply_custom_css
apply_custom_css()


st.title("⬇️ Download FunVFPred")

st.markdown("""
FunVFPred is available as a standalone Python-based tool that you can run on your own system.

You can download the full source code and instructions from GitHub:
""")

st.markdown("""
### 📦 GitHub Repository  
🔗 [https://github.com/ekjotkaurm/FunVFPred](https://github.com/ekjotkaurm/FunVFPred)
""", unsafe_allow_html=True)


st.title("Install Docker and deploy the tool on your own system.")
st.markdown("""
### 📦 GitHub Repository
🔗 [https://github.com/ekjotkaurm/FunVFPred-GUI](https://github.com/ekjotkaurm/FunVFPred-GUI)
""", unsafe_allow_html=True)


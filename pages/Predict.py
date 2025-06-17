# pages/Predict.py
import streamlit as st
import pandas as pd
import os
import tempfile
from Bio import SeqIO
from predict_virulence import compute_aac, compute_dde, extract_unirep_features, predict_from_fasta
import numpy as np
import joblib
from tape import UniRepModel, TAPETokenizer
import torch

st.set_page_config(page_title="Virulence Prediction", layout="wide")
from utils.st_style import apply_custom_css
apply_custom_css()

st.title("🧬 Virulence Prediction Tool")

st.markdown("""
Upload your Fungal Protein FASTA file below to predict virulent proteins.  
""")

@st.cache_data(show_spinner=False)
def load_model():
    return joblib.load("rf_model.joblib")

@st.cache_data(show_spinner=False)
def extract_unirep(sequences):
    model = UniRepModel.from_pretrained("babbler-1900").eval()
    tokenizer = TAPETokenizer(vocab="unirep")
    embeddings = []
    for seq in sequences:
        with torch.no_grad():
            encoded = torch.tensor([tokenizer.encode(seq[:1024])])
            output = model(encoded)[0]
            mean_embedding = output.mean(dim=1).squeeze().numpy()
            embeddings.append(mean_embedding)
    return np.array(embeddings)

@st.cache_data(show_spinner=False)
def extract_features(sequences):
    aac_features = np.array([compute_aac(seq) for seq in sequences])
    dde_features = np.array([compute_dde(seq) for seq in sequences])
    unirep_features = extract_unirep(sequences)
    return np.concatenate([aac_features, dde_features, unirep_features], axis=1)

@st.cache_data(show_spinner=False)
def predict_from_model(_model, features):
    return _model.predict(features)

uploaded_file = st.file_uploader("📁 Upload a FASTA file", type=["fasta", "fa"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
        tmp.write(uploaded_file.getvalue())
        fasta_path = tmp.name

    # Step 1: Read sequences first
    sequences = []
    protein_ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        protein_ids.append(record.id)

    total_seqs = len(sequences)
    if total_seqs == 0:
        st.warning("⚠️ No sequences found in the uploaded file.")
    else:
        st.info(f"📄 Total Sequences Detected: **{total_seqs}**")

        with st.spinner("🔬 Extracting features and predicting... please wait (this may take some time depending on file size)..."):
            try:
                st.markdown("🔁 **Extracting Features...**")
                full_features = extract_features(sequences)

                st.markdown("✅ **Running Model Prediction...**")
                model = load_model()
                preds = predict_from_model(model, full_features)
                pred_labels = ['Virulent' if p == 1 else 'Non-Virulent' for p in preds]

                df_results = pd.DataFrame({
                    'Protein ID': protein_ids,
                    'Prediction': pred_labels
                })

                st.success("✅ Prediction completed!")

                st.markdown("### 🧾 Prediction Summary (Top & Bottom 10 Sequences)")
                st.dataframe(pd.concat([df_results.head(10), df_results.tail(10)]), use_container_width=True)

                csv = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Full Results as CSV",
                    data=csv,
                    file_name="virulence_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")


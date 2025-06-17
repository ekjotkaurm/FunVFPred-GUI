# pages/Algorithm.py

import streamlit as st
import pandas as pd
st.set_page_config(page_title="FunVFPred - Algorithm", layout="wide")

from utils.st_style import apply_custom_css
apply_custom_css()


st.title("🧬 Algorithm Overview - FunVFPred")

st.markdown("""
FunVFPred uses a multi-step pipeline to predict fungal virulence factors using traditional and deep learning-based sequence features. Below are the key steps involved:
""")

# Add image here
st.image("assets/algo.jpg", caption="FunVFPred - Unified Model for Fungal VF Prediction", use_container_width=True)

with st.expander("🔹 Step 1: Preparing Input Files"):
    st.markdown("""
**(i) input.fasta:**  
Contains both positive (virulent) and negative (non-virulent) protein sequences.  
Positive sequences are collected from:  
- PHI-base: http://www.phi-base.org/  
- Victors: http://www.phidias.us/victors  
- DFVF: http://sysbio.unl.edu/DFVF/  

Negative sequences are sourced from UniProt: https://www.uniprot.org/  
Redundancy is removed using CD-HIT at 100% identity.

**(ii) Dataset Balancing:**  
Random undersampling is used to balance the dataset (1:1 ratio of positive to negative).

**(iii) labels.csv:**  
labeled file in csv format containing balanced dataset of positive and negative protein ids    
""")

    st.markdown("#### Table: Virulent Protein Sequences Used")
    df1 = pd.DataFrame({
        "Fungal species": [
            "Candida albicans", "Candida glabrata", "Candida tropicalis",
            "Candida parapsilosis", "Candida dubliniensis", "Candida orthopsilosis",
            "Candida glycerinogenes / Pichia kudriavzevii", "Total"
        ],
        "Positive proteins": [692, 69, 4, 18, 23, 6, 1, 813],
        "After redundancy removal": [508, 64, 4, 15, 21, 3, 1, 616]
    })
    st.table(df1)

with st.expander("🔹 Step 2: Feature Encoding"):
    st.markdown("""
Enhanced predictive accuracy for virulence factors (VFs) was achieved by leveraging multiple sequence-based features. These included amino acid composition (AAC), dipeptide deviation from expected mean (DDE), and UniRep, a protein sequence representation learned through a masked language modeling framework. UniRep, a pretrained model, was utilized to extract comprehensive features from protein sequences.

**AAC (Amino Acid Composition):**  
Protein sequences were read from a FASTA file using Biopython (version 1.81), and the AAC values were calculated using NumPy for efficient computation. Each feature vector was then linked to its respective class label (virulent = 1, non-virulent = 0), as provided in a CSV file containing protein identifiers. The final dataset, combining AAC features and labels, was saved in CSV format for use in machine learning models.

**DDE (Dipeptide Deviation from Expected Mean):**  
The DDE features were computed using an in-house Python script employing the Biopython package. Each protein sequence was parsed from a FASTA file, and its dipeptide composition was calculated. The generated features were merged with corresponding labels and saved in CSV format for downstream classification tasks. 

**UniRep Embeddings:**  
UniRep is a comprehensive deep representation learning method that uses a multidimensional Long Short-Term Memory (mLSTM) model with 1900 dimensions. It has already been trained on the UniRef50 protein database to recognize complex features of amino acid sequences. High-dimensional embeddings that accurately depict the structural and functional characteristics of proteins are produced using this approach. UniRep is an effective instrument for evaluating complicated biological systems as it provides extensive and relevant representations of protein sequences, that has shown significant potential for enhancing the efficiency and accuracy of protein engineering operations. The UniRep features were extracted using the TAPE (Tasks Assessing Protein Embeddings) library in Python. Protein sequences were parsed from a FASTA file and encoded using a UniRep-compatible tokenizer. The pre-trained UniRep model (babbler-1900) was utilized to generate 1900-dimensional embeddings by averaging the hidden states across each sequence. The resulting feature vectors were combined with protein identifiers and corresponding labels from the input dataset. The final feature matrix was saved in CSV format for model development.

    """)

with st.expander("🔹 Step 3: Feature Fusion"):
    st.markdown("""
By merging features from AAC, DDE, and pre-trained UniRep embeddings, feature fusion is accomplished as shown in Table 2. Concatenation serves to integrate these traits into a single, all-inclusive representation, improving their capacity to capture crucial biological data. 
To improve the predictive performance of machine learning and deep learning models, various feature representations were integrated. Feature integration was performed by aligning datasets on a shared identifier (protein_ids), ensuring accurate correspondence across records. Merging was carried out using Python's pandas library, followed by deduplication of label fields and standardization. Any missing entries introduced during the merge were imputed with zeros to maintain consistency across samples. This approach enabled the construction of several comprehensive feature combinations as detailed in Table 2.

""")
    st.markdown("#### Table: Fused Feature Combinations")
    df2 = pd.DataFrame({
        "Concatenate Features": ["F1 + F2", "F1 + F3", "F2 + F3", "F1 + F2 + F3"],
        "Final Feature Name": ["F4", "F5", "F6", "F7"]
    }, index=["AAC + DDE", "AAC + UniRep", "DDE + UniRep", "AAC + DDE + UniRep"])
    st.table(df2)

with st.expander("🔹 Step 4: Data Splitting"):
    st.markdown("""
Following feature extraction and class balancing, the dataset was partitioned into three distinct subsets to facilitate model training, testing, and independent evaluation. Specifically, 70% of the data was allocated to the training set, which was used to develop machine learning and deep learning models. The test set, comprising 20% of the data, was used to evaluate model performance during development. An additional 10% was reserved as an independent validation set (termed validation set 1) to assess the model's ability to generalize to unseen protein sequences, ensuring the robustness and reliability of the predictive framework (Table 3).
    """)
    st.markdown(" #### Table 3. Protein sequence distribution between training, test and validation sets")
    df3 = pd.DataFrame({
        "Dataset Type": ["Positive", "Negative", "Both"],
        "Total Sequences": [616, 616, 1232],
        "Train (70%)": [431, 431, 862],
        "Test (20%)": [123, 123, 246],
        "Validation (10%)": [62, 62, 124]
    })
    st.table(df3)

with st.expander("🔹 Step 5: Model Building & Classification"):
    st.markdown("""To assess the predictive power of the extracted features, both deep learning (DL) and traditional machine learning (ML) algorithms were employed. Specifically, the models included Deep Neural Networks (DNN), Multi-Layer Perceptron (MLP), Artificial Neural Networks (ANN), and Random Forest (RF). Among these, the RF algorithm was particularly favoured due to its robust performance in handling high-dimensional input spaces while mitigating the risk of overfitting, making it well-suited for the complexity of our feature set. RF was implemented using an ensemble of 100 decision trees, incorporating bootstrap aggregation and random feature selection at each node split. To enhance computational efficiency, parallel processing was utilized across all available CPU cores, and a fixed random seed (42) was set to maintain reproducibility of results.""")

with st.expander("📊 Performance evaluation" ):
    st.markdown("""Classification models were trained using various combinations of extracted features—AAC, DDE, UniRep, and their fused forms. Among all, the Random Forest (RF) model, with 100 decision trees and a fixed random seed (42), achieved the highest accuracy of 77.4% and an MCC of 0.5509 when trained on the combined AAC+DDE+UniRep features. The ANN model, with two hidden layers (64 and 32 neurons) and dropout regularization, reached 75% accuracy using AAC+UniRep. The MLP model, comprising three hidden layers, also achieved 75% accuracy on DDE+UniRep. The DNN model, featuring four dense layers and batch normalization, delivered 75.81% accuracy on AAC+UniRep but did not outperform RF (Table 4).
Overall, RF demonstrated superior predictive performance, showing robustness and consistency across all feature sets.
""")

# Table 4: Performance of classifiers on validation set


    st.markdown("#### Table 4. Performance of Machine Learning and Deep Learning Models on Validation Set")

    df4 = pd.DataFrame({
        "Classifier": [
            "RF", "RF", "RF", "RF", "RF", "RF", "RF",
            "ANN", "ANN", "ANN", "ANN", "ANN", "ANN", "ANN",
            "MLP", "MLP", "MLP", "MLP", "MLP", "MLP", "MLP",
            "DNN", "DNN", "DNN", "DNN", "DNN", "DNN", "DNN"
        ],
        "Features": [
            "AAC", "DDE", "UNIREP", "AAC+DDE", "AAC+UNIREP", "DDE+UNIREP", "AAC+DDE+UNIREP",
            "AAC", "DDE", "UNIREP", "AAC+DDE", "AAC+UNIREP", "DDE+UNIREP", "AAC+DDE+UNIREP",
            "AAC", "DDE", "UNIREP", "AAC+DDE", "AAC+UNIREP", "DDE+UNIREP", "AAC+DDE+UNIREP",
            "AAC", "DDE", "UNIREP", "AAC+DDE", "AAC+UNIREP", "DDE+UNIREP", "AAC+DDE+UNIREP"
        ],
        "Accuracy": [
            0.6935, 0.6935, 0.7258, 0.6774, 0.7016, 0.7419, 0.7741,
            0.6048, 0.6613, 0.7258, 0.6774, 0.7500, 0.7258, 0.7419,
            0.6693, 0.6612, 0.7419, 0.6854, 0.7258, 0.7500, 0.7177,
            0.6855, 0.6371, 0.7500, 0.6452, 0.7258, 0.7177, 0.7661
        ],
        "MCC": [
            0.3879, 0.3945, 0.4603, 0.3565, 0.4075, 0.4861, 0.5509,
            0.2104, 0.3241, 0.4518, 0.3595, 0.5001, 0.4526, 0.4880,
            0.34, 0.33, 0.48, 0.38, 0.45, 0.51, 0.44,
            0.3722, 0.2751, 0.5001, 0.2909, 0.4518, 0.4355, 0.5357
        ],
        "AUC": [
            0.7578, 0.8034, 0.8173, 0.7563, 0.8002, 0.8055, 0.8225,
            0.7310, 0.7391, 0.7994, 0.7568, 0.8301, 0.8119, 0.8439,
            0.7393, 0.6992, 0.8251, 0.7319, 0.8344, 0.8069, 0.7950,
            0.7690, 0.7106, 0.8163, 0.7447, 0.8371, 0.8040, 0.8439
        ]
    })

    st.table(df4)



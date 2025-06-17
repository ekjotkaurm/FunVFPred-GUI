# FunVFPred-GUI
# 🧬 FunVFPredGUI: Virulence Prediction Tool

**FunVFPredGUI** is a web application for predicting **virulent proteins** from input FASTA sequences using a pre-trained Random Forest model. It extracts three types of features:

- **AAC** (Amino Acid Composition)
- **DDE** (Dipeptide Deviation from Expected)
- **UniRep** (Universal Protein Embeddings from TAPE)

The tool is fully containerized using **Docker** and runs in any environment where Docker is installed. It provides a browser-accessible GUI that requires no coding expertise to use.

---

## 🌐 Live Demo

Try the hosted version here:  
👉 [https://fgcsl.ihbt.res.in:8501](https://fgcsl.ihbt.res.in:8501)

---

## 🐳 Run Locally with Docker

### 1️⃣ Install Docker

Follow the official Docker installation guide:  
👉 https://docs.docker.com/get-docker/

Then confirm Docker is installed:

```bash
docker --version
```

## ⚙️ Build and Run the Tool

### Step 1: Clone this Repository
```bash
git clone https://github.com/ekjotkaurm/FunVFPred-GUI.git
cd FunVFPred-GUI
```

### Step 2: Build the Docker Image
```bash
docker build -t funvfpred-app .
```

### Step 3: Run the App
```bash
docker run -p 8501:8501 funvfpred-app
```

Then open your browser at:
🔗 http://localhost:8501


## 📥 Output

After uploading a FASTA file, the app displays a table with predictions for each protein:
- **Protein ID**
- **Prediction**: Virulent or Non-Virulent

Results can be previewed in the browser and downloaded as a CSV file.


👥 Team Contribution
Model development and core script creation: Ekjot Kaur

🧑‍💻 GUI Interface development, Docker containerization, and GitHub deployment: Abhishek Khatri
under the supervision of Dr. Vishal Acharya




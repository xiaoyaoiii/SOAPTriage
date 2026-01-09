<div align="center">
  <img src="Doc/Pictures/title.png" alt="title" width="90%">
</div>

<p float="left"><img src="https://img.shields.io/badge/python-v3.10+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">

# SOAPTriage

This repository provides the official implementation of the paper **"SOAPTriage: SOAP-Guided Multi-View Clinical Text Modeling Framework for Automated ESI Prediction."** It includes the data construction pipeline, source code, experimental results, and detailed usage instructions to support reproducible research and future extensions. The repo covers the **Clinical Note Augmentation (CNA)** module for generating a large-scale triage-note dataset from structured ED records, as well as the **SOAP-guided** multi-view modeling components (**SGE / SAAI**) that encode and aggregate **Subjective**, **Objective**, **Assessment**, and **Plan** perspectives for automated **ESI (1–5)** prediction, together with evaluation scripts and reproducibility materials.

<div align="center">
  <img src="Doc/Pictures/figure1.png" alt="figure1" width="70%">
</div>

## 📂 Dataset Overview
<div align="center">
  <img src="Doc/Pictures/table1.png" alt="table1" width="70%">
</div>

### Constructed Triage Datasets
We construct two triage-note datasets from de-identified structured ED visit records via our **Clinical Note Augmentation (CNA)** pipeline:

- **MIMIC-IV (Augmented)** — **15,393** augmented ED triage notes with **ESI labels (1–5)**, split into **train/val/test = 8:1:1**.
- **NHAMCS (Augmented)** — **16,596** augmented ED triage notes labeled by **immediacy rating (IR 1–5)**, a triage scale comparable to ESI, also split into **train/val/test = 8:1:1**.

****For more detailed dataset descriptions, please [click here](Data/README.md).**
**

## 🧠 SOAPTriage Framework
<div align="center">
  <img src="Doc/Pictures/figure2.png" alt="figure2" width="70%">
</div>

Architecture of SOAPTriage. The CNA module converts structured ED records into natural-language triage notes to construct a high-quality dataset. The SGE module, guided by SOAP triage theory, comprehensively assesses patients' conditions and extracts mid-depth representations from four perspectives. The SAAI module adaptively aggregates these SOAP-aware representations to infer the final ESI level.

**For more detailed experimental results, please [Click here!](Doc/Supplementary%20Experiments/README.md)**

## 📊 MIMIC-IV Results
<div align="center">
  <img src="Doc/Pictures/mimic_table.png" alt="mimic_table" width="70%">
</div>

Performance comparison (%) between SOAPTriage and baseline methods on the clinical triage dataset. Best results are highlighted in bold, while the second-best results are indicated by underlining. Lower values indicate better performance.

**For comprehensive details on all baseline models, please [Click here.](Doc/Supplementary%20Experiments/Baseline.md)**

**For more detailed results, please [Click here.](Doc/Supplementary%20Experiments/README.md)**

## ✨ NHAMCS Results
<div align="center">
  <img src="Doc/Pictures/nh_table.png" alt="nh_table" width="70%">
</div>

Performance comparison (%) between SOAPTriage and baseline methods on the clinical triage dataset. Best results are highlighted in bold, while the second-best results are indicated by underlining. Lower values indicate better performance.

## 📖 Usage
You can implement our methods according to the following steps:

1. Install the necessary packages. Run the command:
   ```shell
   pip install -r requirements.txt
   ```
2. Run our code using Python.

   Data Augmentation:
   ```shell
   python CallApi.py
   ```
   
   Encode:
   ```shell
   bash encode.sh
   ```

   Train the SOAPTriage:
   ```shell
   bash train.sh
   ```
   Predict the ESI:
   ```shell
   bash predict.sh


## 🌟 Contributions and suggestions are welcome!

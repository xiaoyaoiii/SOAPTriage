<img src="Doc/Pictures/title.png" alt="title" border="0">

<p float="left"><img src="https://img.shields.io/badge/python-v3.10+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">
   
# SOAPTriage

This repository provides the official implementation of the paper **"SOAPTriage: SOAP-Guided Multi-View Clinical Text Modeling Framework for Automated ESI Prediction."** It includes the data construction pipeline, source code, experimental results, and detailed usage instructions to support reproducible research and future extensions. The repo covers the **Clinical Note Augmentation (CNA)** module for generating a large-scale triage-note dataset from structured ED records, as well as the **SOAP-guided** multi-view modeling components (**SGE / SAAI**) that encode and aggregate **Subjective**, **Objective**, **Assessment**, and **Plan** perspectives for automated **ESI (1–5)** prediction, together with evaluation scripts and reproducibility materials.

<img src="Doc/Pictures/figure1.png" alt="figure1" border="0">


## 📂 Dataset Overview
<img src="Doc/Pictures/table1.png" alt="table1" border="0">

### Constructed Triage Datasets
We release two triage-note datasets built from de-identified structured ED visit records via our **Clinical Note Augmentation (CNA)** pipeline:

- **MIMIC-IV (Augmented)** — **15,393** augmented ED triage notes with **ESI labels (1–5)**, split into **train/val/test = 8:1:1**.
- **NHAMCS (Augmented)** — **16,596** augmented ED triage notes labeled by **immediacy rating (IR 1–5)**, a triage scale comparable to ESI, also split into **train/val/test = 8:1:1**.

## 🧠 SOAPTriage Architecture
<img src="Doc/Pictures/figure2.png" alt="figure2" border="0">

**For more detailed experimental results, please [Click here!](Doc/Supplementary%20Experiments/README.md)**

## 📊 MIMIC-IV Results
<img src="Doc/Pictures/table2.png" alt="mimic_table" border="0">

For comprehensive details on all baseline models, please [Click here.](Doc/Supplementary%20Experiments/Baseline.md)

For more detailed benchmark results, please [Click here.](Doc/Supplementary%20Experiments/README.md)

## ✨ NHAMCS Performance
<img src="Doc/Pictures/table3.png" alt="nh_table" border="0">

For more detailed performance results, please [Click here.](Doc/Supplementary%20Experiments/README.md)

## 📖 Usage
You can implement our methods according to the following steps:

1. Install the necessary packages. Run the command:
   ```shell
   pip install -r requirements.txt
   ```
2. Install Swift to deploy models. Please [Click here.](https://swift.readthedocs.io/zh-cn/latest/index.html)
3. Run our code using Python.
   
   Train the KEA:
   ```shell
   python KEA_train.py
   ```
   Evaluate the KEA:
   ```shell
   python KEA_test.py
   ```
   Zero-Shot Testing:
   ```shell
   python zeroshot.py
   ```
   Few-Shot Testing:
   ```shell
   python fewshot.py
   ```

## 🌟 Contributions and suggestions are welcome!

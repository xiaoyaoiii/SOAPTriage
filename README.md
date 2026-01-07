<img src="Doc/Pictures/title.png" alt="title" border="0">

<p float="left"><img src="https://img.shields.io/badge/python-v3.10+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">
   
# SOAPTriage

This repository provides the official implementation of the paper **"SOAPTriage: SOAP-Guided Multi-View Clinical Text Modeling Framework for Automated ESI Prediction."** It includes the data construction pipeline, source code, experimental results, and detailed usage instructions to support reproducible research and future extensions. The repo covers the **Clinical Note Augmentation (CNA)** module for generating a large-scale triage-note dataset from structured ED records, as well as the **SOAP-guided** multi-view modeling components (**SGE / SAAI**) that encode and aggregate **Subjective**, **Objective**, **Assessment**, and **Plan** perspectives for automated **ESI (1–5)** prediction, together with evaluation scripts and reproducibility materials.

<img src="Doc/Pictures/figure1.png" alt="figure1" border="0">


## 📂 Dataset Overview
<img src="Doc/Pictures/table1.png" alt="table1" border="0">

### PG-Bench Dataset
- **General.jsonl** - General Hospital Patient-Doctor Dialogue Guidance Dataset
- **Gynecological.jsonl** - Gynecological Specialty Hospital Patient-Doctor Dialogue Guidance Dataset
- **Pediatric.jsonl** - Pediatric Specialty Hospital Patient-Doctor Dialogue Guidance Dataset  
- **Stomatological.jsonl** - Stomatological Specialty Hospital Patient-Doctor Dialogue Guidance Dataset
- **TCM.jsonl** - TCM Specialty Hospital Patient-Doctor Dialogue Guidance Dataset

## 🧠 KEA Architecture
<img src="Doc/Pictures/figure2.png" alt="figure2" border="0">

**For more detailed experimental results, please [Click here!](Doc/Supplementary%20Experiments/README.md)**

## 📊 Benchmark Results
<img src="Doc/Pictures/table2.png" alt="table2" border="0">

For comprehensive details on all baseline models, please [Click here.](Doc/Supplementary%20Experiments/Baseline.md)

For more detailed benchmark results, please [Click here.](Doc/Supplementary%20Experiments/README.md)

## ✨ KEA Performance
<img src="Doc/Pictures/table3.png" alt="table3" border="0">
<img src="Doc/Pictures/table4.png" alt="table4" border="0">
<img src="Doc/Pictures/table5.png" alt="table5" border="0">

For more detailed performance results, please [Click here.](Doc/Supplementary%20Experiments/README.md)

## 🔍 Case Study
- A case study of KEA utilizing EKP to recommend the appropriate department. EKP: Evolving Knowledge Pool.
<img src="Doc/Pictures/casestudy1.png" alt="casestudy1" border="0">


## 📝 Prompt Templates
- PG-Bench Dataset Construction Template and Prompt Instructions.
<img src="Doc/Pictures/prompt1.png" alt="prompt1" border="0">

- System Prompt Instructions for PG-Bench.
<img src="Doc/Pictures/prompt2.png" alt="prompt2" border="0">

- System Prompt Instructions for KEA.
<img src="Doc/Pictures/prompt3.png" alt="prompt3" border="0">

- Reflection Process Prompt Instructions for KEA.
<img src="Doc/Pictures/prompt4.png" alt="prompt4" border="0">

- Reflection-Based Response Prompt Instructions for KEA.
<img src="Doc/Pictures/prompt5.png" alt="prompt5" border="0">

## 🏥 Departments List
- Description of the List of Subordinate Departments within the PG-Bench Dataset Subsets.
<img src="Doc/Pictures/list.png" alt="list" border="0">


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

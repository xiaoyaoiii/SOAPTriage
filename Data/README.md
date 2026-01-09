# Dataset
Due to data governance and privacy considerations, we only provide **partial examples** of the two constructed datasets. These examples are intended to illustrate the data format, field structure, and typical triage-note style, and can be used for quick understanding and for reproducing the preprocessing pipeline. To access the full underlying structured sources, please follow the official data access procedures of the corresponding datasets.

## 🗂️ MIMIC-IV Dataset
We primarily conduct our experiments on **MIMIC-IV**, extracting eligible emergency department (ED) visits by integrating **MIMIC-IV**, **MIMIC-IV-ED**, and **MIMIC-IV-Note** to build a comprehensive triage dataset. Using our **Clinical Note Augmentation (CNA)** pipeline, we convert de-identified structured ED records into natural-language triage notes and **construct** a dataset of **15,393** triage notes with **ESI labels (1–5)**. We split the dataset into **train/validation/test = 8:1:1** while preserving the original label distribution.

All MIMIC-derived data used in this work are accessed and processed in accordance with the official MIMIC-IV data use agreement and license.  
Reference: https://physionet.org/content/mimiciv/

## 🧾 NHAMCS Dataset
We also **construct** a triage-note dataset of **16,596** notes from the **2020–2022 NHAMCS** structured emergency department data published by the **U.S. National Center for Health Statistics (NCHS/CDC)**. The notes are generated from de-identified structured visit records to support research on automated triage modeling and **acuity/severity prediction**.

For details about NHAMCS data collection and documentation, please refer to the official CDC/NCHS page: https://www.cdc.gov/nchs/nhamcs/about/index.html

## 🧩 Reproducibility & Data Construction Code
To support reproducibility, we provide the complete codebase for constructing both datasets, including preprocessing scripts for merging structured tables, cleaning and normalizing fields, and generating triage-style notes via the **Clinical Note Augmentation (CNA)** pipeline.

**For the dataset construction scripts and preprocessing pipeline, please [click here](Code/Data).**

Since the full underlying records are governed by the official access policies of **MIMIC-IV** and **NHAMCS**, we only include partial examples here for format illustration. After obtaining access through the official channels (see links above), users can run the provided scripts following the documented pipeline to reconstruct the datasets locally. The pipeline is sequential and modular, and each stage produces intermediate artifacts that serve as inputs to subsequent stages (e.g., table merging → field normalization → section extraction → note generation → JSON export).

We also include configuration placeholders for paths (e.g., `your_input_*`) to facilitate adaptation to different local directory structures. Reproducibility can be achieved under the same source data and consistent settings (e.g., fixed random seeds and generation/decoding parameters where applicable).

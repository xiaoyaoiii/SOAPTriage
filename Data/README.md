# Dataset

## 🗂️ MIMIC-IV Dataset
We primarily conduct our experiments on **MIMIC-IV**, extracting eligible emergency department (ED) visits by integrating **MIMIC-IV**, **MIMIC-IV-ED**, and **MIMIC-IV-Note** to construct a comprehensive triage dataset. Using our **Clinical Note Augmentation (CNA)** pipeline, we convert de-identified structured ED records into natural-language triage notes and generate **15,393** clinical triage notes with **ESI labels (1–5)**. We split the dataset into **training/validation/test = 8:1:1** while preserving the original label distribution.

All MIMIC-derived data used in this work are accessed and processed in accordance with the official MIMIC-IV data use agreement and license.  
Reference: https://physionet.org/content/mimiciv/

## 🧾 NHAMCS Dataset
We release **16,596** triage notes constructed from the **2020–2022 NHAMCS** structured emergency department data published by the **U.S. National Center for Health Statistics (NCHS/CDC)**. The notes are generated from de-identified structured visit records to support research on automated triage modeling and ESI-related severity prediction.

For details about the NHAMCS data collection and documentation, please refer to the official CDC/NCHS page: https://www.cdc.gov/nchs/nhamcs/about/index.html


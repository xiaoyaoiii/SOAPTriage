# Dataset
Due to data governance and privacy considerations, we only release **partial examples** of the two constructed datasets. These examples are provided to illustrate the data format, field organization, and typical triage-note style, and can be used for rapid understanding and reproducibility of the preprocessing pipeline. For access to the full underlying structured sources, please follow the official data access procedures of the corresponding datasets.


## 🗂️ MIMIC-IV Dataset
We primarily conduct our experiments on **MIMIC-IV**, extracting eligible emergency department (ED) visits by integrating **MIMIC-IV**, **MIMIC-IV-ED**, and **MIMIC-IV-Note** to construct a comprehensive triage dataset. Using our **Clinical Note Augmentation (CNA)** pipeline, we convert de-identified structured ED records into natural-language triage notes and generate **15,393** clinical triage notes with **ESI labels (1–5)**. We split the dataset into **training/validation/test = 8:1:1** while preserving the original label distribution.

All MIMIC-derived data used in this work are accessed and processed in accordance with the official MIMIC-IV data use agreement and license.  
Reference: https://physionet.org/content/mimiciv/

## 🧾 NHAMCS Dataset
We release **16,596** triage notes constructed from the **2020–2022 NHAMCS** structured emergency department data published by the **U.S. National Center for Health Statistics (NCHS/CDC)**. The notes are generated from de-identified structured visit records to support research on automated triage modeling and ESI-related severity prediction.

For details about the NHAMCS data collection and documentation, please refer to the official CDC/NCHS page: https://www.cdc.gov/nchs/nhamcs/about/index.html

## 🧩 Reproducibility & Data Construction Code

To support reproducibility, we provide the complete codebase for constructing both datasets, including all preprocessing scripts used to integrate structured sources, clean and normalize fields, and generate triage-style notes via our Clinical Note Augmentation (CNA) pipeline.

Because the full underlying records are governed by the official data access policies of MIMIC-IV and NHAMCS, we only release partial examples here for format illustration. After obtaining access through the official channels (see links above), users can run the provided scripts in the documented order to reconstruct the full datasets locally. In particular, the pipeline is designed to be sequential and modular, and each step produces intermediate artifacts that serve as the inputs to the next step (e.g., table merging → field normalization → section extraction → note generation → JSON export and dataset split).

We also include configuration placeholders for paths (e.g., your_input_*) to facilitate adaptation to different local directory structures, and all steps are deterministic given the same source data and settings.


## 📏Baseline

**Standard Prompting**: Directly queries large language models with task-specific instructions, serving as a strong and widely adopted baseline for clinical reasoning tasks. 

**Prompt-RAG**: Augments standard prompting with retrieval-augmented generation by retrieving relevant clinical triage knowledge from an external corpus and injecting it into the prompt as additional context, thereby improving factual grounding and robustness for triage prediction.

**Chain-of-Thought (CoT)**: Augments prompts with explicit step-by-step reasoning, encouraging LLMs to generate intermediate rationales prior to producing final predictions.

**Self-Consistency (SCons)**: Extends Chain-of-Thought prompting by sampling multiple reasoning chains and selecting the most consistent outcome among them.

**Self-Contrast (SCtr)**: Improves robustness by generating multiple reasoning perspectives and reconciling their differences to derive a final decision.

**Exchange-of-Thought (EoT)**: Facilitates cross-model interaction by exchanging intermediate reasoning traces, enabling complementary reasoning processes to be integrated.

**Knowledge-Evolvable Assistant (KEA)**: A general framework for augmenting large language models. We adopt this framework to construct two specialized repositories based on the MIMIC-IV-ED dataset to support the triage prediction process: (i) an experience bank that stores validated, successful CCPG cases for analogy-based reasoning, and (ii) a reflection bank that records previously misclassified cases along with their corrections and self-summarized error analyses.

**Task Adaptation and Instruction Tuning (TAIT-LoRA)**: Utilizes instruction tuning for task adaptation. We adopt its instruction-tuning component and perform LoRA-based fine-tuning using the natural language data constructed in our work.

**TRIAGEAGENT**: A multi-agent triage framework that aggregates multiple reasoning perspectives with dynamically updated confidence scores and external evidence, serving as a strong baseline for collaborative clinical decision-making.

**BERT**: A general-purpose transformer encoder that we fine-tune for ESI classification from triage notes. 

**TCM-BERT** and **BioBERT**: Domain-specific variants of BERT pre-trained on large-scale medical and biomedical corpora, included to assess the effectiveness of specialized medical representations for ESI prediction. 

**KATE-BERT**: Extracts medical entities from clinical text and leverages the resulting structured representations for downstream triage classification. 

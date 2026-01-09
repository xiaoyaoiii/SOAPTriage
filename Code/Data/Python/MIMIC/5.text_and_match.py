import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def build_medical_prompt(data):
    """
    Construct a medical prompt template based on patient data.

    This version is designed for MIMIC-IV-ED style fields, with fallbacks
    to alternative / legacy ED dataset field names when missing.
    """
    # Unified field mapping with fallbacks
    field_map = {
        "age": ["age", "Age"],
        "gender": ["gender", "Sex"],
        "race": ["race", "Race"],
        "chiefcomplaint": ["chiefcomplaint", "Chief Complaints"],
        "acuity": ["acuity", "IMMEDR"],
        "temperature": ["temperature", "Temperature"],
        "heartrate": ["heartrate", "Pulse"],
        "resprate": ["resprate", "Respiratory Rate"],
        "o2sat": ["o2sat", "Pulse Oximetry"],
        "sbp": ["sbp", "Systolic BP"],
        "dbp": ["dbp", "Diastolic BP"],
        "pain": ["pain", "Pain Scale"],
        "arrival_transport": ["arrival_transport", "Arrived by EMS"],
        "disposition": ["disposition", "Disposition"],
        # "intime": ["intime", "Arrival Time"],
        # "outtime": ["outtime", "Departure Time"],
        "past_medical_history": ["Past Medical History", "History"],
        "allergies": ["Allergies"],
        "physical_exam": ["Physical Exam"],
    }

    # Normalize data with fallbacks
    normalized_data = {}
    for target_key, possible_keys in field_map.items():
        value = None
        for key in possible_keys:
            if key in data and data[key] not in ["", None, "None", "N/A", "Not applicable"]:
                value = data[key]
                break
        normalized_data[target_key] = value

    def safe_get(key, default="not provided"):
        """Return a cleaned string or a default if missing/empty."""
        value = normalized_data.get(key)
        if value is None or str(value).strip() == "":
            return default
        return str(value).strip()

    # Check if all vital signs are empty
    vitals_keys = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]
    all_vitals_empty = all(safe_get(k, "") == "" for k in vitals_keys)

    components = []

    # 1) Demographics and context
    age_val = safe_get("age", "unknown age")
    gender_raw = safe_get("gender", "unknown gender")
    gender_val = "Male" if gender_raw == "M" else "Female" if gender_raw == "F" else gender_raw
    race_val = safe_get("race", "unknown race")

    components.append(
        f"This is a clinical case involving a {age_val}-year-old {gender_val} "
        f"of {race_val} ethnicity, who presented to the emergency department."
    )

    # 2) Presentation details
    if (cc := safe_get("chiefcomplaint")) != "not provided":
        components.append(f'The patient\'s primary concern was: "{cc}".')

    if (transport := safe_get("arrival_transport")) != "not provided":
        components.append(f"The patient arrived via {transport}.")

    # 3) Vital signs OR physical exam (use physical exam if vitals are unavailable)
    if not all_vitals_empty:
        vitals = []
        if (temp := safe_get("temperature")) != "not provided":
            vitals.append(f"temperature {temp}°F")
        if (hr := safe_get("heartrate")) != "not provided":
            vitals.append(f"heart rate {hr}")
        if (rr := safe_get("resprate")) != "not provided":
            vitals.append(f"respiratory rate {rr}")
        if (o2 := safe_get("o2sat")) != "not provided":
            vitals.append(f"SpO₂ {o2}%")
        if (sbp := safe_get("sbp")) != "not provided" and (dbp := safe_get("dbp")) != "not provided":
            vitals.append(f"blood pressure {sbp}/{dbp} mmHg")

        if vitals:
            components.append("Initial vital signs were: " + ", ".join(vitals) + ".")

        if (pain := safe_get("pain")) != "not provided":
            components.append(f"Pain was assessed at {pain}/10.")
    else:
        if (pe := safe_get("physical_exam")) != "not provided":
            components.append(f"Physical examination revealed: {pe}.")
        else:
            components.append("No physical examination findings were documented.")

    # 4) Medical history
    if (pmh := safe_get("past_medical_history")) != "not provided":
        components.append(f"Relevant medical history includes: {pmh}.")

    if (allergies := safe_get("allergies")) != "not provided":
        components.append(f"The patient reported the following allergies: {allergies}.")

    # 5) Disposition (and optional timeline if you enable it)
    # time_details = []
    # if (intime := safe_get("intime")) != "not provided":
    #     time_details.append(f"admitted at {intime}")
    # if (outtime := safe_get("outtime")) != "not provided":
    #     time_details.append(f"discharged at {outtime}")
    # if time_details:
    #     components.append("Timeline: " + " and ".join(time_details) + ".")

    if (disp := safe_get("disposition")) != "not provided":
        components.append(f"Final disposition: {disp}.")

    return " ".join(components).strip()


def load_txt_samples(txt_path):
    """
    Load sample texts from:
    - a directory containing multiple .txt files, OR
    - a single .txt file where each non-empty line is treated as a sample.
    """
    samples = []
    if os.path.isdir(txt_path):
        for fname in os.listdir(txt_path):
            if fname.endswith(".txt"):
                with open(os.path.join(txt_path, fname), "r", encoding="utf-8") as f:
                    samples.append(f.read().strip())
    else:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(line)
    return samples


def encode(text, tokenizer, model):
    """Encode text to a single embedding vector (CLS token from last_hidden_state)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state[:, 0, :]
    return emb[0].cpu().numpy()


def find_topk(query_vec, sample_vecs, k=3):
    """Brute-force cosine similarity search and return top-k indices and scores."""
    def cosine_similarity(a, b):
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    scores = [cosine_similarity(query_vec, v) for v in sample_vecs]
    sorted_idx = np.argsort(scores)[-k:][::-1]
    return sorted_idx, [scores[i] for i in sorted_idx]


if __name__ == "__main__":
    # Replace these with your own paths
    YOUR_JSON_INPUT_PATH = "your_input_sections_json_file_path.json"
    YOUR_SAMPLE_TEXT_PATH = "your_sample_text_file_or_directory_path.txt"
    YOUR_LOCAL_EMBEDDING_MODEL_DIR = "your_local_embedding_model_directory"
    YOUR_OUTPUT_JSON_PATH = "your_output_matched_results_json_file_path.json"

    # 1) Load JSON records
    with open(YOUR_JSON_INPUT_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if not isinstance(json_data, list):
        raise TypeError("Input JSON must be a list, where each item is a patient record.")

    print(f"Loaded {len(json_data)} JSON records successfully.")

    # 2) Load sample corpus
    all_txts = load_txt_samples(YOUR_SAMPLE_TEXT_PATH)
    print(f"Loaded {len(all_txts)} sample texts.")

    # 3) Load embedding model (local)
    tokenizer = AutoTokenizer.from_pretrained(YOUR_LOCAL_EMBEDDING_MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(YOUR_LOCAL_EMBEDDING_MODEL_DIR, local_files_only=True)
    model.eval()

    # 4) Encode all sample texts
    print("Encoding sample texts...")
    sample_vecs = [encode(t, tokenizer, model) for t in all_txts]
    print("Sample encoding done.")

    # 5) Stream processing and save results
    total = len(json_data)

    with open(YOUR_OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")

        for idx, item in enumerate(json_data):
            template_txt = build_medical_prompt(item)
            query_vec = encode(template_txt, tokenizer, model)

            idxs, sims = find_topk(query_vec, sample_vecs, k=3)
            match_texts = [all_txts[i] for i in idxs]

            result_item = {
                "id": item.get("subject_id", f"unknown_{idx}"),
                "template_txt": template_txt,
                "match_text1": match_texts[0],
                "match_text2": match_texts[1],
                "match_text3": match_texts[2],
                # Support both possible field names
                "Triage": item.get("acuity", item.get("IMMEDR", "unknown")),
            }

            json_str = json.dumps(result_item, ensure_ascii=False, indent=2)
            if idx > 0:
                f.write(",\n")
            f.write(json_str)

            if (idx + 1) % 100 == 0 or idx == total - 1:
                print(f"Processed {idx+1}/{total} records (last id: {result_item['id']})")

        f.write("\n]")

    print(f"\nAll done! Results saved to: {YOUR_OUTPUT_JSON_PATH}")
    print(f"Input example (truncated):\n{json.dumps(json_data[0], indent=2)[:500]}...\n")
    print(f"Last result example:\n{json.dumps(result_item, indent=2)}")

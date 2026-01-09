import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Step 1: JSON -> template text (exclude both "id" and "IMMEDR")
def json_to_template(json_data):
    INVALID_VALUES = {
        "", "None", "none",
        "Blank", "blank",
        "Not applicable/Blank", "Not applicable",
        "NA", "N/A",
    }

    def fmt_list(v):
        if v is None:
            return None

        if isinstance(v, (list, tuple, set)):
            items = []
            for x in v:
                if x is None:
                    continue
                s = str(x).strip()
                if not s or s in INVALID_VALUES:
                    continue
                items.append(s)
            return ", ".join(items) if items else None

        s = str(v).strip()
        if not s or s in INVALID_VALUES:
            return None
        return s

    def valid(v):
        if v is None:
            return False
        if isinstance(v, (list, tuple, set)):
            return fmt_list(v) is not None
        if isinstance(v, str):
            s = v.strip()
            return bool(s) and s not in INVALID_VALUES
        return True

    parts = []

    age = json_data.get("Age", "")
    age_phrase = f"{age}-year-old" if str(age).isdigit() else (age or "age unspecified")
    sex = json_data.get("Sex", "") or "patient"

    parts.append(
        f"On {json_data.get('Month', '')} ({json_data.get('Day', '')}), "
        f"a {age_phrase} {sex} arrived at the ED at {json_data.get('Arrival Time', '')}."
    )

    parts.append(
        f"Arrival by EMS: {json_data.get('Arrived by EMS', '')}. "
        f"Transferred: {json_data.get('Transferred', '')}."
    )

    parts.append(
        "Vitals: "
        f"Temperature {json_data.get('Temperature', '')}, "
        f"Pulse {json_data.get('Pulse', '')}, "
        f"RR {json_data.get('Respiratory Rate', '')}, "
        f"BP {json_data.get('Systolic BP', '')}/{json_data.get('Diastolic BP', '')}, "
        f"O2 Sat {json_data.get('Pulse Oximetry', '')}%."
    )

    parts.append(
        f"Pain Scale: {json_data.get('Pain Scale', '')}. "
        f"Seen in last 72h: {json_data.get('Seen in last 72h', '')}."
    )

    history = fmt_list(json_data.get("History"))
    cc = fmt_list(json_data.get("Chief Complaints"))
    parts.append(
        f"History: {history or 'None'}. "
        f"Chief Complaints: {cc or 'None'}."
    )

    def add_if_valid(label, key, is_list=False):
        raw_val = json_data.get(key)
        if is_list:
            val = fmt_list(raw_val)
        else:
            if isinstance(raw_val, (list, tuple, set)):
                val = fmt_list(raw_val)
            else:
                val = raw_val

        if valid(val):
            parts.append(f"{label}: {val}.")

    add_if_valid("Episode of Care", "Episode of Care")
    add_if_valid("Injury/Poisoning/Adverse", "Injury/Poisoning/Adverse")
    add_if_valid("Injury within 72h", "Injury within 72h")
    add_if_valid("Injury Intent", "Injury Intent")
    add_if_valid("Injury Encounter Type", "Injury Encounter Type")
    add_if_valid("External Causes", "External Causes", is_list=True)

    return " ".join(parts)


def load_txt_samples(txt_path):
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs)[0][:, 0, :]
    return embeddings[0].cpu().numpy()


def find_topk(query_vec, sample_vecs, k=3):
    def cosine_similarity(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scores = [cosine_similarity(query_vec, v) for v in sample_vecs]
    sorted_idx = np.argsort(scores)[-k:][::-1]
    return sorted_idx, [scores[i] for i in sorted_idx]


if __name__ == "__main__":
    INPUT_JSON_PATH = "/path/to/your/INPUT.json"
    TXT_SAMPLES_PATH = "/path/to/your/TXT_SAMPLES.txt"
    MODEL_DIR = "/path/to/your/bge-base-en-v1.5"
    OUTPUT_JSON_PATH = "/path/to/your/OUTPUT.json"

    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if not isinstance(json_data, list):
        raise TypeError("Expected a list of patient records in the JSON file.")

    all_txts = load_txt_samples(TXT_SAMPLES_PATH)
    print(f"Loaded {len(all_txts)} sample texts.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True)

    sample_vecs = [encode(t, tokenizer, model) for t in all_txts]

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        f.write("[\n")

        for idx, item in enumerate(json_data):
            template_txt = json_to_template(item)
            query_vec = encode(template_txt, tokenizer, model)
            idxs, sims = find_topk(query_vec, sample_vecs, k=3)
            match_texts = [all_txts[i] for i in idxs]

            result_item = {
                "id": item.get("id"),
                "template_txt": template_txt,
                "match_text1": match_texts[0],
                "match_text2": match_texts[1],
                "match_text3": match_texts[2],
                "Triage": item.get("IMMEDR", ""),
            }

            json_str = json.dumps(result_item, ensure_ascii=False, indent=2)

            if idx > 0:
                f.write(",\n")
            f.write(json_str)

            if idx % 100 == 0:
                print(f"Processed {idx+1}/{len(json_data)}")

        f.write("\n]")

    print("Done. Output written to:", OUTPUT_JSON_PATH)

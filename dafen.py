import json
import os
from datetime import datetime

import streamlit as st

# =========================
# Config
# =========================
st.set_page_config(page_title="SOAPTriage Expert Rating", layout="wide")

DIMENSIONS = [
    ("clinical_consistency", "Clinical Consistency"),
    ("factual_correctness", "Factual Correctness"),
    ("narrative_naturalness", "Narrative Naturalness"),
    ("information_completeness", "Information Completeness"),
    ("readability_clarity", "Readability & Clarity"),
]

# =========================
# Helpers
# =========================
def load_json_from_uploader(uploaded_file):
    raw = uploaded_file.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")
    # minimal validation
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object.")
        for k in ("id", "answer", "label"):
            if k not in item:
                raise ValueError(f"Item {i} missing required field: {k}")
    return data

def read_completed_ids(jsonl_path: str):
    completed = set()
    if not jsonl_path or not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    completed.add(obj["id"])
            except Exception:
                # ignore broken lines
                pass
    return completed

def append_jsonl(jsonl_path: str, obj: dict):
    os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def ensure_state():
    defaults = {
        "data": None,
        "idx": 0,
        "dim_idx": 0,
        "scores": {},
        "error": "",
        "annotator": "",
        "output_path": "ratings.jsonl",
        "completed_ids": set(),
        "score_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def skip_completed():
    # move idx forward if already completed
    data = st.session_state["data"] or []
    completed = st.session_state["completed_ids"]
    while st.session_state["idx"] < len(data) and data[st.session_state["idx"]]["id"] in completed:
        st.session_state["idx"] += 1

def handle_submit_score():
    raw = st.session_state.get("score_input", "").strip()
    if raw == "":
        return

    if not raw.isdigit():
        st.session_state["error"] = "Please enter an integer from 1 to 5."
        return

    val = int(raw)
    if val < 1 or val > 5:
        st.session_state["error"] = "Score must be between 1 and 5."
        return

    st.session_state["error"] = ""

    dim_key, _ = DIMENSIONS[st.session_state["dim_idx"]]
    st.session_state["scores"][dim_key] = val

    # clear input and advance to next dimension / next item
    st.session_state["score_input"] = ""
    st.session_state["dim_idx"] += 1

    # if finished all 5 dims, save and move to next example
    if st.session_state["dim_idx"] >= len(DIMENSIONS):
        data = st.session_state["data"]
        idx = st.session_state["idx"]
        item = data[idx]

        record = {
            # keep original fields (including label) — IMPORTANT
            **item,
            "annotator": st.session_state["annotator"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ratings": st.session_state["scores"],
        }
        append_jsonl(st.session_state["output_path"], record)

        # mark completed locally to enable skip
        st.session_state["completed_ids"].add(item["id"])

        # advance to next item and reset
        st.session_state["idx"] += 1
        st.session_state["dim_idx"] = 0
        st.session_state["scores"] = {}

        skip_completed()

# =========================
# UI
# =========================
ensure_state()

st.title("Expert Rating Tool (1–5)")

with st.sidebar:
    st.subheader("Setup")

    st.session_state["annotator"] = st.text_input(
        "Annotator ID (required)",
        value=st.session_state["annotator"],
        placeholder="e.g., expert_01",
    )

    uploaded = st.file_uploader("Upload input JSON (list of objects)", type=["json"])

    st.session_state["output_path"] = st.text_input(
        "Output JSONL path",
        value=st.session_state["output_path"],
        help="Each completed example will be appended as one JSON line.",
    )

    if st.button("Load / Reload"):
        if not uploaded:
            st.error("Please upload a JSON file first.")
        elif not st.session_state["annotator"].strip():
            st.error("Please fill in Annotator ID.")
        else:
            try:
                data = load_json_from_uploader(uploaded)
                st.session_state["data"] = data
                st.session_state["idx"] = 0
                st.session_state["dim_idx"] = 0
                st.session_state["scores"] = {}
                st.session_state["error"] = ""
                st.session_state["score_input"] = ""

                st.session_state["completed_ids"] = read_completed_ids(st.session_state["output_path"])
                skip_completed()
                st.success(f"Loaded {len(data)} items. Resuming from the first unfinished item.")
            except Exception as e:
                st.error(f"Failed to load JSON: {e}")

    st.divider()
    if st.session_state["output_path"] and os.path.exists(st.session_state["output_path"]):
        with open(st.session_state["output_path"], "rb") as f:
            st.download_button(
                "Download current JSONL",
                data=f,
                file_name=os.path.basename(st.session_state["output_path"]),
                mime="application/jsonl",
            )

# Must have data
data = st.session_state["data"]
if not data:
    st.info("Upload a JSON file in the sidebar and click **Load / Reload**.")
    st.stop()

skip_completed()

# Finished?
if st.session_state["idx"] >= len(data):
    st.success("All items have been rated. ✅")
    st.stop()

item = data[st.session_state["idx"]]

# Display current item (DO NOT show label)
left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown(f"### Item {st.session_state['idx'] + 1} / {len(data)}")
    st.caption(f"ID: {item['id']}")
    st.markdown("**Case Narrative**")
    st.write(item["answer"])

with right:
    st.markdown("### Rating")
    st.progress((st.session_state["dim_idx"]) / len(DIMENSIONS))

    dim_key, dim_name = DIMENSIONS[st.session_state["dim_idx"]]
    st.markdown(f"**Dimension:** {dim_name}")
    st.caption("Enter an integer 1–5, then press Enter to continue.")

    if st.session_state["error"]:
        st.error(st.session_state["error"])

    # single input, step-by-step (Enter triggers on_change)
    st.text_input(
        "Score (1–5)",
        key="score_input",
        on_change=handle_submit_score,
        placeholder="1-5",
    )

    # show already-filled scores for this item (optional, not required but useful)
    if st.session_state["scores"]:
        st.markdown("**Current scores (this item):**")
        for k, v in st.session_state["scores"].items():
            pretty = dict(DIMENSIONS).get(k, k)
            st.write(f"- {pretty}: {v}")

    st.divider()
    if st.button("Reset current item scores"):
        st.session_state["dim_idx"] = 0
        st.session_state["scores"] = {}
        st.session_state["score_input"] = ""
        st.session_state["error"] = ""
        st.success("Reset done.")

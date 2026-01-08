#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED data -> structured JSON (including all legacy + new fields)
python build_ed_json.py
"""
import pandas as pd
import json
import re
import os

# ---------- 1. Path configuration ----------
INPUT_EXCEL_PATH = "/path/to/your/INPUT.xlsx"
RFV_VOCAB_PATH = "/path/to/your/RFV.txt"
EXT_CAUSE_PATH = "/path/to/your/icd10cm-codes-April-2024.txt"
OUTPUT_JSON_PATH = "/path/to/your/OUTPUT.json"

# ---------- 2. Load vocabularies ----------
def load_rfv_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                vocab[parts[0]] = parts[1]
    return vocab


def load_ext_cause_vocab(path):
    vocab = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line, 1)
            if len(parts) != 2:
                continue
            code, desc = parts

            code4 = re.sub(r"[^A-Z0-9]", "", code.upper())[:4]
            desc = re.split(r",", desc, 1)[0].strip()

            vocab[code4] = desc
    return vocab


rfv_vocab = load_rfv_vocab(RFV_VOCAB_PATH)
cause_vocab = load_ext_cause_vocab(EXT_CAUSE_PATH)

# ---------- 3. Value mappings ----------
MONTH_MAP = {
    str(i).zfill(2): name
    for i, name in enumerate(
        [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
    )
    if i
}

DAY_MAP = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}

YN_MAP = {-9: "Blank", -8: "Unknown", -7: "Not applicable", 1: "Yes", 2: "No"}
SEX_MAP = {1: "Female", 2: "Male"}

IMMEDR_MAP = {
    -9: "Blank",
    -8: "Unknown",
    0: "No triage (but ESA conducts nursing triage)",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    7: "No nursing triage at this ESA",
}

EPISODE_MAP = {
    -9: "Blank",
    -8: "Unknown",
    1: "Initial visit to this ED",
    2: "Follow-up visit to this ED",
}

INJ_MAP = {
    -9: "Blank",
    -8: "Unknown",
    1: "Yes, injury/trauma",
    2: "Yes, overdose/poisoning",
    3: "Yes, adverse effect of medical/surgical treatment",
    4: "No",
    5: "Questionable injury status",
}

INTENT_MAP = {
    -9: "Blank",
    -8: "Unknown/intent unclear",
    1: "Intentional",
    2: "Unintentional",
    3: "Intent unclear",
    4: "Questionable injury status",
}

ENC_MAP = {
    -9: "Not applicable/Blank",
    1: "Initial encounter",
    2: "Subsequent encounter",
    3: "Sequela encounter",
    4: "Both initial and subsequent",
    5: "Both initial and sequela",
    6: "Both subsequent and sequela",
    7: "Initial, subsequent, and sequela",
}

# ---------- 4. External cause lookup (prefix match) ----------
def lookup_cause(raw):
    """Return (4-char code, description) or (None, None)."""
    if pd.isna(raw) or raw == "" or raw == -9:
        return None, None

    try:
        if isinstance(raw, bytes) or (isinstance(raw, str) and raw.startswith("b'")):
            raw = eval(raw).decode() if raw.startswith("b'") else raw
    except Exception:
        pass

    raw = str(raw).strip().upper()
    raw = re.sub(r"[^A-Z0-9]", "", raw)[:4]
    if len(raw) < 3:
        return None, None

    desc = cause_vocab.get(raw)
    return (raw, desc) if desc else (None, None)


# ---------- 5. Core conversion ----------
def convert_row(row):
    d = {"id": row.get("id")}

    d["Month"] = MONTH_MAP.get(str(row.get("VMONTH", "")).zfill(2), str(row.get("VMONTH", "")))
    d["Day"] = DAY_MAP.get(row.get("VDAYR"), str(row.get("VDAYR")))

    arrt = str(row.get("ARRTIME", ""))
    m = re.search(r"(\d{4})", arrt)
    d["Arrival Time"] = f"{m.group(1)[:2]}:{m.group(1)[2:]}" if m else "Blank"

    age = row.get("AGE")
    d["Age"] = "Under 1 year" if age == 0 else str(age)

    sex = row.get("SEX")
    d["Sex"] = SEX_MAP.get(int(sex), str(sex)) if pd.notna(sex) else "Blank"

    d["Arrived by EMS"] = YN_MAP.get(row.get("ARREMS"), str(row.get("ARREMS")))
    d["Transferred"] = YN_MAP.get(row.get("AMBTRANSFER"), str(row.get("AMBTRANSFER")))

    d["Temperature"] = "Blank" if row.get("TEMPF") == -9 else f"{row.get('TEMPF', -9)/10:.1f}°F"
    d["Pulse"] = (
        "Blank"
        if row.get("PULSE") == -9
        else ("Doppler" if row.get("PULSE") == 998 else str(row.get("PULSE")))
    )
    d["Respiratory Rate"] = "Blank" if row.get("RESPR") == -9 else str(row.get("RESPR"))
    d["Systolic BP"] = "Blank" if row.get("BPSYS") == -9 else str(row.get("BPSYS"))
    d["Diastolic BP"] = (
        "Blank"
        if row.get("BPDIAS") == -9
        else ("Palp/Doppler" if row.get("BPDIAS") == 998 else str(row.get("BPDIAS")))
    )
    d["Pulse Oximetry"] = "Blank" if row.get("POPCT") == -9 else str(row.get("POPCT"))

    d["Pain Scale"] = {-9: "Blank", -8: "Unknown"}.get(row.get("PAINSCALE"), str(row.get("PAINSCALE")))
    d["Seen in last 72h"] = YN_MAP.get(row.get("SEEN72"), str(row.get("SEEN72")))

    imr = row.get("IMMEDR")
    if pd.isna(imr):
        d["IMMEDR"] = "Blank"
    else:
        d["IMMEDR"] = IMMEDR_MAP.get(int(imr), str(imr))

    hist_fields = [
        "ASTHMA", "COPD", "CHF", "CAD", "HTN", "DIABTYP1", "DIABTYP2",
        "OBESITY", "OSA", "OSTPRSIS", "SUBSTAB", "ETOHAB", "CANCER",
        "CEBVD", "CKD", "DEPRN", "DIABTYP0", "ESRD", "HPE", "HYPLIPID", "ALZHD",
    ]
    d["History"] = [f for f in hist_fields if row.get(f) == 1] or None

    rfv_fields = ["RFV1", "RFV2", "RFV3", "RFV4", "RFV5", "RFV13D", "RFV23D", "RFV33D", "RFV43D", "RFV53D"]
    complaints = [rfv_vocab[str(v)] for f in rfv_fields if (v := row.get(f)) is not None and str(v) in rfv_vocab]
    if complaints:
        d["Chief Complaints"] = complaints

    d["Episode of Care"] = EPISODE_MAP.get(row.get("EPISODE"), str(row.get("EPISODE")))
    d["Injury/Poisoning/Adverse"] = INJ_MAP.get(row.get("INJPOISAD"), str(row.get("INJPOISAD")))
    d["Injury within 72h"] = YN_MAP.get(row.get("INJURY72"), str(row.get("INJURY72")))
    d["Injury Intent"] = INTENT_MAP.get(row.get("INTENT15"), str(row.get("INTENT15")))
    d["Injury Encounter Type"] = ENC_MAP.get(row.get("INJURY_ENC"), str(row.get("INJURY_ENC")))

    causes = []
    for fld in ("CAUSE1", "CAUSE2", "CAUSE3"):
        code, desc = lookup_cause(row.get(fld))
        if code and desc:
            causes.append(desc)
    if causes:
        d["External Causes"] = causes

    return d


# ---------- 6. Batch processing ----------
def main():
    df = pd.read_excel(INPUT_EXCEL_PATH)

    valid_records = []
    for _, row in df.iterrows():
        im = row.get("IMMEDR")
        if pd.notna(im) and int(im) in {1, 2, 3, 4, 5}:
            valid_records.append(convert_row(row))

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(valid_records, f, ensure_ascii=False, indent=2)

    print(f"done -> {OUTPUT_JSON_PATH}  total {len(valid_records)} records (valid IMMEDR)")


if __name__ == "__main__":
    main()

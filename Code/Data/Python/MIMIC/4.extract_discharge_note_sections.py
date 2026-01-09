#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
from collections import defaultdict

# Replace with your own file paths
YOUR_INPUT_CSV_PATH = "your_input_csv_file_path.csv"
YOUR_OUTPUT_CSV_PATH = "your_output_csv_file_path.csv"
TEXT_COL = "text"

# 1) Section header lines must be a full line ending with ":" (or "：").
# Require header to start with a letter to avoid false positives like "VS:" or "10:25PM".
BOUNDARY_HEADER_RE = re.compile(
    r"^\s*(?P<h>[A-Za-z][A-Za-z0-9 \-/&\(\)\.]{0,100})\s*[:：]\s*$"
)


def canon_header(h: str) -> str:
    h = h.strip()
    h = " ".join(h.split())
    return h.upper()


# 2) Target output columns and their possible header synonyms (canonicalized to uppercase)
TARGETS = {
    "Past Medical History": {
        "PAST MEDICAL HISTORY",
    },
    "Allergies": {
        "ALLERGIES",
    },
    # "History of Present Illness": {
    #     "HISTORY OF PRESENT ILLNESS",
    #     "HISTORY OF PRESENTING ILLNESS",
    #     "HPI",
    # },
    "Physical Exam": {
        "PHYSICAL EXAM",
        "PHYSICAL EXAMINATION",
        "ADMISSION PHYSICAL EXAM",
        "ADMISSION PHYSICAL EXAMINATION",
        "DISCHARGE PHYSICAL EXAM",
        "DISCHARGE PHYSICAL EXAMINATION",
        "DISCHARGE PE",
    },
    "Family History": {
        "FAMILY HISTORY",
    },
    # "Chief Complaint": {
    #     "CHIEF COMPLAINT",
    # },
}

# Reverse index: canonical header -> output column name
HEADER2COL = {}
for col, hs in TARGETS.items():
    for h in hs:
        HEADER2COL[h] = col


def extract_sections(note_text: str) -> dict:
    """
    Return a dict: target column -> extracted content.
    If a section appears multiple times, join them with a blank line.
    """
    out = {k: "" for k in TARGETS.keys()}
    if not isinstance(note_text, str) or not note_text.strip():
        return out

    current_col = None
    buf = []
    collected = defaultdict(list)

    def flush():
        nonlocal buf, current_col
        if current_col is not None:
            text = "\n".join(buf).strip()
            if text:
                collected[current_col].append(text)
        buf = []

    for line in note_text.splitlines():
        m = BOUNDARY_HEADER_RE.match(line)
        if m:
            # New header line: close previous section first
            flush()
            hdr = canon_header(m.group("h"))
            # If the header is not in targets, stop collecting until the next target header appears
            current_col = HEADER2COL.get(hdr, None)
            continue

        # Normal content line
        if current_col is not None:
            buf.append(line)

    flush()

    # Join repeated sections
    for col in out.keys():
        if col in collected:
            out[col] = "\n\n".join(collected[col]).strip()

    return out


def main():
    chunksize = 5000
    first = True

    for chunk in pd.read_csv(
        YOUR_INPUT_CSV_PATH,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    ):
        # Extract target sections into separate columns
        extracted = chunk[TEXT_COL].apply(extract_sections).apply(pd.Series)

        # Output: keep original columns except text + add extracted columns
        base = chunk.drop(columns=[TEXT_COL], errors="ignore")
        out_df = pd.concat([base, extracted], axis=1)

        out_df.to_csv(
            YOUR_OUTPUT_CSV_PATH,
            index=False,
            mode="w" if first else "a",
            header=first,
            encoding="utf-8-sig",
        )
        first = False

    print(f"Saved: {YOUR_OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()

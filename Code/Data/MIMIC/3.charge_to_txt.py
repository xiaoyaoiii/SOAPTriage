import pandas as pd

# Replace these with your own file paths
your_sample_csv_path = "your_sample_csv_file_path.csv"
your_discharge_csv_path = "your_discharge_csv_file_path.csv"
your_output_csv_path = "your_output_csv_file_path.csv"


def normalize_ids(df: pd.DataFrame, cols=("subject_id", "hadm_id")) -> pd.DataFrame:
    """Normalize ID columns to integer-like strings to avoid merge mismatches (e.g., '10000032.0', scientific notation)."""
    # Clean column names (avoid issues with whitespace/BOM)
    df.columns = df.columns.str.strip()

    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Existing columns: {list(df.columns)}")

        # Read as string and strip
        s = df[c].astype("string").str.strip()

        # Convert to numeric -> Int64 -> string
        # This normalizes "10000032.0" and "10000032" to "10000032"
        n = pd.to_numeric(s, errors="coerce")
        df[c] = n.astype("Int64").astype("string")

    return df


def join_nonempty(vals, sep: str) -> str:
    """Join non-empty, non-'nan' values with a separator."""
    cleaned = []
    for v in vals:
        sv = str(v).strip()
        if sv != "" and sv.lower() != "nan":
            cleaned.append(sv)
    return sep.join(cleaned)


# Read inputs (force IDs as string; note_id must be string like '10000032-DS-21')
sample = pd.read_csv(
    your_sample_csv_path,
    dtype={"subject_id": "string", "hadm_id": "string"},
    keep_default_na=False,
)
discharge = pd.read_csv(
    your_discharge_csv_path,
    dtype={"subject_id": "string", "hadm_id": "string", "note_id": "string"},
    keep_default_na=False,
)

sample = normalize_ids(sample)
discharge = normalize_ids(discharge)

# Keep only required columns from discharge
required_discharge_cols = ["subject_id", "hadm_id", "note_id", "text"]
missing = [c for c in required_discharge_cols if c not in discharge.columns]
if missing:
    raise ValueError(f"your_discharge_csv_file is missing columns: {missing}. Existing columns: {list(discharge.columns)}")

discharge = discharge[required_discharge_cols].copy()

# Aggregate discharge notes first to avoid row explosion after merge
discharge_agg = (
    discharge.groupby(["subject_id", "hadm_id"], as_index=False)
    .agg(
        {
            "note_id": lambda x: join_nonempty(x, ";"),
            "text": lambda x: join_nonempty(x, "\n\n-----\n\n"),
        }
    )
)

# Merge with indicator for diagnostics
merged = sample.merge(discharge_agg, on=["subject_id", "hadm_id"], how="left", indicator=True)

print(merged["_merge"].value_counts(dropna=False))  # left_only / both
matched_rows = (merged["_merge"] == "both").sum()
print("Matched rows:", matched_rows)

# Put new columns at the end
orig_cols = list(sample.columns)
merged = merged.drop(columns=["_merge"])
merged = merged[orig_cols + ["note_id", "text"]]

merged.to_csv(your_output_csv_path, index=False, encoding="utf-8-sig")
print("Saved:", your_output_csv_path)

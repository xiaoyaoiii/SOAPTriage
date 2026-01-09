import argparse
from pathlib import Path
import pandas as pd


def normalize_id(series: pd.Series) -> pd.Series:
    """Prevent ID mismatches caused by values like 10000032.0 or scientific notation."""
    s = series.copy()
    num = pd.to_numeric(s, errors="coerce")
    mask = num.notna()

    out = pd.Series([None] * len(s), index=s.index, dtype="object")
    out.loc[mask] = num.loc[mask].round().astype("Int64").astype(str)
    out.loc[~mask] = s.loc[~mask].astype(str).str.strip()
    out = out.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Add an 'age' column to your_a_csv_file using anchor_age/anchor_year from your_b_csv_file."
    )
    parser.add_argument(
        "a_csv",
        help="Path to your_a_csv_file (must include: subject_id, intime)"
    )
    parser.add_argument(
        "b_csv",
        help="Path to your_b_csv_file (must include: subject_id, anchor_age, anchor_year)"
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to your_output_csv_file (default: your_a_csv_file_with_age.csv)"
    )
    args = parser.parse_args()

    a_path = Path(args.a_csv)
    b_path = Path(args.b_csv)
    out_path = Path(args.output) if args.output else a_path.with_name(a_path.stem + "_with_age.csv")

    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    a.columns = [c.strip() for c in a.columns]
    b.columns = [c.strip() for c in b.columns]

    for col in ["subject_id", "intime"]:
        if col not in a.columns:
            raise ValueError(f"your_a_csv_file is missing column: {col}; existing columns: {list(a.columns)}")
    for col in ["subject_id", "anchor_age", "anchor_year"]:
        if col not in b.columns:
            raise ValueError(f"your_b_csv_file is missing column: {col}; existing columns: {list(b.columns)}")

    # Normalize subject_id to avoid mismatches like 10000032 vs 10000032.0
    a["subject_id"] = normalize_id(a["subject_id"])
    b["subject_id"] = normalize_id(b["subject_id"])

    # Build per-subject mapping (if duplicates exist in your_b_csv_file, keep the first row)
    b_first = b.drop_duplicates("subject_id", keep="first").copy()
    b_first["anchor_age"] = pd.to_numeric(b_first["anchor_age"], errors="coerce")
    b_first["anchor_year"] = pd.to_numeric(b_first["anchor_year"], errors="coerce")

    age_map = b_first.set_index("subject_id")["anchor_age"]
    year_map = b_first.set_index("subject_id")["anchor_year"]

    # Map anchor fields onto your_a_csv_file
    a["_anchor_age"] = a["subject_id"].map(age_map)
    a["_anchor_year"] = a["subject_id"].map(year_map)

    # Extract year from intime
    intime_year = pd.to_datetime(a["intime"], errors="coerce").dt.year

    # Compute age
    a["age"] = a["_anchor_age"] + (intime_year - a["_anchor_year"])

    # Drop temporary columns
    a.drop(columns=["_anchor_age", "_anchor_year"], inplace=True)

    a.to_csv(out_path, index=False)
    print(f"Done. Output: {out_path} rows={len(a)}")


if __name__ == "__main__":
    main()

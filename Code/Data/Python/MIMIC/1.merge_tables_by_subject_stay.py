import argparse
from pathlib import Path
import pandas as pd


A_COLS = [
    "subject_id", "stay_id",
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "pain", "acuity", "chiefcomplaint",
]

B_COLS = [
    "subject_id", "hadm_id", "stay_id",
    "intime", "outtime", "gender", "race",
    "arrival_transport", "disposition",
]


def read_table(path: Path, sheet_name=0, sep=None) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if ext in [".csv", ".tsv", ".txt"]:
        if sep is None:
            sep = "\t" if ext in [".tsv", ".txt"] else ","
        return pd.read_csv(path, sep=sep)
    raise ValueError(
        f"Unsupported file type: {ext} (supported: xlsx/xls/csv/tsv/txt)"
    )


def write_table(df: pd.DataFrame, path: Path, sep=None) -> None:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df.to_excel(path, index=False)
        return
    if ext in [".csv", ".tsv", ".txt"]:
        if sep is None:
            sep = "\t" if ext in [".tsv", ".txt"] else ","
        df.to_csv(path, index=False, sep=sep)
        return
    raise ValueError(f"Unsupported output file type: {ext}")


def normalize_id(series: pd.Series) -> pd.Series:
    """
    Normalize IDs to avoid merge failures caused by Excel/CSV parsing, e.g.:
    - "10000935.0"
    - "3.2845079e+07"

    Strategy:
    1) Try converting to numeric (handles scientific notation / trailing .0)
    2) Round to integer and convert to string
    3) Fallback to stripped strings for non-numeric values
    4) Remove any trailing ".0"
    """
    s = series.copy()

    num = pd.to_numeric(s, errors="coerce")
    mask = num.notna()

    out = pd.Series([None] * len(s), index=s.index, dtype="object")
    out.loc[mask] = num.loc[mask].round().astype("Int64").astype(str)

    out.loc[~mask] = s.loc[~mask].astype(str).str.strip()
    out = out.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    return out


def ensure_columns(df: pd.DataFrame, required_cols: list, table_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}\n"
            f"Existing columns: {list(df.columns)}"
        )


def _parse_sheet(value):
    """
    Accept either a sheet index (e.g. 0, 1, 2) or a sheet name (string).
    If user passes digits like "0", convert to int to avoid treating it as a sheet name.
    """
    if isinstance(value, int):
        return value
    s = str(value).strip()
    return int(s) if s.isdigit() else s


def main():
    parser = argparse.ArgumentParser(
        description="Merge Table A and Table B by subject_id + stay_id, then output a de-duplicated merged table."
    )
    parser.add_argument("a", help="Path to your_table_a_file (xlsx/csv/tsv/txt)")
    parser.add_argument("b", help="Path to your_table_b_file (xlsx/csv/tsv/txt)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to your_output_file (default: auto-generate '*_merged.xxx' next to your_table_a_file)"
    )
    parser.add_argument(
        "--how",
        default="inner",
        choices=["inner", "left", "right", "outer"],
        help="Merge mode: inner=matched only; left=keep all rows from Table A; right; outer (default: inner)"
    )
    parser.add_argument(
        "--sheet_a",
        default=0,
        help="Excel sheet name/index for Table A (xlsx only). Examples: 0 or 'Sheet1'"
    )
    parser.add_argument(
        "--sheet_b",
        default=0,
        help="Excel sheet name/index for Table B (xlsx only). Examples: 0 or 'Sheet1'"
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Delimiter for csv/tsv/txt (default: comma for csv, tab for tsv/txt)"
    )
    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)

    out_path = Path(args.output) if args.output else a_path.with_name(a_path.stem + "_merged" + a_path.suffix)

    sheet_a = _parse_sheet(args.sheet_a)
    sheet_b = _parse_sheet(args.sheet_b)

    # Read files
    df_a = read_table(a_path, sheet_name=sheet_a, sep=args.sep)
    df_b = read_table(b_path, sheet_name=sheet_b, sep=args.sep)

    # Trim whitespace in column names
    df_a.columns = [c.strip() for c in df_a.columns]
    df_b.columns = [c.strip() for c in df_b.columns]

    # Validate required columns
    ensure_columns(df_a, A_COLS, "Table A")
    ensure_columns(df_b, B_COLS, "Table B")

    # Keep only needed columns
    df_a = df_a[A_COLS].copy()
    df_b = df_b[B_COLS].copy()

    # Normalize join keys to avoid mismatches like 10000935 vs 10000935.0
    for k in ["subject_id", "stay_id"]:
        df_a[k] = normalize_id(df_a[k])
        df_b[k] = normalize_id(df_b[k])

    # Merge
    merged = df_a.merge(
        df_b,
        on=["subject_id", "stay_id"],
        how=args.how,
        suffixes=("", "_b"),
        validate=None,  # If you're sure it's one-to-one, you can set validate="one_to_one"
    )

    # Output column order: all A cols + B cols excluding duplicate keys
    out_cols = A_COLS + [c for c in B_COLS if c not in ["subject_id", "stay_id"]]
    merged = merged[out_cols]

    write_table(merged, out_path, sep=args.sep)
    print(
        f"Done. Table A rows={len(df_a)}, Table B rows={len(df_b)}, "
        f"output rows={len(merged)} -> {out_path}"
    )


if __name__ == "__main__":
    main()

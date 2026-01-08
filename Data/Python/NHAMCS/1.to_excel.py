from pathlib import Path
import pandas as pd

# Replace with your own paths
YOUR_INPUT_SAS_PATH = Path("your_input_sas_file_path.sas7bdat")
YOUR_OUTPUT_XLSX_PATH = Path("your_output_excel_file_path.xlsx")

df = pd.read_sas(YOUR_INPUT_SAS_PATH)
df.reset_index(drop=True, inplace=True)

# Insert a 1-based sequential ID as the first column
df.insert(0, "id", range(1, len(df) + 1))

# Ensure output directory exists
YOUR_OUTPUT_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(YOUR_OUTPUT_XLSX_PATH, index=False)

import pandas as pd

# Columns to keep in the filtered dataset
columns_to_keep = [
    # SOAP: Subjective, Objective, Assessment, Plan

    # Unique identifier
    "id",

    # Basic information
    "VMONTH",      # Visit month
    "VDAYR",       # Day of week (1=Sunday ... 7=Saturday)
    "ARRTIME",     # Arrival time (HHMM, 24-hour format)
    "AGE",         # Patient age (years)
    "SEX",         # Sex (1=Female, 2=Male)
    "ARREMS",      # Arrived by ambulance (1=Yes, 2=No)
    "AMBTRANSFER", # Transferred from another hospital/ED (1=Yes, 2=No)
    "SEEN72",      # Seen in this ED within past 72 hours (1=Yes, 2=No)

    # Subjective: reason for visit / chief complaint
    "RFV1", "RFV2", "RFV3", "RFV4", "RFV5",
    "RFV13D", "RFV23D", "RFV33D", "RFV43D", "RFV53D",
    "EPISODE",     # Visit type (1=Initial, 2=Follow-up)
    "INJURY",      # Injury-related visit (1=Yes, 0=No)
    "CAUSE1",      # Injury cause 1 (ICD-10-CM external cause V-Y)
    "CAUSE2",      # Injury cause 2
    "CAUSE3",      # Injury cause 3
    "INJPOISAD",   # Injury/poisoning/adverse event subtype
    "INJURY72",    # Occurred within 72 hours before visit (1=Yes, 2=No)
    "INTENT15",    # Intent (1=Intentional, 2=Unintentional, 3=Undetermined)
    "INJURY_ENC",  # Injury encounter type (1=Initial, 2=Subsequent, 3=Sequela)

    # Objective: comorbidities / medical history
    "ASTHMA", "COPD", "CHF", "CAD", "HTN",
    "DIABTYP1", "DIABTYP2", "OBESITY", "OSA", "OSTPRSIS",
    "SUBSTAB", "ETOHAB", "CANCER", "CEBVD", "CKD",
    "DEPRN", "DIABTYP0", "ESRD", "HPE", "HYPLIPID", "ALZHD",

    # Objective: vitals
    "TEMPF",       # Temperature (F)
    "PULSE",       # Pulse (beats/min)
    "RESPR",       # Respiration rate (breaths/min)
    "BPSYS",       # Systolic BP (mmHg)
    "BPDIAS",      # Diastolic BP (mmHg)
    "POPCT",       # Oxygen saturation (%)
    "PAINSCALE",   # Pain scale (0-10)

    # Target label
    "IMMEDR",      # Triage level (1-5)
]

# Replace with your own paths
YOUR_INPUT_EXCEL_PATH = "your_input_excel_file_path.xlsx"
YOUR_OUTPUT_EXCEL_PATH = "your_output_excel_file_path.xlsx"

# Load the original Excel file
df = pd.read_excel(YOUR_INPUT_EXCEL_PATH)

# Optional: validate columns exist
missing_cols = [c for c in columns_to_keep if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"Missing columns in your_input_excel_file: {missing_cols}\n"
        f"Existing columns: {list(df.columns)}"
    )

# Keep only the specified columns (in the specified order)
df_filtered = df[columns_to_keep].copy()

# Save the filtered dataset
df_filtered.to_excel(YOUR_OUTPUT_EXCEL_PATH, index=False)
print(f"Saved filtered dataset to: {YOUR_OUTPUT_EXCEL_PATH}")

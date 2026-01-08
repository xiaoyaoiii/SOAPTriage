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

    # Subjective: chief complaint
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

    # Objective: medical history (comorbidities)
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
    "IMMEDR"       # Triage level (1=Immediate, 2=Emergent, 3=Urgent, 4=Semi-urgent, 5=Non-urgent)
]

# Input and output paths (placeholders for GitHub)
INPUT_EXCEL_PATH = "/path/to/your/INPUT.xlsx"
OUTPUT_EXCEL_PATH = "/path/to/your/OUTPUT.xlsx"

# Load the original Excel file
df = pd.read_excel(INPUT_EXCEL_PATH)

# Keep only the specified columns
df_filtered = df[columns_to_keep]

# Save the filtered dataset
df_filtered.to_excel(OUTPUT_EXCEL_PATH, index=False)

import os
import numpy as np
import pandas as pd

# Path to input file (CSV or Excel) and output directory
INPUT_PATH  = r"C:\Users\shaik\OneDrive\Desktop\Capstone\train_data (version 1).csv"
OUTPUT_DIR  = r"C:\Users\shaik\OneDrive\Desktop\Capstone\output"

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_date(s):
    """Convert a series to datetime, coercing invalid values to NaT."""
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def months_between(start, end):
    """
    Calculate the difference in full months between two dates.
    Returns NaN if either date is missing.
    Floors negative values at 0.
    """
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return max((end.year - start.year) * 12 + (end.month - start.month), 0)

# --- Step 1: Load raw data ---
df = pd.read_csv(INPUT_PATH, low_memory=False)
print(f"[RAW] rows={len(df):,}, unique Emp_IDs={df['Emp_ID'].nunique():,}")

# --- Step 2: Standardize and clean columns ---
# Ensure mandatory columns exist
for col in ["Emp_ID", "MMM-YY", "Dateofjoining"]:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

# Create LastWorkingDate column if it is missing
if "LastWorkingDate" not in df.columns:
    df["LastWorkingDate"] = np.nan

# Parse date fields
df["MMM-YY"] = to_date(df["MMM-YY"])
df["Dateofjoining"] = to_date(df["Dateofjoining"])
df["LastWorkingDate"] = to_date(df["LastWorkingDate"])

# Attrition = Yes if a LastWorkingDate exists, otherwise No
df["Attrition"] = np.where(df["LastWorkingDate"].notna(), "Yes", "No")

# End Date = LastWorkingDate if attrited, else latest snapshot date
df["End Date"] = np.where(df["Attrition"].eq("Yes"), df["LastWorkingDate"], df["MMM-YY"])
df["End Date"] = to_date(df["End Date"])

# Calculate tenure in months from joining to end date
df["TenureMonths"] = df.apply(lambda r: months_between(r["Dateofjoining"], r["End Date"]), axis=1).astype("Int64")

print(f"[CLEAN (panel)] rows={len(df):,}, unique Emp_IDs={df['Emp_ID'].nunique():,}")

# Save cleaned monthly panel
panel_csv = os.path.join(OUTPUT_DIR, "employee_attrition_clean_panel.csv")
df.to_csv(panel_csv, index=False)

# --- Step 3: Collapse to one record per employee ---
# Sort records so the latest snapshot comes last
df = df.sort_values(["Emp_ID", "MMM-YY", "End Date"])

# Identify employees who attrited at any point
ever_yes = df.groupby("Emp_ID")["Attrition"].apply(lambda x: (x == "Yes").any()).rename("AttritionEverYes")

# Keep only the latest row per employee
collapsed = df.groupby("Emp_ID").tail(1).copy()

# Override attrition to Yes if employee ever attrited
collapsed = collapsed.merge(ever_yes, on="Emp_ID", how="left")
collapsed["Attrition"] = np.where(collapsed["AttritionEverYes"], "Yes", collapsed["Attrition"])
collapsed.drop(columns=["AttritionEverYes"], inplace=True)

# If attrited, make End Date equal to LastWorkingDate (if available)
mask = collapsed["Attrition"].eq("Yes") & collapsed["LastWorkingDate"].notna()
collapsed.loc[mask, "End Date"] = collapsed.loc[mask, "LastWorkingDate"]

# Recompute tenure after overrides
collapsed["TenureMonths"] = collapsed.apply(
    lambda r: months_between(r["Dateofjoining"], r["End Date"]),
    axis=1
).astype("Int64")

# Summary of collapsed dataset
yes = (collapsed["Attrition"] == "Yes").sum()
no = (collapsed["Attrition"] == "No").sum()
rate = f"{yes / (yes + no):.2%}" if (yes + no) else "n/a"
print(f"[COLLAPSED] rows={len(collapsed):,}, unique Emp_IDs={collapsed['Emp_ID'].nunique():,}, "
      f"attrition={rate} (Yes={yes}, No={no})")

# Save collapsed dataset
collapsed_csv = os.path.join(OUTPUT_DIR, "employee_attrition_collapsed.csv")
collapsed.to_csv(collapsed_csv, index=False)

print("\n✅ Data processing complete.")
print(f"Saved:\n - {panel_csv}\n - {collapsed_csv}")

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_PATH = r"C:\Users\shaik\OneDrive\Desktop\Capstone\output\employee_attrition_collapsed.csv"
OUTPUT_DIR = r"C:\Users\shaik\OneDrive\Desktop\Capstone\output"
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def safe_rate_table(df, cat_col, target_col="Attrition"):
    """
    Build a summary table for a categorical column with counts and attrition rate.
    Returns a DataFrame with columns: [category, total, yes, rate].
    """
    if cat_col not in df.columns:
        return None

    tmp = df.copy()
    tmp[cat_col] = tmp[cat_col].astype("category")

    total = tmp.groupby(cat_col, dropna=False)[target_col].count()
    yes   = tmp.groupby(cat_col, dropna=False)[target_col].apply(lambda x: (x == "Yes").sum())
    out = pd.DataFrame({
        cat_col: total.index,
        "total": total.values,
        "yes":   yes.values
    })
    out["rate"] = (out["yes"] / out["total"]).round(4)
    out = out.sort_values("rate", ascending=False).reset_index(drop=True)
    return out

def save_table(df, name):
    """Save a DataFrame as CSV in the tables folder (if df is not None)."""
    if df is None:
        return
    path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(path, index=False)

def bar_from_table(tbl, cat_col, value_col, title, fname, rotation=0):
    """
    Quick bar plot from a summary table.
    Uses matplotlib only; no custom colors/styles to keep it simple and portable.
    """
    if tbl is None or tbl.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(tbl[cat_col].astype(str), tbl[value_col].values)
    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel(value_col)
    ax.tick_params(axis='x', rotation=rotation)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)

def hist_plot(series, title, xlabel, fname, bins=20):
    """Simple histogram helper."""
    s = series.dropna().values
    if s.size == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(s, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)

def rate_bar_from_table(tbl, cat_col, title, fname, rotation=0):
    """Bar plot for attrition rate (%) from a summary table produced by safe_rate_table."""
    if tbl is None or tbl.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(tbl[cat_col].astype(str), (tbl["rate"] * 100.0).values)
    ax.set_title(title)
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Attrition rate (%)")
    ax.tick_params(axis='x', rotation=rotation)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, low_memory=False)

required_cols = ["Emp_ID", "Attrition", "TenureMonths"]
for c in required_cols:
    if c not in df.columns:
        raise KeyError(f"Missing required column: {c}")

# Optional columns (EDA will skip any that are missing)
opt_cols = [
    "Age", "Gender", "Education_Level", "Salary", "Designation",
    "City", "Quarterly Rating", "Total Business Value"
]

# ---------------------------------------------------------------------
# 2) Basic sanity checks and overall stats
# ---------------------------------------------------------------------
n_rows = len(df)
n_ids  = df["Emp_ID"].nunique()
yes = (df["Attrition"] == "Yes").sum()
no  = (df["Attrition"] == "No").sum()
rate = yes / (yes + no) if (yes + no) else float("nan")

summary = pd.DataFrame({
    "metric": ["rows", "unique_emp_ids", "attrition_yes", "attrition_no", "attrition_rate"],
    "value":  [n_rows,   n_ids,            yes,            no,            round(rate, 4)]
})
save_table(summary, "00_overall_summary")

# ---------------------------------------------------------------------
# 3) Create analysis-friendly buckets
# ---------------------------------------------------------------------
# Tenure buckets (in months)
bins_tenure = [0, 12, 36, 60, 120, 10_000]
labels_tenure = ["0-12m", "13-36m", "37-60m", "61-120m", "120m+"]

df["TenureBucket"] = pd.cut(df["TenureMonths"], bins=bins_tenure, labels=labels_tenure, right=True, include_lowest=True)

# Age buckets (if Age exists)
if "Age" in df.columns:
    bins_age = [0, 25, 30, 35, 40, 45, 50, 60, 120]
    labels_age = ["<=25", "26-30", "31-35", "36-40", "41-45", "46-50", "51-60", "60+"]
    df["AgeBucket"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age, right=True, include_lowest=True)

# Salary bands (if Salary exists). Use quantiles for robust bands.
if "Salary" in df.columns:
    try:
        df["SalaryBand"] = pd.qcut(df["Salary"], q=5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
    except Exception:
        # Fallback if too many duplicates or bad data
        bins_sal = [df["Salary"].min()-1, 25000, 50000, 75000, 100000, df["Salary"].max()+1]
        labels_sal = ["<=25k", "25-50k", "50-75k", "75-100k", "100k+"]
        df["SalaryBand"] = pd.cut(df["Salary"], bins=bins_sal, labels=labels_sal, include_lowest=True)

# ---------------------------------------------------------------------
# 4) Summary tables (saved to CSV)
# ---------------------------------------------------------------------
tables = {}

tables["01_attrition_by_tenure"]   = safe_rate_table(df, "TenureBucket")
tables["02_attrition_by_gender"]   = safe_rate_table(df, "Gender")
tables["03_attrition_by_edu"]      = safe_rate_table(df, "Education_Level")
tables["04_attrition_by_city"]     = safe_rate_table(df, "City")
tables["05_attrition_by_title"]    = safe_rate_table(df, "Designation")
tables["06_attrition_by_age"]      = safe_rate_table(df, "AgeBucket") if "AgeBucket" in df.columns else None
tables["07_attrition_by_salary"]   = safe_rate_table(df, "SalaryBand") if "SalaryBand" in df.columns else None
tables["08_attrition_by_rating"]   = safe_rate_table(df, "Quarterly Rating") if "Quarterly Rating" in df.columns else None

for name, t in tables.items():
    save_table(t, name)

# ---------------------------------------------------------------------
# 5) Plots (saved to PNG)
# ---------------------------------------------------------------------
# Overall distribution charts
if "Age" in df.columns:
    hist_plot(df["Age"], "Age distribution", "Age", "age_hist.png", bins=20)

if "Salary" in df.columns:
    hist_plot(df["Salary"], "Salary distribution", "Salary", "salary_hist.png", bins=20)

hist_plot(df["TenureMonths"], "Tenure (months) distribution", "Tenure (months)", "tenure_hist.png", bins=20)

# Rate plots by category
rate_bar_from_table(tables["01_attrition_by_tenure"], "TenureBucket",
                    "Attrition rate by tenure bucket", "rate_tenure.png")

rate_bar_from_table(tables["02_attrition_by_gender"], "Gender",
                    "Attrition rate by gender", "rate_gender.png")

rate_bar_from_table(tables["03_attrition_by_edu"], "Education_Level",
                    "Attrition rate by education level", "rate_education.png", rotation=30)

rate_bar_from_table(tables["05_attrition_by_title"], "Designation",
                    "Attrition rate by designation", "rate_designation.png", rotation=45)

rate_bar_from_table(tables["06_attrition_by_age"], "AgeBucket",
                    "Attrition rate by age bucket", "rate_age.png", rotation=0) if tables["06_attrition_by_age"] is not None else None

rate_bar_from_table(tables["07_attrition_by_salary"], "SalaryBand",
                    "Attrition rate by salary band", "rate_salary.png", rotation=0) if tables["07_attrition_by_salary"] is not None else None

rate_bar_from_table(tables["08_attrition_by_rating"], "Quarterly Rating",
                    "Attrition rate by performance rating", "rate_rating.png", rotation=0) if tables["08_attrition_by_rating"] is not None else None

print("\n✅ EDA complete.")
print(f"Tables saved in: {TABLES_DIR}")
print(f"Plots  saved in: {PLOTS_DIR}")

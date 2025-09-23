import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
INPUT_PATH = r"C:\Users\shaik\OneDrive\Desktop\Capstone\output\employee_attrition_collapsed.csv"
OUTPUT_DIR = r"C:\Users\shaik\OneDrive\Desktop\Capstone\output"
PLOTS_DIR  = os.path.join(OUTPUT_DIR, "plots_model")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables_model")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Columns that leak the label (must be excluded from features)
LEAKAGE_COLS = ["End Date", "LastWorkingDate", "TenureMonths"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title, out_path):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_roc(y_true, y_prob, title, out_path):
    """ROC curve with AUC label."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_feature_importance(names, values, title, out_path, top_n=15):
    """Horizontal bar chart for feature importance values."""
    if len(values) != len(names):
        # Fall back gracefully if sizes mismatch
        idx = np.argsort(values)[::-1][:top_n]
        names = [str(i) for i in range(len(values))]
        names = np.array(names)[idx]
        values = np.array(values)[idx]
    else:
        idx = np.argsort(values)[::-1][:top_n]
        names = np.array(names)[idx]
        values = np.array(values)[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names[::-1], values[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)

# ---------------------------------------------------------------------
# 1) Load and basic preprocessing
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Target
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Feature set: drop id, target, and known leakage columns
drop_cols = ["Emp_ID", "Attrition"] + [c for c in LEAKAGE_COLS if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")

# Keep original feature names for importance plots
feature_names = X.columns.tolist()

# Encode object (categorical) columns
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Impute any missing numeric values with median
X = X.fillna(X.median(numeric_only=True))

# Scale features for Logistic Regression stability; tree models are robust either way
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratified to preserve Yes/No ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 2) Define models
# ---------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
}

# ---------------------------------------------------------------------
# 3) Train, evaluate, and save metrics/plots
# ---------------------------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    results.append({
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "roc_auc": round(auc, 4)
    })

    # Plots: confusion matrix and ROC
    plot_confusion(
        y_test, y_pred,
        title=f"{name} — Confusion Matrix",
        out_path=os.path.join(PLOTS_DIR, f"{name}_confusion.png")
    )
    plot_roc(
        y_test, y_prob,
        title=f"{name} — ROC Curve",
        out_path=os.path.join(PLOTS_DIR, f"{name}_roc.png")
    )

    # Feature importance / coefficients
    if name == "LogisticRegression":
        # Use absolute coefficients as importance proxy
        # Coefficients correspond to the scaled features order
        coef = np.abs(model.coef_.ravel())
        plot_feature_importance(
            names=np.array(feature_names),
            values=coef,
            title="Logistic Regression — |Coefficients| (Top)",
            out_path=os.path.join(PLOTS_DIR, "LogisticRegression_importance.png"),
            top_n=15
        )

        # Also save a CSV of all coefficients
        pd.DataFrame({
            "feature": feature_names,
            "abs_coef": coef
        }).sort_values("abs_coef", ascending=False).to_csv(
            os.path.join(TABLES_DIR, "logistic_coefficients.csv"), index=False
        )

    elif name == "RandomForest":
        fi = model.feature_importances_
        plot_feature_importance(
            names=np.array(feature_names),
            values=fi,
            title="Random Forest — Feature Importance (Top)",
            out_path=os.path.join(PLOTS_DIR, "RandomForest_importance.png"),
            top_n=15
        )
        pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values(
            "importance", ascending=False
        ).to_csv(os.path.join(TABLES_DIR, "rf_feature_importance.csv"), index=False)

    elif name == "XGBoost":
        fi = model.feature_importances_
        plot_feature_importance(
            names=np.array(feature_names),
            values=fi,
            title="XGBoost — Feature Importance (Top)",
            out_path=os.path.join(PLOTS_DIR, "XGBoost_importance.png"),
            top_n=15
        )
        pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values(
            "importance", ascending=False
        ).to_csv(os.path.join(TABLES_DIR, "xgb_feature_importance.csv"), index=False)

# ---------------------------------------------------------------------
# 4) Save metrics table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(TABLES_DIR, "model_results.csv"), index=False)

print("\n✅ Modeling complete.")
print(f"Results table: {os.path.join(TABLES_DIR, 'model_results.csv')}")
print(f"Plots saved to: {PLOTS_DIR}")
print(results_df)

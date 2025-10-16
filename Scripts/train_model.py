import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

# Exact file path
csv_file = r"C:\Users\shaik\OneDrive\Desktop\Capstone\Data\train_data (version 1).csv"

# Load data
print("Loading data...")
print(f"From: {csv_file}")

if not os.path.exists(csv_file):
    print(f"ERROR: File not found at {csv_file}")
    exit(1)

df = pd.read_csv(csv_file)

print(f"✓ Loaded {len(df)} employees")
print(f"✓ Columns: {df.columns.tolist()}\n")

# Feature engineering
print("Feature engineering...")

# Convert dates to datetime
if 'Dateofjoining' in df.columns:
    df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')

# Create features
if 'Designation' in df.columns and 'Joining Designation' in df.columns:
    df['PromotionCount'] = (df['Designation'] - df['Joining Designation']).clip(lower=0)

if 'Total Business Value' in df.columns and 'TenureMonths' in df.columns:
    df['ValuePerMonth'] = df['Total Business Value'] / (df['TenureMonths'] + 1)
    df['SalaryToValueRatio'] = df['Salary'] / (df['Total Business Value'] + 1)

if 'Quarterly Rating' in df.columns:
    df['HighPerformer'] = (df['Quarterly Rating'] >= 4).astype(int)

# Encode categorical
if 'Gender' in df.columns:
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])

if 'Education_Level' in df.columns:
    le_education = LabelEncoder()
    df['Education_Encoded'] = le_education.fit_transform(df['Education_Level'])

if 'City' in df.columns:
    le_city = LabelEncoder()
    df['City_Encoded'] = le_city.fit_transform(df['City'])

# Prepare target
if 'Attrition' in df.columns:
    df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
    attrition_rate = df['Attrition_Binary'].mean()*100
    print(f"✓ Attrition rate: {attrition_rate:.1f}%")
else:
    print("ERROR: 'Attrition' column not found")
    exit(1)

# Select features
feature_cols = [col for col in [
    'Age', 'Gender_Encoded', 'City_Encoded', 'Education_Encoded',
    'Salary', 'Joining Designation', 'Designation', 
    'Total Business Value', 'Quarterly Rating', 'TenureMonths',
    'PromotionCount', 'ValuePerMonth', 'SalaryToValueRatio', 'HighPerformer'
] if col in df.columns]

X = df[feature_cols].fillna(0)
y = df['Attrition_Binary']

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ X shape: {X.shape}")
print(f"✓ y shape: {y.shape}\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}\n")

# Train model
print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("Training complete!\n")

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"AUC-ROC:   {auc:.4f}\n")

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))

print("\nFeature Importance (Top 10):")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']:25s} {row['Importance']:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'random_forest_model.pkl')
joblib.dump(model, model_path)

print("\n" + "="*60)
print(f"SUCCESS: Model saved to {model_path}")
print("="*60)
print("\nNext steps:")
print("1. Restart your Streamlit app")
print("2. The red error message will be gone")
print("3. Predictions will use this trained model")
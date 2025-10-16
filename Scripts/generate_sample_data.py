import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of employees
n_employees = 20000

# Generate data
data = {
    'Emp_ID': range(1001, 1001 + n_employees),
    'Age': np.random.randint(22, 65, n_employees),
    'Gender': np.random.choice(['Male', 'Female'], n_employees),
    'City': np.random.choice(['Bangalore', 'Pune', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Kolkata', 'Ahmedabad'], n_employees),
    'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD'], n_employees, p=[0.6, 0.3, 0.1]),
    'Salary': np.random.randint(300000, 2000000, n_employees),
    'Dateofjoining': [datetime(2015, 1, 1) + timedelta(days=random.randint(0, 3650)) for _ in range(n_employees)],
    'Joining Designation': np.random.randint(1, 6, n_employees),
    'Designation': np.random.randint(1, 6, n_employees),
    'Total Business Value': np.random.randint(500000, 10000000, n_employees),
    'Quarterly Rating': np.random.randint(1, 5, n_employees),
    'TenureMonths': np.random.randint(1, 120, n_employees)
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate Attrition based on some logic (makes it realistic)
# Higher attrition for: low tenure, low rating, young age
attrition_probability = []
for idx, row in df.iterrows():
    prob = 0.05  # base probability
    
    if row['TenureMonths'] < 12:
        prob += 0.15
    if row['TenureMonths'] < 24:
        prob += 0.08
    if row['Quarterly Rating'] <= 2:
        prob += 0.12
    if row['Age'] < 30:
        prob += 0.05
    if row['Designation'] == row['Joining Designation'] and row['TenureMonths'] > 24:
        prob += 0.08
    
    # Ensure probability is between 0 and 1
    prob = min(prob, 0.95)
    attrition_probability.append(prob)

df['Attrition'] = [np.random.choice(['Yes', 'No'], p=[prob, 1-prob]) for prob in attrition_probability]

# Add derived columns (optional but realistic)
df['Salary'] = df['Salary'].astype(int)
df['Dateofjoining'] = df['Dateofjoining'].dt.strftime('%d-%m-%Y')

# Reorder columns to match template
column_order = ['Emp_ID', 'Age', 'Gender', 'City', 'Education_Level', 
                'Salary', 'Dateofjoining', 'Joining Designation', 'Designation',
                'Total Business Value', 'Quarterly Rating', 'TenureMonths', 'Attrition']

df = df[column_order]

# Save to CSV
output_file = 'sample_employee_data_20000.csv'
df.to_csv(output_file, index=False)

print(f"Sample data created: {output_file}")
print(f"Total employees: {len(df)}")
print(f"Attrition rate: {(df['Attrition'] == 'Yes').sum() / len(df) * 100:.1f}%")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData summary:")
print(df.describe())
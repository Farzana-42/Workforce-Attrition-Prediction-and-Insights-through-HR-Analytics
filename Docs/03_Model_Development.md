# Model Development & Training

## Approach

### 1. Feature Engineering

Created 14 features from raw data:
- ValuePerMonth: Business value generated per month
- TenureMonths: Time at company
- SalaryToValueRatio: Cost-to-output metric
- PromotionCount: Number of promotions
- HighPerformer: Binary flag for ratings ≥4

### 2. Model Selection

Compared three algorithms:
- Logistic Regression: 78% accuracy (baseline)
- Random Forest: 85.5% accuracy (selected)
- XGBoost: 84% accuracy

Random Forest chosen for:
- Highest accuracy
- Better recall (catches more actual leavers)
- Reasonable interpretability

### 3. Hyperparameters
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Prevent overfitting
    min_samples_split=20,  # Require 20+ samples to split
    class_weight='balanced' # Handle class imbalance (8.5% attrition)
)
```

### 4. Training Details

- Training data: 15,283 employees (80%)
- Test data: 3,821 employees (20%)
- Cross-validation: 5-fold stratified
- No data leakage: Train/test split before feature engineering

## Performance Metrics

### Test Set Results
```
Accuracy:  85.5%
Precision: 34.45%
Recall:    79.26%
F1-Score:  0.48
AUC-ROC:   0.8886
```

### Confusion Matrix

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actually No** | 3,011 (TN) | 487 (FP) |
| **Actually Yes** | 67 (FN) | 256 (TP) |

### Interpretation

- **True Positives (256)**: Correctly identified 79% of actual leavers
- **False Negatives (67)**: Missed 21% (67 people who left)
- **False Positives (487)**: Flagged as at-risk but stayed (acceptable cost)
- **True Negatives (3,011)**: Correctly identified stable employees

## Feature Importance

Top 10 features driving predictions:

1. ValuePerMonth (21.3%)
2. TotalBusinessValue (20.6%)
3. TenureMonths (16.4%)
4. SalaryToValueRatio (15.9%)
5. QuarterlyRating (11.8%)
6. Salary (3.2%)
7. Age (2.5%)
8. City (2.5%)
9. PromotionCount (1.6%)
10. HighPerformer (1.0%)
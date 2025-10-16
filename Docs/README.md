# Workforce Attrition Prediction and Insights through HR Analytics

## Overview

This project develops a machine learning-based predictive analytics system to identify employees at risk of attrition. Using ensemble methods trained on 19,104 employee records spanning 2015-2024, the system achieves 85.5% accuracy in predicting voluntary employee resignations. The solution is designed as a decision support tool for HR professionals, enabling proactive retention strategies and reducing the organizational cost of unwanted turnover.

## Problem Statement

Employee attrition represents a significant operational and financial challenge for organizations. Research indicates that replacing an employee costs between 30% and 200% of their annual salary, depending on role complexity. Most organizations discover resignations after they occur, leaving limited time for intervention. This reactive approach is inefficient compared to early identification and proactive engagement.

The core challenge lies in identifying which employees are most likely to leave before they resign. While departure risk factors exist within organizational data, they are often overlooked without systematic analysis. This project addresses that gap by building a data-driven model to surface at-risk employees for targeted retention efforts.

## Solution Architecture

The system comprises three integrated components:

**1. Predictive Model**
- Random Forest ensemble with 100 decision trees
- Trained on historical employee data with engineered features
- Optimized for high recall (79%) to minimize missed at-risk employees
- Validated against held-out test set to ensure generalization

**2. Data Processing Pipeline**
- Accepts employee data in multiple formats (CSV, Excel)
- Automated feature engineering from raw employee attributes
- Intelligent handling of missing values using statistical defaults
- Support for both standardized templates and flexible input formats

**3. Interactive Web Application**
- Built with Streamlit for rapid deployment and accessibility
- Real-time risk scoring for individual and batch predictions
- Visualization of risk patterns by department and geography
- Export functionality for action planning and reporting

## Key Results

### Model Performance

- **Accuracy**: 85.5% on held-out test set
- **Recall**: 79.26% (captures approximately 4 of 5 employees who actually leave)
- **Precision**: 34.45% (appropriate for retention context where false positives are lower cost)
- **AUC-ROC**: 0.8886 (excellent discriminative ability)
- **F1-Score**: 0.48 (balanced across precision and recall)

### Business Outcomes

On the 19,104 employee dataset:
- Identified 2,405 high-risk employees (12.6%) requiring immediate engagement
- Flagged 10,352 medium-risk employees (54.2%) for proactive monitoring
- Classified 6,347 employees (33.2%) as low-risk with stable tenure trajectories

### Feature Importance

The model relies on a hierarchy of predictive factors:

1. **Business Value Per Month** (21.3%) - Productivity relative to tenure
2. **Total Business Value** (20.6%) - Overall career contribution
3. **Tenure in Months** (16.4%) - Time with organization
4. **Salary to Value Ratio** (15.9%) - Cost-to-productivity metric
5. **Quarterly Rating** (11.8%) - Performance assessment

Additional factors: salary, age, city, promotion history, and high performer status contribute incrementally to predictions.

## Technical Implementation

### Technology Stack

- **Language**: Python 3.9+
- **ML Framework**: scikit-learn
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Model Serialization**: joblib

### Dataset Characteristics

- **Records**: 19,104 employees
- **Time Span**: 2015-2024
- **Features**: 16 original columns, 14 engineered features
- **Target Variable**: Voluntary attrition (Yes/No)
- **Class Distribution**: 8.5% attrition, 91.5% retention (imbalanced)
- **Train/Test Split**: 80/20 with stratification

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=100,           # 100 trees for stability
    max_depth=10,               # Prevent overfitting
    min_samples_split=20,       # Require sufficient samples
    min_samples_leaf=10,        # Regularization
    class_weight='balanced'     # Handle class imbalance
)
```

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Approximately 500MB disk space for dependencies and model

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Farzana-42/Workforce-Attrition-Prediction-and-Insights-through-HR-Analytics.git
cd Workforce-Attrition-Prediction-and-Insights-through-HR-Analytics
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Train the model (optional - pre-trained model included):
```bash
python Scripts/train_model.py
```

4. Launch the application:
```bash
python -m streamlit run Scripts/app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage

### Running Predictions

The application supports two input modes:

**Standard Template Mode**
- Download the provided CSV template
- Fill with employee data matching exact column specifications
- Upload for guaranteed format compatibility
- Recommended for mission-critical predictions

**Flexible Format Mode**
- Upload employee data in any CSV or Excel format
- Automatic column mapping and feature extraction
- Suitable for exploratory analysis and testing
- Note: May have slightly reduced accuracy due to format variability

### Workflow

1. Select upload method (template or flexible)
2. Upload employee dataset
3. System validates data quality and completeness
4. Review validation report and confidence metrics
5. Generate predictions
6. Analyze results through interactive visualizations
7. Download predictions and high-risk employee lists
8. Share findings with stakeholders

### Sample Data Generation

To test the system without sensitive data:
```bash
python Scripts/generate_sample_data.py
```

This creates a file with 10,000 synthetic employee records matching the data structure.

## Project Structure

```
Workforce-Attrition-Prediction-and-Insights-through-HR-Analytics/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git configuration
│
├── Data/
│   └── train_data (version 1).csv     # Training dataset
│
├── Scripts/
│   ├── app.py                         # Streamlit application
│   ├── train_model.py                 # Model training script
│   └── generate_sample_data.py        # Sample data generator
│
├── models/
│   └── random_forest_model.pkl        # Serialized trained model
│
├── Docs/
│   ├── 01_Project_Overview.md         # Business context
│   ├── 02_Data_Analysis.md            # Exploratory analysis
│   ├── 03_Model_Development.md        # Technical methodology
│   └── 04_Model_Limitations.md        # Constraints and ethics
│
└── output/
    └── (Predictions and reports generated during use)
```

## Model Insights and Findings

### Primary Attrition Drivers

Analysis of the 256 correctly identified at-risk employees revealed:

**Tenure Effect**: Employees within their first 12 months show 5x higher attrition risk compared to those with 3+ years tenure. This "honeymoon period" exit pattern is consistent across departments.

**Performance Recognition**: Employees with quarterly ratings of 2 or below show significantly elevated risk. However, notably high performers also show measurable attrition, suggesting external opportunities pull talent upward.

**Career Progression**: Stagnation in designation level beyond 24 months correlates with attrition independent of other factors. Organizations without clear advancement pathways lose employees across performance tiers.

**Business Value Alignment**: Employees with low business value relative to salary costs show higher attrition, likely due to role mismatch or limited engagement.

**Geographic Variation**: Certain cities (e.g., City 23 at 52% average risk vs. Bangalore at 6%) show structural differences warranting investigation into local management quality or cost-of-living factors.

### Limitations of the Predictive Approach

**1. Hidden Variables**

The model operates on organizational data only. It cannot account for personal circumstances including family relocation, health conditions, spousal employment changes, or external opportunities. An employee flagged as low-risk may resign due to life changes invisible in HR systems.

**2. Incomplete Satisfaction Measurement**

Quarterly ratings provide limited insight into actual job satisfaction. The model cannot detect dissatisfaction brewing quietly before resignation or the psychological state of employees experiencing burnout. True engagement requires qualitative assessment beyond quantitative metrics.

**3. Temporal Discontinuities**

The model learns from historical patterns. It cannot anticipate novel circumstances like organizational restructuring, management changes, market disruptions, or competitive hiring surges that alter departure risk.

**4. Historical Bias Propagation**

If historical hiring, promotion, or rating decisions contained bias, the model learns and reproduces that bias. For example, if certain demographic groups were historically rated lower or promoted less frequently, predictions will reflect that past inequity.

**5. Prediction Uncertainty**

With 14.5% error rate, individual predictions carry inherent uncertainty. Without comparing against actual outcomes, it's impossible to determine which specific predictions are correct. The model is probabilistic, not deterministic.

## Ethical Considerations

### Responsible Use Guidelines

This tool should function as a decision support mechanism, not an autonomous decision maker. HR professionals must combine model recommendations with contextual knowledge, manager input, and supportive conversation.

**Appropriate Applications:**
- Screening for proactive retention conversations
- Identifying cohorts for career development programs
- Prioritizing stay interviews and engagement initiatives
- Benchmarking attrition patterns across departments
- Triggering escalation to management when patterns emerge

**Prohibited Applications:**
- Sole basis for termination or demotion decisions
- Justification for reducing compensation or benefits
- Public disclosure of individual risk scores
- Automated triggering of employment actions without human review

### Risk Mitigation Strategies

**1. Avoid Self-Fulfilling Prophecy**

Employees treated as at-risk may internalize the classification and actually leave. Engagement should be supportive and development-focused rather than punitive or dismissive.

**2. Regular Bias Audits**

Monitor prediction patterns across demographic groups (age, gender, location, function). Investigate disparities to determine if model is capturing legitimate risk factors or reflecting organizational bias.

**3. Validation Tracking**

Compare predictions against actual outcomes monthly. If recall drops below 75% or precision patterns change, investigate causes before trusting new predictions.

**4. Privacy Protection**

Restrict access to individual risk scores to authorized HR personnel. Attrition predictions constitute sensitive personal information subject to data protection regulations (GDPR, CCPA, etc.).

## Model Validation and Performance Monitoring

### Confusion Matrix Interpretation

On the test set (3,821 employees):

|  | Predicted Retained | Predicted At-Risk |
|---|---|---|
| **Actually Retained** | 3,011 | 487 |
| **Actually Left** | 67 | 256 |

**True Positives (256)**: Correctly identified employees who left. Represents 79% of actual departures.

**False Negatives (67)**: Missed employees who left (21%). These represent prediction failures for individuals who departed despite favorable model inputs.

**False Positives (487)**: Flagged as at-risk but remained. In retention context, this is acceptable—cost of engagement is modest compared to replacement cost of misses.

**True Negatives (3,011)**: Correctly identified retained employees.

### Production Monitoring

Upon deployment, track these metrics monthly:

- **Recall Decay**: Monitor whether the model continues catching actual leavers. Declining recall suggests concept drift.
- **Precision Drift**: Track the proportion of flagged employees who actually depart. Changing precision indicates shifting attrition dynamics.
- **Demographic Parity**: Compare prediction accuracy across demographic groups to detect algorithmic bias.
- **Feature Distribution**: Monitor whether new data has different distributions than training data, signaling potential model drift.

### Retraining Schedule

Plan to retrain the model quarterly when:
- 3+ months of new attrition data accumulates
- Organizational changes occur (restructuring, policy changes, market shifts)
- Prediction accuracy metrics decline below 80%
- Significant data distribution changes are detected

## Future Enhancement Opportunities

**1. SHAP Explanations**

Implement SHAP (SHapley Additive exPlanations) to provide transparent, individual-level explanations of why each prediction was made. This supports HR professionals in understanding and communicating risk factors to managers and employees.

**2. Causal Inference**

Transition from correlational to causal analysis to identify interventions most likely to prevent attrition. Current model identifies at-risk employees but not optimal retention actions.

**3. Automated Recommendations**

Integrate retention action recommendations based on individual risk factors. For example, "This employee is high-risk due to stagnation—recommend promotion discussion."

**4. Real-time Integration**

Connect directly to HRIS systems (Workday, SAP SuccessFactors) for automated data pipeline and real-time scoring versus periodic batch processing.

**5. Intervention Tracking**

Build feedback mechanism to track what retention actions were taken for flagged employees and their outcomes, enabling continuous model refinement.

**6. Multi-model Ensemble**

Combine Random Forest predictions with deep learning models for certain use cases, leveraging strengths of different algorithms.

## Data and Reproducibility

All analysis code is version controlled and documented. To reproduce results:

```bash
# Train model from scratch
python Scripts/train_model.py

# Generate sample predictions
python Scripts/generate_sample_data.py

# Run model evaluation
# (Evaluation metrics printed during training)
```

The trained model is included as `models/random_forest_model.pkl` for immediate use without retraining.

## References and Resources

- Scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- SHAP for model interpretability: https://github.com/slundberg/shap
- Employee retention research: Society for Human Resource Management (SHRM)

## License

This project is intended for educational and internal organizational use.

## Author

Developed as an MS Business Analytics Capstone Project

## Support and Questions

For issues, questions, or contributions, please open an issue on the GitHub repository. For specific technical problems, include error messages and steps to reproduce.

---

**Last Updated**: October 2025  
**Model Version**: 1.0  
**Status**: Production Ready
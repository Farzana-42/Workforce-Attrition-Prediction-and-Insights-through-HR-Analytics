# Workforce Attrition Prediction — MSBA Capstone

## 📌 Project Overview
Employee attrition is a critical challenge for organizations, driving costs in recruitment, training, and lost knowledge.  
This project builds a predictive analytics framework to identify employees at risk of attrition and uncover key drivers using HR Analytics.

## 🎯 Objectives
- Develop predictive models to estimate employee attrition likelihood
- Identify key factors influencing attrition
- Provide actionable insights through visualizations
- Support HR teams with data-driven retention strategies

## 📂 Repository Structure
Capstone-Workforce-Attrition/
│
├── data/ # Raw + processed datasets
├── output/ # EDA plots, model results
├── scripts/ # Python scripts (EDA, modeling, collapsing)
├── docs/ # Documentation (Proposal, Reports)
├── README.md # Project overview
└── .gitignore

## 🛠️ Tools & Technologies
- **Python**: Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Tableau**: Interactive dashboards
- **GitHub**: Version control & portfolio presentation

## 📊 Models Implemented
- Logistic Regression
- Random Forest
- XGBoost

Performance Metrics:
| Model                | Accuracy | Precision | Recall | ROC AUC |
|----------------------|----------|-----------|--------|---------|
| Logistic Regression  | 0.951    | 0.985     | 0.942  | 0.977   |
| Random Forest        | 0.962    | 0.973     | 0.971  | 0.983   |
| XGBoost              | 0.961    | 0.965     | 0.977  | 0.984   |

## 📈 Key Insights
- Younger employees (≤25) and those in lower salary bands showed higher attrition rates
- Attrition was strongly associated with lower performance ratings
- Tenure below 1 year also showed higher turnover

## 🚀 Future Work
- Add SHAP value analysis for model explainability
- Deploy as a dashboard in Tableau / Streamlit
- Expand dataset with more real-world HRIS features

## 👩‍💻 Author
**Farzana Shaik**  
MS Business Analytics, University of Cincinnati  




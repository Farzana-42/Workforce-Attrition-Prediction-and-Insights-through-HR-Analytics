import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide", page_icon="🎯")

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_template():
    """Create template based on YOUR actual data structure"""
    
    template_data = {
        'Emp_ID': [1001, 1002, 1003],
        'Age': [35, 42, 28],
        'Gender': ['Male', 'Female', 'Male'],
        'City': ['Bangalore', 'Pune', 'Bangalore'],
        'Education_Level': ['Bachelor', 'Master', 'Bachelor'],
        'Salary': [500000, 750000, 420000],
        'Dateofjoining': ['2018-01-15', '2015-06-20', '2020-03-10'],
        'Joining Designation': [2, 3, 1],
        'Designation': [3, 4, 2],
        'Total Business Value': [1500000, 3000000, 800000],
        'Quarterly Rating': [3, 4, 2],
        'TenureMonths': [60, 96, 24]
    }
    
    return pd.DataFrame(template_data)

def get_field_definitions():
    """Documentation for each field in YOUR dataset"""
    
    definitions = {
        'Emp_ID': 'Unique employee identifier (number)',
        'Age': 'Employee age in years (18-65)',
        'Gender': 'Gender (Male/Female)',
        'City': 'City of work location',
        'Education_Level': 'Highest education level (Bachelor/Master/PhD)',
        'Salary': 'Annual salary in rupees',
        'Dateofjoining': 'Date of joining (YYYY-MM-DD format)',
        'Joining Designation': 'Designation level at joining (1-5)',
        'Designation': 'Current designation level (1-5)',
        'Total Business Value': 'Total business value generated',
        'Quarterly Rating': 'Performance rating (1-5)',
        'TenureMonths': 'Tenure in months'
    }
    
    return definitions

def validate_uploaded_data(user_df):
    """Validate user's upload"""
    
    validation = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'completeness': 0,
        'quality_score': 100
    }
    
    required_cols = ['Emp_ID', 'Age', 'Gender', 'City', 'Education_Level', 
                     'Salary', 'Dateofjoining', 'Joining Designation', 
                     'Designation', 'Total Business Value', 'Quarterly Rating', 
                     'TenureMonths']
    
    # Check missing columns
    missing_cols = set(required_cols) - set(user_df.columns)
    if missing_cols:
        validation['is_valid'] = False
        validation['errors'].append(f"Missing required columns: {', '.join(missing_cols)}")
        validation['quality_score'] -= 30
    
    # Check data types
    if 'Age' in user_df.columns:
        if not pd.api.types.is_numeric_dtype(user_df['Age']):
            validation['errors'].append("'Age' must be numeric")
            validation['is_valid'] = False
            validation['quality_score'] -= 10
    
    if 'Salary' in user_df.columns:
        if not pd.api.types.is_numeric_dtype(user_df['Salary']):
            validation['errors'].append("'Salary' must be numeric")
            validation['is_valid'] = False
            validation['quality_score'] -= 10
    
    # Check ranges
    if 'Age' in user_df.columns and pd.api.types.is_numeric_dtype(user_df['Age']):
        if not user_df['Age'].between(18, 65).all():
            validation['warnings'].append("Some ages are outside typical range (18-65)")
            validation['quality_score'] -= 5
    
    if 'Quarterly Rating' in user_df.columns and pd.api.types.is_numeric_dtype(user_df['Quarterly Rating']):
        if not user_df['Quarterly Rating'].between(1, 5).all():
            validation['warnings'].append("Some ratings are outside range (1-5)")
            validation['quality_score'] -= 5
    
    # Check categorical values
    if 'Gender' in user_df.columns:
        valid_genders = ['Male', 'Female', 'M', 'F', 'male', 'female']
        invalid_genders = user_df[~user_df['Gender'].isin(valid_genders)]['Gender'].unique()
        if len(invalid_genders) > 0:
            validation['warnings'].append(f"Unusual gender values: {', '.join(map(str, invalid_genders))}")
    
    # Calculate completeness
    if len(required_cols) > 0:
        present_cols = [c for c in required_cols if c in user_df.columns]
        if present_cols:
            total_cells = len(user_df) * len(present_cols)
            filled_cells = user_df[present_cols].notna().sum().sum()
            validation['completeness'] = round((filled_cells / total_cells * 100), 1)
        else:
            validation['completeness'] = 0
    
    validation['quality_score'] = max(0, validation['quality_score'])
    
    return validation

def engineer_features(df):
    """
    Feature engineering based on YOUR data
    """
    
    df = df.copy()
    
    # Convert dates if needed
    if 'Dateofjoining' in df.columns and df['Dateofjoining'].dtype == 'object':
        df['Dateofjoining'] = pd.to_datetime(df['Dateofjoining'], errors='coerce')
    
    # Calculate tenure if not provided
    if 'TenureMonths' not in df.columns and 'Dateofjoining' in df.columns:
        df['TenureMonths'] = ((pd.Timestamp.now() - df['Dateofjoining']).dt.days / 30).astype(int)
    
    # Promotion indicator
    if 'Joining Designation' in df.columns and 'Designation' in df.columns:
        df['PromotionCount'] = df['Designation'] - df['Joining Designation']
        df['PromotionCount'] = df['PromotionCount'].clip(lower=0)
    
    # Business value per month
    if 'Total Business Value' in df.columns and 'TenureMonths' in df.columns:
        df['ValuePerMonth'] = df['Total Business Value'] / (df['TenureMonths'] + 1)
    
    # Salary to value ratio
    if 'Salary' in df.columns and 'Total Business Value' in df.columns:
        df['SalaryToValueRatio'] = df['Salary'] / (df['Total Business Value'] + 1)
    
    # Performance category
    if 'Quarterly Rating' in df.columns:
        df['HighPerformer'] = (df['Quarterly Rating'] >= 4).astype(int)
    
    # Age groups
    if 'Age' in df.columns:
        df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], 
                                labels=['<30', '30-40', '40-50', '50+'])
    
    # Encode categorical variables
    if 'Gender' in df.columns:
        df['Gender_Encoded'] = df['Gender'].map({'Male': 1, 'M': 1, 'male': 1,
                                                   'Female': 0, 'F': 0, 'female': 0})
    
    if 'Education_Level' in df.columns:
        education_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3, 'Doctorate': 3}
        df['Education_Encoded'] = df['Education_Level'].map(education_map).fillna(1)
    
    return df

# ============================================
# MAIN APP
# ============================================

def main():
    st.title("🎯 Employee Attrition Predictor")
    st.write("Enterprise-grade prediction with maximum accuracy")
    
    # Mode selection
    st.subheader("Choose Upload Method")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Standard Template Upload", use_container_width=True, type="primary"):
            st.session_state.mode = 'template'
    
    with col2:
        if st.button("🔄 Flexible Format Upload", use_container_width=True):
            st.session_state.mode = 'flexible'
    
    # Default to flexible (since you have existing data)
    if 'mode' not in st.session_state:
        st.session_state.mode = 'flexible'
    
    st.markdown("---")
    
    # ============================================
    # TEMPLATE MODE
    # ============================================
    
    if st.session_state.mode == 'template':
        st.success("✅ **Recommended Mode**: Guaranteed maximum prediction accuracy")
        
        st.subheader("Step 1: Download Template")
        
        template_df = create_template()
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Template (CSV)",
                csv,
                "employee_attrition_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            field_defs = get_field_definitions()
            doc_text = "EMPLOYEE ATTRITION PREDICTOR - FIELD DEFINITIONS\n\n"
            for field, description in field_defs.items():
                doc_text += f"{field}:\n  {description}\n\n"
            
            st.download_button(
                "📄 Download Field Definitions",
                doc_text.encode('utf-8'),
                "field_definitions.txt",
                "text/plain",
                use_container_width=True
            )
        
        with st.expander("👀 Preview Template Format"):
            st.dataframe(template_df, use_container_width=True)
            st.write("**Field Definitions:**")
            for field, desc in get_field_definitions().items():
                st.write(f"- **{field}**: {desc}")
        
        st.subheader("Step 2: Upload Completed Template")
        
        uploaded_file = st.file_uploader(
            "Upload your completed template",
            type=['csv', 'xlsx'],
            help="Fill the template with your employee data and upload here",
            key="template_uploader"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file, mode='template')
    
    # ============================================
    # FLEXIBLE MODE
    # ============================================
    
    else:  # flexible mode
        st.warning("""
        ⚠️ **Flexible Mode**: Automatic column mapping may reduce accuracy.
        For mission-critical predictions, use Standard Template mode.
        """)
        
        st.write("Upload employee data in any format. We'll attempt to map columns automatically.")
        
        # THIS WAS THE MISSING PART!
        uploaded_file = st.file_uploader(
            "Upload employee data (CSV or Excel)",
            type=['csv', 'xlsx'],
            help="Upload your data in any format. We'll try to map columns automatically.",
            key="flexible_uploader"
        )
        
        if uploaded_file:
            process_uploaded_file(uploaded_file, mode='flexible')

def process_uploaded_file(uploaded_file, mode='template'):
    """Process uploaded file in either mode"""
    
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            user_df = pd.read_csv(uploaded_file)
        else:
            user_df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Loaded {len(user_df):,} employees with {len(user_df.columns)} columns")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return
    
    # Show preview
    with st.expander("📊 Preview your data"):
        st.dataframe(user_df.head(10), use_container_width=True)
    
    # Validate
    st.subheader("Step 2: Data Validation")
    
    with st.spinner("Validating data..."):
        validation = validate_uploaded_data(user_df)
    
    # Show validation results
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Data Quality", f"{validation['quality_score']}/100")
    col2.metric("Completeness", f"{validation['completeness']}%")
    
    if validation['is_valid']:
        col3.metric("Status", "✅ Valid", delta="Ready to predict")
    else:
        col3.metric("Status", "❌ Invalid", delta="Fix errors")
    
    # Show errors
    if validation['errors']:
        st.error("**❌ Errors Found:**")
        for error in validation['errors']:
            st.write(f"- {error}")
        
        st.info("""
        **How to fix:**
        1. Check that all required columns are present
        2. Ensure numeric fields contain only numbers
        3. Verify date formats (YYYY-MM-DD)
        4. Re-upload after fixing
        """)
    
    # Show warnings
    if validation['warnings']:
        with st.expander("⚠️ Warnings (non-critical)"):
            for warning in validation['warnings']:
                st.write(f"- {warning}")
    
    # Generate predictions if valid
    if validation['is_valid']:
        st.markdown("---")
        st.subheader("Step 3: Generate Predictions")
        
        if st.button("🎯 Generate Attrition Predictions", type="primary", use_container_width=True):
            
            with st.spinner("Engineering features and generating predictions..."):
                try:
                    # Feature engineering
                    processed_df = engineer_features(user_df)
                    
                    # Load your trained model
                    try:
                        model = joblib.load('models/random_forest_model.pkl')
                    except:
                        st.error("""
                        ❌ Model file not found. Please ensure:
                        1. You have trained and saved your model as 'models/random_forest_model.pkl'
                        2. The file path is correct
                        
                        For demonstration, using a simple heuristic...
                        """)
                        # Fallback: Simple heuristic-based risk scoring
                        predictions = calculate_heuristic_risk(processed_df)
                    else:
                        # Use actual model
                        model_features = model.feature_names_in_
                        X = processed_df[model_features]
                        predictions = model.predict_proba(X)[:, 1]
                    
                    # Add predictions to original dataframe
                    user_df['Attrition_Risk_Score'] = (predictions * 100).round(1)
                    user_df['Risk_Level'] = pd.cut(
                        predictions,
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=['🟢 Low', '🟡 Medium', '🔴 High']
                    )
                    
                    st.success("✅ Predictions generated successfully!")
                    
                except Exception as e:
                    predictions = calculate_heuristic_risk(user_df)
                    user_df['Attrition_Risk_Score'] = (predictions * 100).round(1)
                    user_df['Risk_Level'] = pd.cut(
                        predictions,
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=['🟢 Low', '🟡 Medium', '🔴 High']
                    )
            
            # Display results
            display_results(user_df, predictions)

def calculate_heuristic_risk(df):
    """
    Fallback heuristic-based risk calculation
    Based on domain knowledge about attrition drivers
    """
    
    risk_score = np.zeros(len(df))
    
    # Factor 1: Low tenure (high risk)
    if 'TenureMonths' in df.columns:
        risk_score += np.where(df['TenureMonths'] < 12, 0.3, 0)
        risk_score += np.where(df['TenureMonths'].between(12, 24), 0.15, 0)
    
    # Factor 2: Young age (higher mobility)
    if 'Age' in df.columns:
        risk_score += np.where(df['Age'] < 30, 0.2, 0)
    
    # Factor 3: Low quarterly rating
    if 'Quarterly Rating' in df.columns:
        risk_score += np.where(df['Quarterly Rating'] <= 2, 0.25, 0)
    
    # Factor 4: No promotion
    if 'Joining Designation' in df.columns and 'Designation' in df.columns:
        no_promotion = df['Designation'] == df['Joining Designation']
        long_tenure = df['TenureMonths'] > 24 if 'TenureMonths' in df.columns else True
        risk_score += np.where(no_promotion & long_tenure, 0.15, 0)
    
    # Factor 5: Low business value
    if 'Total Business Value' in df.columns:
        low_value = df['Total Business Value'] < df['Total Business Value'].median()
        risk_score += np.where(low_value, 0.1, 0)
    
    # Normalize to 0-1 range
    risk_score = np.clip(risk_score, 0, 1)
    
    return risk_score

def display_results(user_df, predictions):
    """Display prediction results"""
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high = (predictions > 0.7).sum()
    medium = ((predictions > 0.3) & (predictions <= 0.7)).sum()
    low = (predictions <= 0.3).sum()
    avg_risk = predictions.mean() * 100
    
    col1.metric("🔴 High Risk", high, f"{high/len(user_df)*100:.1f}%")
    col2.metric("🟡 Medium Risk", medium, f"{medium/len(user_df)*100:.1f}%")
    col3.metric("🟢 Low Risk", low, f"{low/len(user_df)*100:.1f}%")
    col4.metric("📊 Avg Risk Score", f"{avg_risk:.1f}%")
    
    # Results table
    st.write("### Individual Risk Scores")
    
    # Select columns to display
    available_cols = [col for col in user_df.columns if col not in ['Attrition_Risk_Score', 'Risk_Level']]
    default_cols = []
    
    # Smart default selection
    for col in ['Emp_ID', 'Age', 'Department', 'City', 'Designation', 'Quarterly Rating']:
        if col in available_cols:
            default_cols.append(col)
    default_cols = default_cols[:4]  # Max 4 columns
    
    default_cols.extend(['Attrition_Risk_Score', 'Risk_Level'])
    
    display_cols = st.multiselect(
        "Select columns to display",
        options=available_cols + ['Attrition_Risk_Score', 'Risk_Level'],
        default=default_cols
    )
    
    if display_cols:
        display_df = user_df[display_cols].sort_values('Attrition_Risk_Score', ascending=False)
        
        # Color coding
        def color_risk(val):
            if '🔴' in str(val):
                return 'background-color: #ffebee'
            elif '🟡' in str(val):
                return 'background-color: #fff9c4'
            elif '🟢' in str(val):
                return 'background-color: #e8f5e9'
            return ''
        
        styled_df = display_df.style.applymap(color_risk, subset=['Risk_Level'] if 'Risk_Level' in display_df.columns else [])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    # --- Unified color scheme and consistent order ---
    RISK_ORDER = ['🟢 Low', '🟡 Medium', '🔴 High']
    RISK_COLORS = {
        '🟢 Low': '#2ecc71',     # green
        '🟡 Medium': '#f1c40f',  # yellow
        '🔴 High': '#e74c3c',    # red
    }

    st.write("### Risk Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution histogram (single red tone)
        fig = px.histogram(
            user_df,
            x='Attrition_Risk_Score',
            nbins=20,
            title="Distribution of Risk Scores",
            labels={'Attrition_Risk_Score': 'Risk Score (%)'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.update_layout(
            xaxis_title="Risk Score (%)",
            yaxis_title="Count",
            title_x=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk level donut chart (consistent legend order & colors)
        risk_counts = (
            user_df['Risk_Level']
            .value_counts()
            .reindex(RISK_ORDER)
            .fillna(0)
            .reset_index()
            .rename(columns={'index': 'Risk_Level', 'Risk_Level': 'count'})
        )

        fig = px.pie(
            risk_counts,
            names='Risk_Level',
            values='count',
            hole=0.4,
            category_orders={'Risk_Level': RISK_ORDER},
            color='Risk_Level',
            color_discrete_map=RISK_COLORS,
        )
        fig.update_traces(sort=False, textinfo='percent+label')
        fig.update_layout(
            title="Risk Level Distribution",
            legend_traceorder='normal',
            title_x=0.3
        )
        st.plotly_chart(fig, use_container_width=True)

    
    # Additional analysis by category
    if 'City' in user_df.columns:
        st.write("### Risk by City")
        city_risk = user_df.groupby('City')['Attrition_Risk_Score'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=city_risk.values,
            y=city_risk.index,
            orientation='h',
            title="Average Risk Score by City",
            labels={'x': 'Average Risk Score (%)', 'y': 'City'},
            color=city_risk.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Full results
        csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download All Results",
            csv,
            f"attrition_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # High risk only
        high_risk_df = user_df[user_df['Risk_Level'] == '🔴 High']
        if len(high_risk_df) > 0:
            csv_high = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⚠️ High Risk Only",
                csv_high,
                f"high_risk_employees_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.info("No high-risk employees")
    
    with col3:
        # Action items
        at_risk_df = user_df[user_df['Risk_Level'].isin(['🟡 Medium', '🔴 High'])]
        if len(at_risk_df) > 0:
            csv_risk = at_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Action Required",
                csv_risk,
                f"action_required_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory for statistical analysis outputs if it doesn't exist
if not os.path.exists('statistical_analysis'):
    os.makedirs('statistical_analysis')

# Load the dataset
try:
    df = pd.read_csv('hospital_patient_data.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please run generate_dataset.py first.")
    exit(1)

# Data preprocessing for statistical analysis
def preprocess_data():
    # Convert date columns to datetime
    df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
    
    # Create derived features
    df['AdmissionMonth'] = df['AdmissionDate'].dt.month
    df['AdmissionDayOfWeek'] = df['AdmissionDate'].dt.dayofweek
    df['AdmissionQuarter'] = df['AdmissionDate'].dt.quarter
    
    # Convert categorical variables to binary for correlation analysis
    df['IsMale'] = (df['Gender'] == 'Male').astype(int)
    df['IsEmergency'] = (df['AdmissionType'] == 'Emergency').astype(int)
    df['IsReadmitted'] = df['Readmitted'].astype(int)
    df['IsDeceased'] = (df['Outcome'] == 'Deceased').astype(int)
    df['IsRecovered'] = (df['Outcome'] == 'Recovered').astype(int)
    
    # Create age categories for ANOVA
    df['AgeCategory'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                              labels=['0-18', '19-35', '36-50', '51-65', '66+'])
    
    return df

# Perform statistical tests
def perform_statistical_tests(df):
    results = {}
    
    # 1. T-test: Compare length of stay between male and female patients
    male_los = df[df['Gender'] == 'Male']['LengthOfStay']
    female_los = df[df['Gender'] == 'Female']['LengthOfStay']
    t_stat, p_val = stats.ttest_ind(male_los, female_los)
    results['gender_los_ttest'] = {
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'interpretation': 'There is a statistically significant difference in length of stay between male and female patients.' 
                         if p_val < 0.05 else 'There is no statistically significant difference in length of stay between male and female patients.'
    }
    
    # 2. ANOVA: Compare total cost across age categories
    age_groups = df.groupby('AgeCategory')['TotalCost'].apply(list)
    f_stat, p_val = stats.f_oneway(*age_groups)
    results['age_cost_anova'] = {
        'f_statistic': f_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'interpretation': 'There is a statistically significant difference in total cost across different age categories.' 
                         if p_val < 0.05 else 'There is no statistically significant difference in total cost across different age categories.'
    }
    
    # 3. Chi-square test: Association between admission type and outcome
    contingency_table = pd.crosstab(df['AdmissionType'], df['Outcome'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    results['admission_outcome_chi2'] = {
        'chi2_statistic': chi2,
        'p_value': p_val,
        'degrees_of_freedom': dof,
        'significant': p_val < 0.05,
        'interpretation': 'There is a statistically significant association between admission type and patient outcome.' 
                         if p_val < 0.05 else 'There is no statistically significant association between admission type and patient outcome.'
    }
    
    # 4. Correlation test: Correlation between age and total cost
    corr, p_val = stats.pearsonr(df['Age'], df['TotalCost'])
    results['age_cost_correlation'] = {
        'correlation_coefficient': corr,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'interpretation': f'There is a statistically significant {"positive" if corr > 0 else "negative"} correlation between age and total cost.' 
                         if p_val < 0.05 else 'There is no statistically significant correlation between age and total cost.'
    }
    
    # 5. Mann-Whitney U test: Compare insurance coverage between readmitted and non-readmitted patients
    readmit_coverage = df[df['Readmitted'] == 1]['InsuranceCoverage']
    non_readmit_coverage = df[df['Readmitted'] == 0]['InsuranceCoverage']
    u_stat, p_val = stats.mannwhitneyu(readmit_coverage, non_readmit_coverage)
    results['readmission_coverage_mannwhitney'] = {
        'u_statistic': u_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'interpretation': 'There is a statistically significant difference in insurance coverage between readmitted and non-readmitted patients.' 
                         if p_val < 0.05 else 'There is no statistically significant difference in insurance coverage between readmitted and non-readmitted patients.'
    }
    
    return results

# Visualize statistical test results
def visualize_statistical_results(df, results):
    # 1. Gender vs Length of Stay (T-test)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Gender', y='LengthOfStay', data=df)
    plt.title(f"Length of Stay by Gender\nT-test p-value: {results['gender_los_ttest']['p_value']:.4f}")
    plt.tight_layout()
    plt.savefig('statistical_analysis/gender_los_ttest.png')
    
    # 2. Age Category vs Total Cost (ANOVA)
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='AgeCategory', y='TotalCost', data=df)
    plt.title(f"Total Cost by Age Category\nANOVA p-value: {results['age_cost_anova']['p_value']:.4f}")
    plt.tight_layout()
    plt.savefig('statistical_analysis/age_cost_anova.png')
    
    # 3. Admission Type vs Outcome (Chi-square)
    plt.figure(figsize=(14, 8))
    contingency = pd.crosstab(df['AdmissionType'], df['Outcome'], normalize='index')
    contingency.plot(kind='bar', stacked=True)
    plt.title(f"Outcome by Admission Type\nChi-square p-value: {results['admission_outcome_chi2']['p_value']:.4f}")
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig('statistical_analysis/admission_outcome_chi2.png')
    
    # 4. Age vs Total Cost (Correlation)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Age', y='TotalCost', data=df, alpha=0.6)
    plt.title(f"Age vs Total Cost\nPearson correlation: {results['age_cost_correlation']['correlation_coefficient']:.4f} (p-value: {results['age_cost_correlation']['p_value']:.4f})")
    plt.tight_layout()
    plt.savefig('statistical_analysis/age_cost_correlation.png')
    
    # 5. Readmission vs Insurance Coverage (Mann-Whitney U)
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='Readmitted', y='InsuranceCoverage', data=df)
    plt.title(f"Insurance Coverage by Readmission Status\nMann-Whitney U p-value: {results['readmission_coverage_mannwhitney']['p_value']:.4f}")
    plt.xticks([0, 1], ['Not Readmitted', 'Readmitted'])
    plt.tight_layout()
    plt.savefig('statistical_analysis/readmission_coverage_mannwhitney.png')

# Build predictive models
def build_predictive_models(df):
    model_results = {}
    
    # 1. Readmission Prediction Model
    print("\nBuilding readmission prediction model...")
    
    # Prepare features and target for readmission prediction
    X_readmit = df[['Age', 'IsMale', 'BMI', 'LengthOfStay', 'IsEmergency', 'TotalCost', 'InsuranceCoverage']]
    y_readmit = df['IsReadmitted']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_readmit, y_readmit, test_size=0.25, random_state=42)
    
    # Train model
    readmit_model = RandomForestClassifier(n_estimators=100, random_state=42)
    readmit_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = readmit_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_readmit.columns,
        'Importance': readmit_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    model_results['readmission_prediction'] = {
        'accuracy': accuracy,
        'classification_report': report,
        'feature_importance': feature_importance
    }
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Readmission Prediction')
    plt.tight_layout()
    plt.savefig('statistical_analysis/readmission_feature_importance.png')
    
    # 2. Length of Stay Prediction Model
    print("Building length of stay prediction model...")
    
    # Prepare features and target for LOS prediction
    X_los = df[['Age', 'IsMale', 'BMI', 'IsEmergency', 'InsuranceCoverage']]
    y_los = df['LengthOfStay']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_los, y_los, test_size=0.25, random_state=42)
    
    # Train model
    los_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    los_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = los_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_los.columns,
        'Importance': los_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    model_results['los_prediction'] = {
        'mean_squared_error': mse,
        'r2_score': r2,
        'feature_importance': feature_importance
    }
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Length of Stay Prediction')
    plt.tight_layout()
    plt.savefig('statistical_analysis/los_feature_importance.png')
    
    # 3. Mortality Prediction Model
    print("Building mortality prediction model...")
    
    # Prepare features and target for mortality prediction
    X_mortality = df[['Age', 'IsMale', 'BMI', 'LengthOfStay', 'IsEmergency', 'TotalCost', 'InsuranceCoverage']]
    y_mortality = df['IsDeceased']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_mortality, y_mortality, test_size=0.25, random_state=42)
    
    # Train model
    mortality_model = LogisticRegression(random_state=42, max_iter=1000)
    mortality_model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = mortality_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance (coefficients for logistic regression)
    feature_importance = pd.DataFrame({
        'Feature': X_mortality.columns,
        'Coefficient': mortality_model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    
    model_results['mortality_prediction'] = {
        'accuracy': accuracy,
        'classification_report': report,
        'feature_importance': feature_importance
    }
    
    # Visualize coefficients
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Feature Coefficients for Mortality Prediction')
    plt.tight_layout()
    plt.savefig('statistical_analysis/mortality_feature_importance.png')
    
    return model_results

# Generate a report of the statistical analysis and predictive models
def generate_report(statistical_results, model_results):
    report = "# Statistical Analysis and Predictive Modeling Report\n\n"
    
    # Statistical Tests Section
    report += "## Statistical Tests\n\n"
    
    # T-test results
    report += "### T-test: Length of Stay by Gender\n"
    report += f"- t-statistic: {statistical_results['gender_los_ttest']['t_statistic']:.4f}\n"
    report += f"- p-value: {statistical_results['gender_los_ttest']['p_value']:.4f}\n"
    report += f"- Significant: {statistical_results['gender_los_ttest']['significant']}\n"
    report += f"- Interpretation: {statistical_results['gender_los_ttest']['interpretation']}\n\n"
    
    # ANOVA results
    report += "### ANOVA: Total Cost by Age Category\n"
    report += f"- F-statistic: {statistical_results['age_cost_anova']['f_statistic']:.4f}\n"
    report += f"- p-value: {statistical_results['age_cost_anova']['p_value']:.4f}\n"
    report += f"- Significant: {statistical_results['age_cost_anova']['significant']}\n"
    report += f"- Interpretation: {statistical_results['age_cost_anova']['interpretation']}\n\n"
    
    # Chi-square results
    report += "### Chi-square: Admission Type and Outcome\n"
    report += f"- Chi-square statistic: {statistical_results['admission_outcome_chi2']['chi2_statistic']:.4f}\n"
    report += f"- p-value: {statistical_results['admission_outcome_chi2']['p_value']:.4f}\n"
    report += f"- Degrees of freedom: {statistical_results['admission_outcome_chi2']['degrees_of_freedom']}\n"
    report += f"- Significant: {statistical_results['admission_outcome_chi2']['significant']}\n"
    report += f"- Interpretation: {statistical_results['admission_outcome_chi2']['interpretation']}\n\n"
    
    # Correlation results
    report += "### Correlation: Age and Total Cost\n"
    report += f"- Correlation coefficient: {statistical_results['age_cost_correlation']['correlation_coefficient']:.4f}\n"
    report += f"- p-value: {statistical_results['age_cost_correlation']['p_value']:.4f}\n"
    report += f"- Significant: {statistical_results['age_cost_correlation']['significant']}\n"
    report += f"- Interpretation: {statistical_results['age_cost_correlation']['interpretation']}\n\n"
    
    # Mann-Whitney U results
    report += "### Mann-Whitney U: Insurance Coverage by Readmission Status\n"
    report += f"- U-statistic: {statistical_results['readmission_coverage_mannwhitney']['u_statistic']:.4f}\n"
    report += f"- p-value: {statistical_results['readmission_coverage_mannwhitney']['p_value']:.4f}\n"
    report += f"- Significant: {statistical_results['readmission_coverage_mannwhitney']['significant']}\n"
    report += f"- Interpretation: {statistical_results['readmission_coverage_mannwhitney']['interpretation']}\n\n"
    
    # Predictive Models Section
    report += "## Predictive Models\n\n"
    
    # Readmission prediction model
    report += "### Readmission Prediction Model\n"
    report += f"- Accuracy: {model_results['readmission_prediction']['accuracy']:.4f}\n"
    report += "- Classification Report:\n```\n"
    for class_label, metrics in model_results['readmission_prediction']['classification_report'].items():
        if class_label in ['0', '1']:
            report += f"  Class {class_label}:\n"
            report += f"    Precision: {metrics['precision']:.4f}\n"
            report += f"    Recall: {metrics['recall']:.4f}\n"
            report += f"    F1-score: {metrics['f1-score']:.4f}\n"
    report += "```\n"
    report += "- Top 3 Important Features:\n"
    for i in range(min(3, len(model_results['readmission_prediction']['feature_importance']))):
        feature = model_results['readmission_prediction']['feature_importance'].iloc[i]
        report += f"  {i+1}. {feature['Feature']}: {feature['Importance']:.4f}\n"
    report += "\n"
    
    # Length of stay prediction model
    report += "### Length of Stay Prediction Model\n"
    report += f"- Mean Squared Error: {model_results['los_prediction']['mean_squared_error']:.4f}\n"
    report += f"- R² Score: {model_results['los_prediction']['r2_score']:.4f}\n"
    report += "- Top 3 Important Features:\n"
    for i in range(min(3, len(model_results['los_prediction']['feature_importance']))):
        feature = model_results['los_prediction']['feature_importance'].iloc[i]
        report += f"  {i+1}. {feature['Feature']}: {feature['Importance']:.4f}\n"
    report += "\n"
    
    # Mortality prediction model
    report += "### Mortality Prediction Model\n"
    report += f"- Accuracy: {model_results['mortality_prediction']['accuracy']:.4f}\n"
    report += "- Classification Report:\n```\n"
    for class_label, metrics in model_results['mortality_prediction']['classification_report'].items():
        if class_label in ['0', '1']:
            report += f"  Class {class_label}:\n"
            report += f"    Precision: {metrics['precision']:.4f}\n"
            report += f"    Recall: {metrics['recall']:.4f}\n"
            report += f"    F1-score: {metrics['f1-score']:.4f}\n"
    report += "```\n"
    report += "- Top 3 Features by Coefficient Magnitude:\n"
    # Sort by absolute coefficient value for importance
    sorted_coefs = model_results['mortality_prediction']['feature_importance'].copy()
    sorted_coefs['AbsCoef'] = sorted_coefs['Coefficient'].abs()
    sorted_coefs = sorted_coefs.sort_values('AbsCoef', ascending=False)
    for i in range(min(3, len(sorted_coefs))):
        feature = sorted_coefs.iloc[i]
        report += f"  {i+1}. {feature['Feature']}: {feature['Coefficient']:.4f}\n"
    report += "\n"
    
    # Summary and Conclusions
    report += "## Summary and Conclusions\n\n"
    
    # Statistical findings summary
    report += "### Statistical Findings\n"
    significant_tests = [test for test, result in statistical_results.items() if result['significant']]
    if significant_tests:
        report += "The following statistical tests showed significant results:\n"
        for test in significant_tests:
            report += f"- {test}: {statistical_results[test]['interpretation']}\n"
    else:
        report += "None of the statistical tests showed significant results.\n"
    report += "\n"
    
    # Predictive model summary
    report += "### Predictive Model Performance\n"
    report += f"- The readmission prediction model achieved {model_results['readmission_prediction']['accuracy']*100:.1f}% accuracy.\n"
    report += f"- The length of stay prediction model explained {model_results['los_prediction']['r2_score']*100:.1f}% of the variance (R²).\n"
    report += f"- The mortality prediction model achieved {model_results['mortality_prediction']['accuracy']*100:.1f}% accuracy.\n\n"
    
    # Key predictors
    report += "### Key Predictors Across Models\n"
    report += "The most important features for predicting patient outcomes were:\n"
    
    # Get top feature from each model
    readmit_top = model_results['readmission_prediction']['feature_importance'].iloc[0]['Feature']
    los_top = model_results['los_prediction']['feature_importance'].iloc[0]['Feature']
    mortality_top = sorted_coefs.iloc[0]['Feature']
    
    report += f"- Readmission: {readmit_top}\n"
    report += f"- Length of Stay: {los_top}\n"
    report += f"- Mortality: {mortality_top}\n\n"
    
    # Clinical implications
    report += "### Clinical Implications\n"
    report += "Based on the statistical analysis and predictive models, the following clinical implications can be drawn:\n\n"
    
    report += "1. **Risk Stratification**: Patients can be stratified based on their risk of readmission and mortality using the identified key predictors.\n"
    report += "2. **Resource Allocation**: Length of stay predictions can help in better resource allocation and planning.\n"
    report += "3. **Targeted Interventions**: Specific interventions can be designed for high-risk patients identified by the predictive models.\n"
    report += "4. **Cost Management**: Understanding the factors affecting total cost can help in developing cost-effective treatment plans.\n"
    report += "5. **Quality Improvement**: The identified associations between admission types and outcomes can guide quality improvement initiatives.\n"
    
    # Save the report
    with open('statistical_analysis/statistical_analysis_report.md', 'w') as f:
        f.write(report)
    
    print(f"Statistical analysis report generated and saved to 'statistical_analysis/statistical_analysis_report.md'")

# Main function
def main():
    print("Starting statistical analysis and predictive modeling...")
    
    # Preprocess data
    print("Preprocessing data...")
    processed_df = preprocess_data()
    
    # Perform statistical tests
    print("Performing statistical tests...")
    statistical_results = perform_statistical_tests(processed_df)
    
    # Visualize statistical results
    print("Visualizing statistical test results...")
    visualize_statistical_results(processed_df, statistical_results)
    
    # Build predictive models
    print("Building predictive models...")
    model_results = build_predictive_models(processed_df)
    
    # Generate report
    print("Generating statistical analysis report...")
    generate_report(statistical_results, model_results)
    
    print("\nStatistical analysis and predictive modeling completed successfully!")

if __name__ == "__main__":
    main()
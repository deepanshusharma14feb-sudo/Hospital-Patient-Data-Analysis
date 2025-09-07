import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Loading hospital patient dataset...")

# Load the dataset
try:
    df = pd.read_csv('hospital_patient_data.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please run generate_dataset.py first.")
    exit(1)

# Create a directory for EDA outputs if it doesn't exist
import os
if not os.path.exists('eda_outputs'):
    os.makedirs('eda_outputs')

# Basic data exploration
print("\n=== DATASET OVERVIEW ===")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary statistics:")
print(df.describe(include='all').T)

# Check for missing values
print("\n=== MISSING VALUES ANALYSIS ===")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_df[missing_df['Missing Values'] > 0])

# If there are missing values, we would handle them here
# For this synthetic dataset, we don't expect missing values

# Data type conversion and feature engineering
print("\n=== DATA PREPROCESSING ===")

# Convert date columns to datetime
df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])

# Verify length of stay calculation
df['CalculatedLOS'] = (df['DischargeDate'] - df['AdmissionDate']).dt.days
los_match = (df['LengthOfStay'] == df['CalculatedLOS']).all()
print(f"Length of stay values match calculated values: {los_match}")

# Create month and year features for time analysis
df['AdmissionMonth'] = df['AdmissionDate'].dt.month
df['AdmissionYear'] = df['AdmissionDate'].dt.year
df['AdmissionDayOfWeek'] = df['AdmissionDate'].dt.dayofweek

# Create age groups for demographic analysis
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[0, 18, 35, 50, 65, 80, 100], 
                        labels=['0-18', '19-35', '36-50', '51-65', '66-80', '81+'])

# Create BMI categories
df['BMICategory'] = pd.cut(df['BMI'],
                           bins=[0, 18.5, 25, 30, 35, 100],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])

# Create cost categories
df['CostCategory'] = pd.qcut(df['TotalCost'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

print("\nFeature engineering complete. New columns added:")
print("- CalculatedLOS: Calculated length of stay")
print("- AdmissionMonth: Month of admission")
print("- AdmissionYear: Year of admission")
print("- AdmissionDayOfWeek: Day of week of admission (0=Monday, 6=Sunday)")
print("- AgeGroup: Age categorized into groups")
print("- BMICategory: BMI categorized according to standard ranges")
print("- CostCategory: Total cost divided into quintiles")

# Univariate Analysis
print("\n=== UNIVARIATE ANALYSIS ===")

# Categorical variables analysis
print("\nCategorical Variables Distribution:")
categorical_cols = ['Gender', 'BloodType', 'AdmissionType', 'PrimaryDiagnosis', 
                   'Treatment', 'InsuranceType', 'Outcome', 'Readmitted',
                   'AgeGroup', 'BMICategory', 'CostCategory']

for col in categorical_cols:
    print(f"\n{col} Distribution:")
    value_counts = df[col].value_counts()
    print(value_counts)
    print(f"Percentage:\n{value_counts / len(df) * 100}")
    
    # Create and save plots for categorical variables
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(y=col, data=df, order=value_counts.index)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(f'eda_outputs/{col}_distribution.png')
    plt.close()

# Numerical variables analysis
print("\nNumerical Variables Distribution:")
numerical_cols = ['Age', 'BMI', 'LengthOfStay', 'TotalCost', 'InsuranceCoverage', 'PatientPayment']

for col in numerical_cols:
    print(f"\n{col} Statistics:")
    print(df[col].describe())
    
    # Create and save histograms for numerical variables
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.savefig(f'eda_outputs/{col}_histogram.png')
    plt.close()
    
    # Create and save boxplots for numerical variables
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(f'eda_outputs/{col}_boxplot.png')
    plt.close()

# Bivariate Analysis
print("\n=== BIVARIATE ANALYSIS ===")

# Correlation matrix for numerical variables
print("\nCorrelation Matrix for Numerical Variables:")
corr_matrix = df[numerical_cols].corr()
print(corr_matrix)

# Create and save correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig('eda_outputs/correlation_heatmap.png')
plt.close()

# Age vs Length of Stay
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='LengthOfStay', data=df, alpha=0.6)
plt.title('Age vs Length of Stay')
plt.savefig('eda_outputs/age_vs_los_scatter.png')
plt.close()

# Age vs Total Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='TotalCost', data=df, alpha=0.6)
plt.title('Age vs Total Cost')
plt.savefig('eda_outputs/age_vs_cost_scatter.png')
plt.close()

# Length of Stay vs Total Cost
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LengthOfStay', y='TotalCost', data=df, alpha=0.6)
plt.title('Length of Stay vs Total Cost')
plt.savefig('eda_outputs/los_vs_cost_scatter.png')
plt.close()

# Categorical vs Numerical Analysis
print("\nCategorical vs Numerical Variables Analysis:")

# Gender vs Age
plt.figure(figsize=(10, 6))
sns.boxplot(x='Gender', y='Age', data=df)
plt.title('Age Distribution by Gender')
plt.savefig('eda_outputs/gender_vs_age_boxplot.png')
plt.close()

# Admission Type vs Length of Stay
plt.figure(figsize=(12, 6))
sns.boxplot(x='AdmissionType', y='LengthOfStay', data=df)
plt.title('Length of Stay by Admission Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/admission_type_vs_los_boxplot.png')
plt.close()

# Treatment vs Total Cost
plt.figure(figsize=(14, 8))
sns.boxplot(x='Treatment', y='TotalCost', data=df)
plt.title('Total Cost by Treatment Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/treatment_vs_cost_boxplot.png')
plt.close()

# Insurance Type vs Patient Payment
plt.figure(figsize=(12, 6))
sns.boxplot(x='InsuranceType', y='PatientPayment', data=df)
plt.title('Patient Payment by Insurance Type')
plt.tight_layout()
plt.savefig('eda_outputs/insurance_vs_payment_boxplot.png')
plt.close()

# Outcome vs Length of Stay
plt.figure(figsize=(12, 6))
sns.boxplot(x='Outcome', y='LengthOfStay', data=df)
plt.title('Length of Stay by Outcome')
plt.tight_layout()
plt.savefig('eda_outputs/outcome_vs_los_boxplot.png')
plt.close()

# Age Group vs Readmission
plt.figure(figsize=(12, 6))
sns.countplot(x='AgeGroup', hue='Readmitted', data=df)
plt.title('Readmission by Age Group')
plt.tight_layout()
plt.savefig('eda_outputs/age_group_vs_readmission_count.png')
plt.close()

# Primary Diagnosis Analysis
print("\n=== PRIMARY DIAGNOSIS ANALYSIS ===")

# Top 10 diagnoses
top_diagnoses = df['PrimaryDiagnosis'].value_counts().head(10)
print("\nTop 10 Primary Diagnoses:")
print(top_diagnoses)

# Create and save plot for top diagnoses
plt.figure(figsize=(12, 8))
sns.barplot(y=top_diagnoses.index, x=top_diagnoses.values)
plt.title('Top 10 Primary Diagnoses')
plt.tight_layout()
plt.savefig('eda_outputs/top_diagnoses_barplot.png')
plt.close()

# Average length of stay by diagnosis
diagnosis_los = df.groupby('PrimaryDiagnosis')['LengthOfStay'].agg(['mean', 'median', 'count']).sort_values('count', ascending=False).head(10)
print("\nAverage Length of Stay for Top 10 Diagnoses:")
print(diagnosis_los)

# Create and save plot for length of stay by diagnosis
plt.figure(figsize=(12, 8))
sns.barplot(y=diagnosis_los.index, x=diagnosis_los['mean'])
plt.title('Average Length of Stay by Primary Diagnosis (Top 10)')
plt.tight_layout()
plt.savefig('eda_outputs/diagnosis_vs_los_barplot.png')
plt.close()

# Average cost by diagnosis
diagnosis_cost = df.groupby('PrimaryDiagnosis')['TotalCost'].agg(['mean', 'median', 'count']).sort_values('count', ascending=False).head(10)
print("\nAverage Cost for Top 10 Diagnoses:")
print(diagnosis_cost)

# Create and save plot for cost by diagnosis
plt.figure(figsize=(12, 8))
sns.barplot(y=diagnosis_cost.index, x=diagnosis_cost['mean'])
plt.title('Average Cost by Primary Diagnosis (Top 10)')
plt.tight_layout()
plt.savefig('eda_outputs/diagnosis_vs_cost_barplot.png')
plt.close()

# Outcome Analysis
print("\n=== OUTCOME ANALYSIS ===")

# Outcome distribution
outcome_counts = df['Outcome'].value_counts()
print("\nOutcome Distribution:")
print(outcome_counts)
print(f"Percentage:\n{outcome_counts / len(df) * 100}")

# Create and save plot for outcome distribution
plt.figure(figsize=(10, 6))
sns.countplot(y='Outcome', data=df, order=outcome_counts.index)
plt.title('Distribution of Patient Outcomes')
plt.tight_layout()
plt.savefig('eda_outputs/outcome_distribution.png')
plt.close()

# Outcome by age group
outcome_by_age = pd.crosstab(df['AgeGroup'], df['Outcome'], normalize='index') * 100
print("\nOutcome Percentage by Age Group:")
print(outcome_by_age)

# Create and save plot for outcome by age group
plt.figure(figsize=(12, 8))
sns.heatmap(outcome_by_age, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Outcome Percentage by Age Group')
plt.tight_layout()
plt.savefig('eda_outputs/outcome_by_age_heatmap.png')
plt.close()

# Readmission Analysis
print("\n=== READMISSION ANALYSIS ===")

# Readmission rate
readmission_rate = df['Readmitted'].mean() * 100
print(f"\nOverall Readmission Rate: {readmission_rate:.2f}%")

# Readmission by outcome
readmission_by_outcome = pd.crosstab(df['Outcome'], df['Readmitted'], normalize='index') * 100
print("\nReadmission Percentage by Outcome:")
print(readmission_by_outcome)

# Create and save plot for readmission by outcome
plt.figure(figsize=(10, 6))
sns.countplot(x='Outcome', hue='Readmitted', data=df)
plt.title('Readmission by Outcome')
plt.tight_layout()
plt.savefig('eda_outputs/readmission_by_outcome_count.png')
plt.close()

# Readmission by primary diagnosis (top 10)
readmission_by_diagnosis = df.groupby('PrimaryDiagnosis')['Readmitted'].mean().sort_values(ascending=False).head(10) * 100
print("\nTop 10 Diagnoses with Highest Readmission Rates:")
print(readmission_by_diagnosis)

# Create and save plot for readmission by diagnosis
plt.figure(figsize=(12, 8))
sns.barplot(y=readmission_by_diagnosis.index, x=readmission_by_diagnosis.values)
plt.title('Top 10 Diagnoses with Highest Readmission Rates (%)')
plt.tight_layout()
plt.savefig('eda_outputs/readmission_by_diagnosis_barplot.png')
plt.close()

# Cost Analysis
print("\n=== COST ANALYSIS ===")

# Cost statistics
print("\nCost Statistics:")
print(df[['TotalCost', 'PatientPayment', 'InsurancePayment']].describe())

# Average cost by insurance type
cost_by_insurance = df.groupby('InsuranceType')[['TotalCost', 'PatientPayment', 'InsurancePayment']].mean()
print("\nAverage Cost by Insurance Type:")
print(cost_by_insurance)

# Create and save plot for cost by insurance type
plt.figure(figsize=(12, 8))
cost_by_insurance.plot(kind='bar', figsize=(12, 6))
plt.title('Average Costs by Insurance Type')
plt.ylabel('Amount ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_outputs/cost_by_insurance_barplot.png')
plt.close()

# Average cost by age group
cost_by_age = df.groupby('AgeGroup')['TotalCost'].mean().reset_index()
print("\nAverage Cost by Age Group:")
print(cost_by_age)

# Create and save plot for cost by age group
plt.figure(figsize=(10, 6))
sns.barplot(x='AgeGroup', y='TotalCost', data=cost_by_age)
plt.title('Average Cost by Age Group')
plt.tight_layout()
plt.savefig('eda_outputs/cost_by_age_barplot.png')
plt.close()

# Time Analysis
print("\n=== TIME ANALYSIS ===")

# Admissions by month
admissions_by_month = df['AdmissionMonth'].value_counts().sort_index()
print("\nAdmissions by Month:")
print(admissions_by_month)

# Create and save plot for admissions by month
plt.figure(figsize=(12, 6))
sns.barplot(x=admissions_by_month.index, y=admissions_by_month.values)
plt.title('Admissions by Month')
plt.xlabel('Month')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
plt.savefig('eda_outputs/admissions_by_month_barplot.png')
plt.close()

# Admissions by day of week
admissions_by_dow = df['AdmissionDayOfWeek'].value_counts().sort_index()
print("\nAdmissions by Day of Week:")
print(admissions_by_dow)

# Create and save plot for admissions by day of week
plt.figure(figsize=(10, 6))
sns.barplot(x=admissions_by_dow.index, y=admissions_by_dow.values)
plt.title('Admissions by Day of Week')
plt.xlabel('Day of Week')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.tight_layout()
plt.savefig('eda_outputs/admissions_by_dow_barplot.png')
plt.close()

# Summary of Key Findings
print("\n=== SUMMARY OF KEY FINDINGS ===")
print("""
1. Demographics:
   - Age distribution shows [summary of age distribution]
   - Gender distribution is [summary of gender distribution]
   - Most common blood type is [most common blood type]

2. Admissions:
   - Most common admission type is [most common admission type]
   - Average length of stay is [average LOS] days
   - Admissions are [higher/lower] on weekends compared to weekdays

3. Diagnoses and Treatments:
   - Most common diagnosis is [most common diagnosis]
   - Most expensive diagnosis to treat is [most expensive diagnosis]
   - Most common treatment is [most common treatment]

4. Outcomes:
   - [Percentage] of patients fully recovered
   - [Percentage] of patients were readmitted
   - Readmission rates are highest for [diagnosis with highest readmission]

5. Costs:
   - Average total cost per patient is $[average cost]
   - Insurance coverage averages [average coverage]%
   - [Insurance type] insurance has the highest average coverage
""")

print("\nExploratory Data Analysis complete. Results saved to 'eda_outputs' directory.")
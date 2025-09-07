import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set style for matplotlib plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories for outputs if they don't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')
if not os.path.exists('visualizations/static'):
    os.makedirs('visualizations/static')
if not os.path.exists('visualizations/interactive'):
    os.makedirs('visualizations/interactive')

print("Loading hospital patient dataset...")

# Load the dataset
try:
    df = pd.read_csv('hospital_patient_data.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please run generate_dataset.py first.")
    exit(1)

# Data preprocessing
print("Preprocessing data for visualizations...")

# Convert date columns to datetime
df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])

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

# Create month and year features for time analysis
df['AdmissionMonth'] = df['AdmissionDate'].dt.month
df['AdmissionYear'] = df['AdmissionDate'].dt.year
df['AdmissionDayOfWeek'] = df['AdmissionDate'].dt.dayofweek
df['AdmissionMonthName'] = df['AdmissionDate'].dt.month_name()
df['AdmissionDayName'] = df['AdmissionDate'].dt.day_name()

# 1. DEMOGRAPHIC VISUALIZATIONS
print("\nGenerating demographic visualizations...")

# Age and Gender Distribution - Pyramid Chart
def create_age_gender_pyramid():
    # Create age bins for better visualization
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91+']    
    df['AgeBin'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
    
    # Count by age bin and gender
    male_counts = df[df['Gender'] == 'Male'].groupby('AgeBin').size()
    female_counts = df[df['Gender'] == 'Female'].groupby('AgeBin').size()
    
    # Create the pyramid chart
    fig = go.Figure()
    
    # Add male bars
    fig.add_trace(go.Bar(
        y=age_labels,
        x=-male_counts,  # Negative values for left side
        name='Male',
        orientation='h',
        marker=dict(color='#1E88E5')
    ))
    
    # Add female bars
    fig.add_trace(go.Bar(
        y=age_labels,
        x=female_counts,  # Positive values for right side
        name='Female',
        orientation='h',
        marker=dict(color='#FFC107')
    ))
    
    # Update layout
    fig.update_layout(
        title='Age and Gender Distribution',
        barmode='relative',
        bargap=0.1,
        xaxis=dict(
            title='Count',
            tickvals=[-100, -75, -50, -25, 0, 25, 50, 75, 100],
            ticktext=['100', '75', '50', '25', '0', '25', '50', '75', '100'],
        ),
        yaxis=dict(title='Age Group'),
        legend=dict(x=0.5, y=1.1, orientation='h'),
        height=600,
        width=900
    )
    
    # Save the figure
    fig.write_html('visualizations/interactive/age_gender_pyramid.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    
    # Plot the pyramid
    y_pos = np.arange(len(age_labels))
    plt.barh(y_pos, -male_counts, align='center', color='#1E88E5', label='Male')
    plt.barh(y_pos, female_counts, align='center', color='#FFC107', label='Female')
    
    # Add labels and title
    plt.yticks(y_pos, age_labels)
    plt.xlabel('Count')
    plt.ylabel('Age Group')
    plt.title('Age and Gender Distribution')
    plt.legend()
    
    # Add a grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('visualizations/static/age_gender_pyramid.png', dpi=300)
    plt.close()

# BMI Distribution by Gender
def create_bmi_distribution():
    # Create a violin plot with Plotly
    fig = px.violin(df, y="BMI", x="Gender", color="Gender", box=True, points="all",
                   title="BMI Distribution by Gender",
                   labels={"BMI": "Body Mass Index", "Gender": "Gender"},
                   color_discrete_map={"Male": "#1E88E5", "Female": "#FFC107"})
    
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/bmi_distribution.html')
    
    # Create a static version for the report
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Gender", y="BMI", data=df, palette={"Male": "#1E88E5", "Female": "#FFC107"})
    plt.title('BMI Distribution by Gender')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/bmi_distribution.png', dpi=300)
    plt.close()

# Blood Type Distribution
def create_blood_type_distribution():
    # Count blood types
    blood_counts = df['BloodType'].value_counts().reset_index()
    blood_counts.columns = ['BloodType', 'Count']
    
    # Calculate percentages
    blood_counts['Percentage'] = blood_counts['Count'] / blood_counts['Count'].sum() * 100
    
    # Create a pie chart with Plotly
    fig = px.pie(blood_counts, values='Percentage', names='BloodType',
                title='Blood Type Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/blood_type_distribution.html')
    
    # Create a static version for the report
    plt.figure(figsize=(10, 8))
    plt.pie(blood_counts['Percentage'], labels=blood_counts['BloodType'], autopct='%1.1f%%',
           startangle=90, colors=sns.color_palette("Set3", len(blood_counts)))
    plt.title('Blood Type Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.savefig('visualizations/static/blood_type_distribution.png', dpi=300)
    plt.close()

# 2. ADMISSION AND DIAGNOSIS VISUALIZATIONS
print("\nGenerating admission and diagnosis visualizations...")

# Top 10 Diagnoses
def create_top_diagnoses_chart():
    # Get top 10 diagnoses
    top_diagnoses = df['PrimaryDiagnosis'].value_counts().head(10).reset_index()
    top_diagnoses.columns = ['Diagnosis', 'Count']
    
    # Create a horizontal bar chart with Plotly
    fig = px.bar(top_diagnoses, x='Count', y='Diagnosis', orientation='h',
                title='Top 10 Primary Diagnoses',
                color='Count', color_continuous_scale='Viridis',
                labels={'Count': 'Number of Patients', 'Diagnosis': 'Primary Diagnosis'})
    
    fig.update_layout(height=600, width=900, yaxis={'categoryorder':'total ascending'})
    fig.write_html('visualizations/interactive/top_diagnoses.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Count', y='Diagnosis', data=top_diagnoses, palette='viridis')
    plt.title('Top 10 Primary Diagnoses')
    plt.xlabel('Number of Patients')
    plt.ylabel('Primary Diagnosis')
    
    # Add count labels to the bars
    for i, v in enumerate(top_diagnoses['Count']):
        ax.text(v + 1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/static/top_diagnoses.png', dpi=300)
    plt.close()

# Admission Types Distribution
def create_admission_types_chart():
    # Count admission types
    admission_counts = df['AdmissionType'].value_counts().reset_index()
    admission_counts.columns = ['AdmissionType', 'Count']
    
    # Create a donut chart with Plotly
    fig = px.pie(admission_counts, values='Count', names='AdmissionType',
                title='Admission Types Distribution',
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.5)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/admission_types.html')
    
    # Create a static version for the report
    plt.figure(figsize=(10, 8))
    plt.pie(admission_counts['Count'], labels=admission_counts['AdmissionType'], autopct='%1.1f%%',
           startangle=90, colors=sns.color_palette("Set2", len(admission_counts)))
    plt.title('Admission Types Distribution')
    centre_circle = plt.Circle((0,0),0.5,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/static/admission_types.png', dpi=300)
    plt.close()

# Admissions by Month and Day of Week
def create_admissions_time_charts():
    # Admissions by month
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_counts = df['AdmissionMonthName'].value_counts().reindex(month_order).reset_index()
    month_counts.columns = ['Month', 'Count']
    
    # Admissions by day of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['AdmissionDayName'].value_counts().reindex(day_order).reset_index()
    day_counts.columns = ['Day', 'Count']
    
    # Create subplots with Plotly
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Admissions by Month', 'Admissions by Day of Week'),
                       vertical_spacing=0.1, row_heights=[0.5, 0.5])
    
    # Add month bar chart
    fig.add_trace(
        go.Bar(x=month_counts['Month'], y=month_counts['Count'], marker_color='#1E88E5', name='Month'),
        row=1, col=1
    )
    
    # Add day of week bar chart
    fig.add_trace(
        go.Bar(x=day_counts['Day'], y=day_counts['Count'], marker_color='#FFC107', name='Day'),
        row=2, col=1
    )
    
    fig.update_layout(height=800, width=1000, title_text="Hospital Admissions by Time")
    fig.write_html('visualizations/interactive/admissions_time.html')
    
    # Create static versions for the report
    # Month chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month', y='Count', data=month_counts, color='#1E88E5')
    plt.title('Admissions by Month')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/admissions_by_month.png', dpi=300)
    plt.close()
    
    # Day of week chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Day', y='Count', data=day_counts, color='#FFC107')
    plt.title('Admissions by Day of Week')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/admissions_by_day.png', dpi=300)
    plt.close()

# 3. TREATMENT AND OUTCOME VISUALIZATIONS
print("\nGenerating treatment and outcome visualizations...")

# Treatment Types and Costs
def create_treatment_cost_chart():
    # Calculate average cost by treatment
    treatment_costs = df.groupby('Treatment')['TotalCost'].agg(['mean', 'count']).reset_index()
    treatment_costs.columns = ['Treatment', 'AverageCost', 'Count']
    treatment_costs = treatment_costs.sort_values('AverageCost', ascending=False)
    
    # Create a bar chart with Plotly
    fig = px.bar(treatment_costs, x='Treatment', y='AverageCost',
                title='Average Cost by Treatment Type',
                color='Count', color_continuous_scale='Viridis',
                labels={'AverageCost': 'Average Cost ($)', 'Treatment': 'Treatment Type', 'Count': 'Number of Patients'},
                text_auto='.2s')
    
    fig.update_traces(texttemplate='$%{y:.2f}', textposition='outside')
    fig.update_layout(height=600, width=1000, xaxis={'categoryorder':'total descending'})
    fig.write_html('visualizations/interactive/treatment_costs.html')
    
    # Create a static version for the report
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Treatment', y='AverageCost', data=treatment_costs, palette='viridis')
    plt.title('Average Cost by Treatment Type')
    plt.xlabel('Treatment Type')
    plt.ylabel('Average Cost ($)')
    plt.xticks(rotation=45)
    
    # Add cost labels to the bars
    for i, v in enumerate(treatment_costs['AverageCost']):
        ax.text(i, v + 100, f'${v:.2f}', ha='center', va='bottom', rotation=90, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visualizations/static/treatment_costs.png', dpi=300)
    plt.close()

# Patient Outcomes Visualization
def create_outcomes_chart():
    # Count outcomes
    outcome_counts = df['Outcome'].value_counts().reset_index()
    outcome_counts.columns = ['Outcome', 'Count']
    
    # Calculate percentages
    outcome_counts['Percentage'] = outcome_counts['Count'] / outcome_counts['Count'].sum() * 100
    
    # Create a pie chart with Plotly
    fig = px.pie(outcome_counts, values='Count', names='Outcome',
                title='Patient Outcomes Distribution',
                color_discrete_sequence=px.colors.sequential.Viridis,
                hole=0.3)
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/patient_outcomes.html')
    
    # Create a static version for the report
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(outcome_counts))
    plt.pie(outcome_counts['Count'], labels=outcome_counts['Outcome'], autopct='%1.1f%%',
           startangle=90, colors=colors, explode=[0.05] * len(outcome_counts))
    plt.title('Patient Outcomes Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('visualizations/static/patient_outcomes.png', dpi=300)
    plt.close()

# Outcome by Age Group
def create_outcome_by_age_chart():
    # Create a crosstab of age group and outcome
    outcome_age = pd.crosstab(df['AgeGroup'], df['Outcome'])
    outcome_age_pct = pd.crosstab(df['AgeGroup'], df['Outcome'], normalize='index') * 100
    
    # Create a heatmap with Plotly
    fig = px.imshow(outcome_age_pct, text_auto='.1f', aspect='auto',
                   labels=dict(x='Outcome', y='Age Group', color='Percentage'),
                   title='Outcome Percentage by Age Group',
                   color_continuous_scale='Viridis')
    
    fig.update_layout(height=600, width=1000)
    fig.write_html('visualizations/interactive/outcome_by_age.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    sns.heatmap(outcome_age_pct, annot=True, cmap='viridis', fmt='.1f')
    plt.title('Outcome Percentage by Age Group')
    plt.tight_layout()
    plt.savefig('visualizations/static/outcome_by_age.png', dpi=300)
    plt.close()
    
    # Create a stacked bar chart for absolute numbers
    outcome_age_melted = outcome_age.reset_index().melt(id_vars='AgeGroup', var_name='Outcome', value_name='Count')
    
    fig = px.bar(outcome_age_melted, x='AgeGroup', y='Count', color='Outcome',
                title='Outcome by Age Group (Absolute Numbers)',
                labels={'AgeGroup': 'Age Group', 'Count': 'Number of Patients'},
                color_discrete_sequence=px.colors.qualitative.Bold)
    
    fig.update_layout(height=600, width=1000)
    fig.write_html('visualizations/interactive/outcome_by_age_counts.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    outcome_pivot = outcome_age_melted.pivot(index='AgeGroup', columns='Outcome', values='Count')
    outcome_pivot.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='tab10')
    plt.title('Outcome by Age Group (Absolute Numbers)')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Patients')
    plt.legend(title='Outcome')
    plt.tight_layout()
    plt.savefig('visualizations/static/outcome_by_age_counts.png', dpi=300)
    plt.close()

# 4. FINANCIAL VISUALIZATIONS
print("\nGenerating financial visualizations...")

# Cost Distribution
def create_cost_distribution_chart():
    # Create a histogram with Plotly
    fig = px.histogram(df, x='TotalCost', nbins=50,
                      title='Distribution of Total Hospital Costs',
                      labels={'TotalCost': 'Total Cost ($)'},
                      color_discrete_sequence=['#1E88E5'])
    
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/cost_distribution.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    sns.histplot(df['TotalCost'], bins=50, kde=True, color='#1E88E5')
    plt.title('Distribution of Total Hospital Costs')
    plt.xlabel('Total Cost ($)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/cost_distribution.png', dpi=300)
    plt.close()

# Insurance Coverage and Costs
def create_insurance_cost_chart():
    # Calculate average costs by insurance type
    insurance_costs = df.groupby('InsuranceType').agg({
        'TotalCost': 'mean',
        'PatientPayment': 'mean',
        'InsurancePayment': 'mean',
        'InsuranceCoverage': 'mean'
    }).reset_index()
    
    # Create a grouped bar chart with Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=insurance_costs['InsuranceType'],
        y=insurance_costs['TotalCost'],
        name='Total Cost',
        marker_color='#1E88E5'
    ))
    
    fig.add_trace(go.Bar(
        x=insurance_costs['InsuranceType'],
        y=insurance_costs['InsurancePayment'],
        name='Insurance Payment',
        marker_color='#43A047'
    ))
    
    fig.add_trace(go.Bar(
        x=insurance_costs['InsuranceType'],
        y=insurance_costs['PatientPayment'],
        name='Patient Payment',
        marker_color='#FFC107'
    ))
    
    fig.update_layout(
        title='Average Costs by Insurance Type',
        xaxis_title='Insurance Type',
        yaxis_title='Amount ($)',
        barmode='group',
        height=600,
        width=1000
    )
    
    fig.write_html('visualizations/interactive/insurance_costs.html')
    
    # Create a static version for the report
    plt.figure(figsize=(14, 8))
    
    # Reshape data for seaborn
    insurance_costs_melted = insurance_costs.melt(
        id_vars='InsuranceType',
        value_vars=['TotalCost', 'InsurancePayment', 'PatientPayment'],
        var_name='Cost Type',
        value_name='Amount'
    )
    
    # Create the grouped bar chart
    sns.barplot(x='InsuranceType', y='Amount', hue='Cost Type', data=insurance_costs_melted,
               palette={'TotalCost': '#1E88E5', 'InsurancePayment': '#43A047', 'PatientPayment': '#FFC107'})
    
    plt.title('Average Costs by Insurance Type')
    plt.xlabel('Insurance Type')
    plt.ylabel('Amount ($)')
    plt.xticks(rotation=0)
    plt.legend(title='Cost Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/insurance_costs.png', dpi=300)
    plt.close()
    
    # Create a coverage percentage chart
    fig = px.bar(insurance_costs, x='InsuranceType', y='InsuranceCoverage',
                title='Average Insurance Coverage Percentage by Type',
                labels={'InsuranceType': 'Insurance Type', 'InsuranceCoverage': 'Coverage Percentage (%)'},
                color='InsuranceCoverage', color_continuous_scale='Viridis',
                text_auto='.1f')
    
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/insurance_coverage.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='InsuranceType', y='InsuranceCoverage', data=insurance_costs, palette='viridis')
    plt.title('Average Insurance Coverage Percentage by Type')
    plt.xlabel('Insurance Type')
    plt.ylabel('Coverage Percentage (%)')
    
    # Add percentage labels to the bars
    for i, v in enumerate(insurance_costs['InsuranceCoverage']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/insurance_coverage.png', dpi=300)
    plt.close()

# 5. CORRELATION AND RELATIONSHIP VISUALIZATIONS
print("\nGenerating correlation and relationship visualizations...")

# Correlation Heatmap
def create_correlation_heatmap():
    # Select numerical columns for correlation
    numerical_cols = ['Age', 'BMI', 'LengthOfStay', 'TotalCost', 'InsuranceCoverage', 'PatientPayment', 'InsurancePayment']
    corr_matrix = df[numerical_cols].corr()
    
    # Create a heatmap with Plotly
    fig = px.imshow(corr_matrix, text_auto='.2f',
                   labels=dict(x='Variable', y='Variable', color='Correlation'),
                   title='Correlation Matrix of Numerical Variables',
                   color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    
    fig.update_layout(height=700, width=700)
    fig.write_html('visualizations/interactive/correlation_heatmap.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', vmin=-1, vmax=1, mask=mask, fmt='.2f')
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.savefig('visualizations/static/correlation_heatmap.png', dpi=300)
    plt.close()

# Age vs Length of Stay vs Cost
def create_age_los_cost_chart():
    # Create a scatter plot with Plotly
    fig = px.scatter(df, x='Age', y='LengthOfStay', color='TotalCost', size='TotalCost',
                    title='Relationship between Age, Length of Stay, and Total Cost',
                    labels={'Age': 'Age (years)', 'LengthOfStay': 'Length of Stay (days)', 'TotalCost': 'Total Cost ($)'},
                    color_continuous_scale='Viridis')
    
    fig.update_layout(height=600, width=900)
    fig.write_html('visualizations/interactive/age_los_cost.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['Age'], df['LengthOfStay'], c=df['TotalCost'], cmap='viridis', alpha=0.7, s=df['TotalCost']/100)
    plt.colorbar(scatter, label='Total Cost ($)')
    plt.title('Relationship between Age, Length of Stay, and Total Cost')
    plt.xlabel('Age (years)')
    plt.ylabel('Length of Stay (days)')
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/age_los_cost.png', dpi=300)
    plt.close()

# 6. READMISSION ANALYSIS VISUALIZATIONS
print("\nGenerating readmission analysis visualizations...")

# Readmission Rates by Diagnosis
def create_readmission_by_diagnosis_chart():
    # Calculate readmission rates by diagnosis
    readmission_rates = df.groupby('PrimaryDiagnosis')['Readmitted'].mean().sort_values(ascending=False).head(10) * 100
    readmission_df = readmission_rates.reset_index()
    readmission_df.columns = ['PrimaryDiagnosis', 'ReadmissionRate']
    
    # Create a bar chart with Plotly
    fig = px.bar(readmission_df, x='ReadmissionRate', y='PrimaryDiagnosis', orientation='h',
                title='Top 10 Diagnoses with Highest Readmission Rates',
                labels={'ReadmissionRate': 'Readmission Rate (%)', 'PrimaryDiagnosis': 'Primary Diagnosis'},
                color='ReadmissionRate', color_continuous_scale='Reds',
                text_auto='.1f')
    
    fig.update_traces(texttemplate='%{x:.1f}%', textposition='outside')
    fig.update_layout(height=600, width=900, yaxis={'categoryorder':'total ascending'})
    fig.write_html('visualizations/interactive/readmission_by_diagnosis.html')
    
    # Create a static version for the report
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='ReadmissionRate', y='PrimaryDiagnosis', data=readmission_df, palette='Reds_r')
    plt.title('Top 10 Diagnoses with Highest Readmission Rates')
    plt.xlabel('Readmission Rate (%)')
    plt.ylabel('Primary Diagnosis')
    
    # Add percentage labels to the bars
    for i, v in enumerate(readmission_df['ReadmissionRate']):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center')
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/readmission_by_diagnosis.png', dpi=300)
    plt.close()

# Readmission by Age Group and Outcome
def create_readmission_by_age_outcome_chart():
    # Calculate readmission rates by age group
    readmission_by_age = df.groupby('AgeGroup')['Readmitted'].mean() * 100
    readmission_by_age_df = readmission_by_age.reset_index()
    readmission_by_age_df.columns = ['AgeGroup', 'ReadmissionRate']
    
    # Calculate readmission rates by outcome
    readmission_by_outcome = df.groupby('Outcome')['Readmitted'].mean() * 100
    readmission_by_outcome_df = readmission_by_outcome.reset_index()
    readmission_by_outcome_df.columns = ['Outcome', 'ReadmissionRate']
    
    # Create subplots with Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Readmission Rate by Age Group', 'Readmission Rate by Outcome'),
                       horizontal_spacing=0.1)
    
    # Add age group bar chart
    fig.add_trace(
        go.Bar(x=readmission_by_age_df['AgeGroup'], y=readmission_by_age_df['ReadmissionRate'],
              marker_color='#1E88E5', text=readmission_by_age_df['ReadmissionRate'].round(1).astype(str) + '%',
              textposition='outside', name='Age Group'),
        row=1, col=1
    )
    
    # Add outcome bar chart
    fig.add_trace(
        go.Bar(x=readmission_by_outcome_df['Outcome'], y=readmission_by_outcome_df['ReadmissionRate'],
              marker_color='#FFC107', text=readmission_by_outcome_df['ReadmissionRate'].round(1).astype(str) + '%',
              textposition='outside', name='Outcome'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, width=1200, title_text="Readmission Analysis",
                     showlegend=False, yaxis_title="Readmission Rate (%)")
    fig.write_html('visualizations/interactive/readmission_analysis.html')
    
    # Create static versions for the report
    # Age group chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='AgeGroup', y='ReadmissionRate', data=readmission_by_age_df, color='#1E88E5')
    plt.title('Readmission Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Readmission Rate (%)')
    
    # Add percentage labels to the bars
    for i, v in enumerate(readmission_by_age_df['ReadmissionRate']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/readmission_by_age.png', dpi=300)
    plt.close()
    
    # Outcome chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Outcome', y='ReadmissionRate', data=readmission_by_outcome_df, color='#FFC107')
    plt.title('Readmission Rate by Outcome')
    plt.xlabel('Outcome')
    plt.ylabel('Readmission Rate (%)')
    
    # Add percentage labels to the bars
    for i, v in enumerate(readmission_by_outcome_df['ReadmissionRate']):
        ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/static/readmission_by_outcome.png', dpi=300)
    plt.close()

# Execute all visualization functions
print("\nGenerating all visualizations...")

# Demographic visualizations
create_age_gender_pyramid()
create_bmi_distribution()
create_blood_type_distribution()

# Admission and diagnosis visualizations
create_top_diagnoses_chart()
create_admission_types_chart()
create_admissions_time_charts()

# Treatment and outcome visualizations
create_treatment_cost_chart()
create_outcomes_chart()
create_outcome_by_age_chart()

# Financial visualizations
create_cost_distribution_chart()
create_insurance_cost_chart()

# Correlation and relationship visualizations
create_correlation_heatmap()
create_age_los_cost_chart()

# Readmission analysis visualizations
create_readmission_by_diagnosis_chart()
create_readmission_by_age_outcome_chart()

print("\nAll visualizations have been generated and saved to the 'visualizations' directory.")
print("- Static visualizations are in the 'visualizations/static' folder")
print("- Interactive visualizations are in the 'visualizations/interactive' folder")
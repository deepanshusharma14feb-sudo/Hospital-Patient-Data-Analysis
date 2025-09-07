import pandas as pd
import numpy as np
import os
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Create directory for dashboard if it doesn't exist
if not os.path.exists('dashboard'):
    os.makedirs('dashboard')

# Load the dataset
try:
    df = pd.read_csv('hospital_patient_data.csv')
    print(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
except FileNotFoundError:
    print("Error: Dataset file not found. Please run generate_dataset.py first.")
    exit(1)

# Function to convert an image file to base64 for embedding in HTML
def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found: {image_path}")
        return ""

# Function to create HTML for embedding an interactive Plotly chart
def embed_interactive_chart(html_path, height=600):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            # Extract just the chart part (between <body> and </body>)
            start_idx = html_content.find('<body>')
            end_idx = html_content.find('</body>')
            if start_idx != -1 and end_idx != -1:
                chart_content = html_content[start_idx + len('<body>'):end_idx]
                return f"<div style='height:{height}px;'>{chart_content}</div>"
            else:
                return f"<iframe src='{os.path.basename(html_path)}' width='100%' height='{height}px' frameborder='0'></iframe>"
    except FileNotFoundError:
        print(f"Warning: Interactive chart file not found: {html_path}")
        return f"<p>Chart not available: {html_path}</p>"

# Calculate key statistics for the dashboard
def calculate_key_stats():
    stats = {}
    
    # Patient demographics
    stats['total_patients'] = len(df)
    stats['avg_age'] = round(df['Age'].mean(), 1)
    stats['gender_ratio'] = f"{round(sum(df['Gender'] == 'Male') / len(df) * 100, 1)}% Male / {round(sum(df['Gender'] == 'Female') / len(df) * 100, 1)}% Female"
    
    # Hospital stays
    stats['avg_los'] = round(df['LengthOfStay'].mean(), 1)
    stats['max_los'] = df['LengthOfStay'].max()
    
    # Financial
    stats['avg_cost'] = f"${round(df['TotalCost'].mean(), 2):,.2f}"
    stats['total_cost'] = f"${round(df['TotalCost'].sum(), 2):,.2f}"
    stats['avg_insurance_coverage'] = f"{round(df['InsuranceCoverage'].mean() * 100, 1)}%"
    
    # Outcomes
    stats['recovery_rate'] = f"{round(sum(df['Outcome'] == 'Recovered') / len(df) * 100, 1)}%"
    stats['mortality_rate'] = f"{round(sum(df['Outcome'] == 'Deceased') / len(df) * 100, 1)}%"
    stats['readmission_rate'] = f"{round(df['Readmitted'].mean() * 100, 1)}%"
    
    # Most common
    stats['top_diagnosis'] = df['PrimaryDiagnosis'].value_counts().index[0]
    stats['top_treatment'] = df['Treatment'].value_counts().index[0]
    stats['top_insurance'] = df['InsuranceType'].value_counts().index[0]
    
    return stats

# Generate key insights based on the data
def generate_insights():
    insights = []
    
    # Age and demographics insights
    elderly_patients = sum(df['Age'] > 65) / len(df) * 100
    insights.append(f"Elderly patients (over 65) make up {elderly_patients:.1f}% of all hospital admissions.")
    
    # Gender-specific insights
    gender_los = df.groupby('Gender')['LengthOfStay'].mean()
    gender_cost = df.groupby('Gender')['TotalCost'].mean()
    if gender_los['Male'] > gender_los['Female']:
        insights.append(f"Male patients stay in the hospital longer on average ({gender_los['Male']:.1f} days vs {gender_los['Female']:.1f} days for females).")
    else:
        insights.append(f"Female patients stay in the hospital longer on average ({gender_los['Female']:.1f} days vs {gender_los['Male']:.1f} days for males).")
    
    # Cost insights
    top_cost_diagnosis = df.groupby('PrimaryDiagnosis')['TotalCost'].mean().sort_values(ascending=False).index[0]
    insights.append(f"{top_cost_diagnosis} is the most expensive diagnosis to treat, with an average cost of ${df[df['PrimaryDiagnosis'] == top_cost_diagnosis]['TotalCost'].mean():,.2f}.")
    
    # Length of stay insights
    los_corr = df['LengthOfStay'].corr(df['Age'])
    if los_corr > 0.3:
        insights.append(f"There is a moderate positive correlation ({los_corr:.2f}) between patient age and length of stay, suggesting older patients typically require longer hospitalization.")
    
    # Readmission insights
    top_readmit_diagnosis = df.groupby('PrimaryDiagnosis')['Readmitted'].mean().sort_values(ascending=False).index[0]
    readmit_rate = df[df['PrimaryDiagnosis'] == top_readmit_diagnosis]['Readmitted'].mean() * 100
    insights.append(f"Patients diagnosed with {top_readmit_diagnosis} have the highest readmission rate at {readmit_rate:.1f}%.")
    
    # Outcome insights
    # Create age categories if they don't exist
    if 'AgeCategory' not in df.columns:
        df['AgeCategory'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100], 
                                  labels=['0-18', '19-35', '36-50', '51-65', '66+'])
    
    outcome_by_age = pd.crosstab(df['AgeCategory'], df['Outcome'], normalize='index')
    worst_outcome_age = outcome_by_age['Deceased'].idxmax()
    best_outcome_age = outcome_by_age['Recovered'].idxmax()
    insights.append(f"The {worst_outcome_age} age group has the highest mortality rate, while the {best_outcome_age} age group has the best recovery rate.")
    
    # Insurance insights
    insurance_coverage = df.groupby('InsuranceType')['InsuranceCoverage'].mean().sort_values(ascending=False)
    best_insurance = insurance_coverage.index[0]
    worst_insurance = insurance_coverage.index[-1]
    insights.append(f"{best_insurance} provides the highest average coverage rate at {insurance_coverage[best_insurance]*100:.1f}%, while {worst_insurance} has the lowest at {insurance_coverage[worst_insurance]*100:.1f}%.")
    
    # Treatment effectiveness
    treatment_recovery = df.groupby('Treatment').apply(lambda x: sum(x['Outcome'] == 'Recovered') / len(x) * 100).sort_values(ascending=False)
    best_treatment = treatment_recovery.index[0]
    insights.append(f"{best_treatment} shows the highest recovery rate at {treatment_recovery[best_treatment]:.1f}% among all treatments.")
    
    # Seasonal patterns
    if 'AdmissionMonth' not in df.columns:
        df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
        df['AdmissionMonth'] = df['AdmissionDate'].dt.month
    
    busiest_month = df['AdmissionMonth'].value_counts().idxmax()
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 
                  7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    insights.append(f"{month_names[busiest_month]} is the busiest month for hospital admissions.")
    
    return insights

# Create the HTML dashboard
def create_dashboard():
    # Calculate statistics
    stats = calculate_key_stats()
    
    # Generate insights
    insights = generate_insights()
    
    # Current date for the report
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Start building the HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hospital Patient Data Analysis Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .dashboard-header {{ background-color: #343a40; color: white; padding: 20px 0; margin-bottom: 30px; }}
            .stat-card {{ background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }}
            .stat-card:hover {{ transform: translateY(-5px); box-shadow: 0 6px 8px rgba(0,0,0,0.15); }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #343a40; }}
            .stat-label {{ font-size: 14px; color: #6c757d; }}
            .insight-card {{ background-color: #e9ecef; border-left: 4px solid #007bff; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
            .section-title {{ margin: 40px 0 20px 0; padding-bottom: 10px; border-bottom: 2px solid #dee2e6; }}
            .chart-container {{ background-color: white; border-radius: 10px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            .nav-pills .nav-link.active {{ background-color: #343a40; }}
            .tab-content {{ padding: 20px 0; }}
            footer {{ background-color: #343a40; color: white; padding: 20px 0; margin-top: 50px; }}
        </style>
    </head>
    <body>
        <div class="dashboard-header text-center">
            <div class="container">
                <h1>Hospital Patient Data Analysis</h1>
                <p class="lead">Comprehensive analysis of patient demographics, treatments, outcomes, and costs</p>
                <p>Report generated on {current_date}</p>
            </div>
        </div>
        
        <div class="container">
            <!-- Executive Summary -->
            <div class="row">
                <div class="col-12">
                    <h2 class="section-title">Executive Summary</h2>
                    <p>This dashboard presents a comprehensive analysis of hospital patient data, focusing on patient demographics, admission patterns, diagnoses, treatments, outcomes, and financial aspects. The analysis provides valuable insights into hospital operations, patient care effectiveness, and areas for potential improvement.</p>
                </div>
            </div>
            
            <!-- Key Statistics -->
            <div class="row">
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['total_patients']}</div>
                        <div class="stat-label">Total Patients</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['avg_age']}</div>
                        <div class="stat-label">Average Age</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['avg_los']}</div>
                        <div class="stat-label">Average Length of Stay (days)</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['avg_cost']}</div>
                        <div class="stat-label">Average Cost per Patient</div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['recovery_rate']}</div>
                        <div class="stat-label">Recovery Rate</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['readmission_rate']}</div>
                        <div class="stat-label">Readmission Rate</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['gender_ratio']}</div>
                        <div class="stat-label">Gender Ratio</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-center">
                        <div class="stat-value">{stats['avg_insurance_coverage']}</div>
                        <div class="stat-label">Average Insurance Coverage</div>
                    </div>
                </div>
            </div>
            
            <!-- Key Insights -->
            <div class="row">
                <div class="col-12">
                    <h2 class="section-title">Key Insights</h2>
                </div>
            </div>
            
            <div class="row">
    """
    
    # Add insights
    for i, insight in enumerate(insights):
        if i % 2 == 0:
            html_content += '<div class="col-md-6">'
        html_content += f'<div class="insight-card">{insight}</div>'
        if i % 2 == 1 or i == len(insights) - 1:
            html_content += '</div>'
    
    # Continue with the dashboard content - Tabs for different sections
    html_content += """
            </div>
            
            <!-- Analysis Tabs -->
            <div class="row mt-4">
                <div class="col-12">
                    <ul class="nav nav-pills mb-3" id="analysis-tabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="demographics-tab" data-bs-toggle="pill" data-bs-target="#demographics" type="button" role="tab" aria-controls="demographics" aria-selected="true">Demographics</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="diagnoses-tab" data-bs-toggle="pill" data-bs-target="#diagnoses" type="button" role="tab" aria-controls="diagnoses" aria-selected="false">Diagnoses & Admissions</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="treatments-tab" data-bs-toggle="pill" data-bs-target="#treatments" type="button" role="tab" aria-controls="treatments" aria-selected="false">Treatments & Outcomes</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="financial-tab" data-bs-toggle="pill" data-bs-target="#financial" type="button" role="tab" aria-controls="financial" aria-selected="false">Financial Analysis</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="correlations-tab" data-bs-toggle="pill" data-bs-target="#correlations" type="button" role="tab" aria-controls="correlations" aria-selected="false">Correlations</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="readmissions-tab" data-bs-toggle="pill" data-bs-target="#readmissions" type="button" role="tab" aria-controls="readmissions" aria-selected="false">Readmissions</button>
                        </li>
                    </ul>
                    
                    <div class="tab-content" id="analysis-tabsContent">
    """
    
    # Demographics Tab
    html_content += """
                        <div class="tab-pane fade show active" id="demographics" role="tabpanel" aria-labelledby="demographics-tab">
                            <h3>Patient Demographics</h3>
                            <p>Analysis of patient population by age, gender, BMI, and blood type.</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Age and Gender Distribution</h4>
    """
    
    # Add age-gender pyramid chart
    age_gender_img = get_image_base64('visualizations/static/age_gender_pyramid.png')
    if age_gender_img:
        html_content += f'<img src="data:image/png;base64,{age_gender_img}" class="img-fluid" alt="Age and Gender Distribution">'
    else:
        # Try to embed interactive version if static not found
        html_content += embed_interactive_chart('visualizations/interactive/age_gender_pyramid.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>BMI Distribution by Gender</h4>
    """
    
    # Add BMI distribution chart
    bmi_img = get_image_base64('visualizations/static/bmi_distribution.png')
    if bmi_img:
        html_content += f'<img src="data:image/png;base64,{bmi_img}" class="img-fluid" alt="BMI Distribution by Gender">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/bmi_distribution.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Blood Type Distribution</h4>
    """
    
    # Add blood type distribution chart
    blood_type_img = get_image_base64('visualizations/static/blood_type_distribution.png')
    if blood_type_img:
        html_content += f'<img src="data:image/png;base64,{blood_type_img}" class="img-fluid" alt="Blood Type Distribution">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/blood_type_distribution.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Key Demographic Findings</h4>
                                        <ul class="list-group">
                                            <li class="list-group-item">Average patient age is {stats['avg_age']} years</li>
                                            <li class="list-group-item">Gender distribution: {stats['gender_ratio']}</li>
                                            <li class="list-group-item">Most common blood type: {df['BloodType'].value_counts().index[0]}</li>
                                            <li class="list-group-item">Average BMI: {df['BMI'].mean():.1f}</li>
                                            <li class="list-group-item">{sum(df['BMI'] >= 30) / len(df) * 100:.1f}% of patients are classified as obese (BMI ≥ 30)</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
    """
    
    # Diagnoses & Admissions Tab
    html_content += """
                        <div class="tab-pane fade" id="diagnoses" role="tabpanel" aria-labelledby="diagnoses-tab">
                            <h3>Diagnoses and Admission Patterns</h3>
                            <p>Analysis of common diagnoses, admission types, and temporal patterns in hospital admissions.</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Top 10 Primary Diagnoses</h4>
    """
    
    # Add top diagnoses chart
    top_diag_img = get_image_base64('visualizations/static/top_diagnoses.png')
    if top_diag_img:
        html_content += f'<img src="data:image/png;base64,{top_diag_img}" class="img-fluid" alt="Top 10 Primary Diagnoses">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/top_diagnoses.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Admission Types Distribution</h4>
    """
    
    # Add admission types chart
    admission_types_img = get_image_base64('visualizations/static/admission_types.png')
    if admission_types_img:
        html_content += f'<img src="data:image/png;base64,{admission_types_img}" class="img-fluid" alt="Admission Types Distribution">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/admission_types.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Admissions by Month and Day of Week</h4>
    """
    
    # Add admissions time charts
    admissions_month_img = get_image_base64('visualizations/static/admissions_by_month.png')
    if admissions_month_img:
        html_content += f'<div class="row"><div class="col-md-6"><img src="data:image/png;base64,{admissions_month_img}" class="img-fluid" alt="Admissions by Month"></div>'
        
        admissions_day_img = get_image_base64('visualizations/static/admissions_by_day.png')
        if admissions_day_img:
            html_content += f'<div class="col-md-6"><img src="data:image/png;base64,{admissions_day_img}" class="img-fluid" alt="Admissions by Day"></div></div>'
        else:
            html_content += '</div>'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/admissions_time.html', height=700)
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Key Diagnosis and Admission Findings</h4>
                                        <ul class="list-group">
                                            <li class="list-group-item">Most common diagnosis: {stats['top_diagnosis']}</li>
                                            <li class="list-group-item">Most common admission type: {df['AdmissionType'].value_counts().index[0]} ({df['AdmissionType'].value_counts().iloc[0] / len(df) * 100:.1f}% of admissions)</li>
                                            <li class="list-group-item">Average length of stay: {stats['avg_los']} days</li>
                                            <li class="list-group-item">Maximum length of stay: {stats['max_los']} days</li>
                                            <li class="list-group-item">{sum(df['SecondaryDiagnosis'] != '') / len(df) * 100:.1f}% of patients have a secondary diagnosis</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
    """
    
    # Treatments & Outcomes Tab
    html_content += """
                        <div class="tab-pane fade" id="treatments" role="tabpanel" aria-labelledby="treatments-tab">
                            <h3>Treatments and Patient Outcomes</h3>
                            <p>Analysis of treatment effectiveness, patient outcomes, and factors affecting recovery.</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Treatment Types and Costs</h4>
    """
    
    # Add treatment cost chart
    treatment_cost_img = get_image_base64('visualizations/static/treatment_costs.png')
    if treatment_cost_img:
        html_content += f'<img src="data:image/png;base64,{treatment_cost_img}" class="img-fluid" alt="Treatment Types and Costs">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/treatment_costs.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Patient Outcomes Distribution</h4>
    """
    
    # Add outcomes chart
    outcomes_img = get_image_base64('visualizations/static/patient_outcomes.png')
    if outcomes_img:
        html_content += f'<img src="data:image/png;base64,{outcomes_img}" class="img-fluid" alt="Patient Outcomes Distribution">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/patient_outcomes.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Outcome by Age Group</h4>
    """
    
    # Add outcome by age chart
    outcome_age_img = get_image_base64('visualizations/static/outcome_by_age.png')
    if outcome_age_img:
        html_content += f'<img src="data:image/png;base64,{outcome_age_img}" class="img-fluid" alt="Outcome by Age Group">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/outcome_by_age.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Key Treatment and Outcome Findings</h4>
                                        <ul class="list-group">
                                            <li class="list-group-item">Most common treatment: {stats['top_treatment']}</li>
                                            <li class="list-group-item">Recovery rate: {stats['recovery_rate']}</li>
                                            <li class="list-group-item">Mortality rate: {stats['mortality_rate']}</li>
                                            <li class="list-group-item">Most effective treatment (highest recovery rate): {df.groupby('Treatment').apply(lambda x: sum(x['Outcome'] == 'Recovered') / len(x) * 100).sort_values(ascending=False).index[0]}</li>
                                            <li class="list-group-item">Age group with best outcomes: {pd.crosstab(df['AgeGroup'], df['Outcome'], normalize='index')['Recovered'].idxmax()}</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
    """
    
    # Financial Analysis Tab
    html_content += """
                        <div class="tab-pane fade" id="financial" role="tabpanel" aria-labelledby="financial-tab">
                            <h3>Financial Analysis</h3>
                            <p>Analysis of hospital costs, insurance coverage, and financial patterns.</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Cost Distribution</h4>
    """
    
    # Add cost distribution chart
    cost_dist_img = get_image_base64('visualizations/static/cost_distribution.png')
    if cost_dist_img:
        html_content += f'<img src="data:image/png;base64,{cost_dist_img}" class="img-fluid" alt="Cost Distribution">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/cost_distribution.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Insurance Coverage and Costs</h4>
    """
    
    # Add insurance cost chart
    insurance_cost_img = get_image_base64('visualizations/static/insurance_costs.png')
    if insurance_cost_img:
        html_content += f'<img src="data:image/png;base64,{insurance_cost_img}" class="img-fluid" alt="Insurance Coverage and Costs">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/insurance_costs.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Insurance Coverage Percentage by Type</h4>
    """
    
    # Add insurance coverage chart
    insurance_coverage_img = get_image_base64('visualizations/static/insurance_coverage.png')
    if insurance_coverage_img:
        html_content += f'<img src="data:image/png;base64,{insurance_coverage_img}" class="img-fluid" alt="Insurance Coverage Percentage by Type">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/insurance_coverage.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Key Financial Findings</h4>
                                        <ul class="list-group">
                                            <li class="list-group-item">Average cost per patient: {stats['avg_cost']}</li>
                                            <li class="list-group-item">Total hospital revenue: {stats['total_cost']}</li>
                                            <li class="list-group-item">Most common insurance type: {stats['top_insurance']}</li>
                                            <li class="list-group-item">Average insurance coverage: {stats['avg_insurance_coverage']}</li>
                                            <li class="list-group-item">Most expensive treatment: {df.groupby('Treatment')['TotalCost'].mean().sort_values(ascending=False).index[0]} (${df.groupby('Treatment')['TotalCost'].mean().sort_values(ascending=False).iloc[0]:,.2f})</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
    """
    
    # Correlations Tab
    html_content += """
                        <div class="tab-pane fade" id="correlations" role="tabpanel" aria-labelledby="correlations-tab">
                            <h3>Correlations and Relationships</h3>
                            <p>Analysis of relationships between different variables in the dataset.</p>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Correlation Heatmap</h4>
    """
    
    # Add correlation heatmap
    corr_heatmap_img = get_image_base64('visualizations/static/correlation_heatmap.png')
    if corr_heatmap_img:
        html_content += f'<img src="data:image/png;base64,{corr_heatmap_img}" class="img-fluid" alt="Correlation Heatmap">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/correlation_heatmap.html')
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Age vs Length of Stay vs Cost</h4>
    """
    
    # Add age-los-cost chart
    age_los_cost_img = get_image_base64('visualizations/static/age_los_cost.png')
    if age_los_cost_img:
        html_content += f'<img src="data:image/png;base64,{age_los_cost_img}" class="img-fluid" alt="Age vs Length of Stay vs Cost">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/age_los_cost.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Key Correlation Findings</h4>
                                        <ul class="list-group">
    """
    
    # Calculate some correlations for the findings
    age_los_corr = df['Age'].corr(df['LengthOfStay'])
    los_cost_corr = df['LengthOfStay'].corr(df['TotalCost'])
    age_cost_corr = df['Age'].corr(df['TotalCost'])
    bmi_los_corr = df['BMI'].corr(df['LengthOfStay'])
    
    html_content += f"""
                                            <li class="list-group-item">Age and Length of Stay: {age_los_corr:.2f} correlation coefficient</li>
                                            <li class="list-group-item">Length of Stay and Total Cost: {los_cost_corr:.2f} correlation coefficient</li>
                                            <li class="list-group-item">Age and Total Cost: {age_cost_corr:.2f} correlation coefficient</li>
                                            <li class="list-group-item">BMI and Length of Stay: {bmi_los_corr:.2f} correlation coefficient</li>
                                            <li class="list-group-item">Strongest correlation: {df[['Age', 'BMI', 'LengthOfStay', 'TotalCost', 'InsuranceCoverage']].corr().unstack().sort_values(ascending=False).drop_duplicates().index[1][0]} and {df[['Age', 'BMI', 'LengthOfStay', 'TotalCost', 'InsuranceCoverage']].corr().unstack().sort_values(ascending=False).drop_duplicates().index[1][1]} ({df[['Age', 'BMI', 'LengthOfStay', 'TotalCost', 'InsuranceCoverage']].corr().unstack().sort_values(ascending=False).drop_duplicates().iloc[1]:.2f})</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
    """
    
    # Readmissions Tab
    html_content += """
                        <div class="tab-pane fade" id="readmissions" role="tabpanel" aria-labelledby="readmissions-tab">
                            <h3>Readmission Analysis</h3>
                            <p>Analysis of patient readmissions and factors affecting readmission rates.</p>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Readmission Rates by Diagnosis</h4>
    """
    
    # Add readmission by diagnosis chart
    readmission_diag_img = get_image_base64('visualizations/static/readmission_by_diagnosis.png')
    if readmission_diag_img:
        html_content += f'<img src="data:image/png;base64,{readmission_diag_img}" class="img-fluid" alt="Readmission Rates by Diagnosis">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/readmission_by_diagnosis.html')
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Readmission by Age Group</h4>
    """
    
    # Add readmission by age chart
    readmission_age_img = get_image_base64('visualizations/static/readmission_by_age.png')
    if readmission_age_img:
        html_content += f'<img src="data:image/png;base64,{readmission_age_img}" class="img-fluid" alt="Readmission by Age Group">'
    else:
        html_content += embed_interactive_chart('visualizations/interactive/readmission_analysis.html', height=500)
    
    html_content += """
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="chart-container">
                                        <h4>Readmission by Outcome</h4>
    """
    
    # Add readmission by outcome chart
    readmission_outcome_img = get_image_base64('visualizations/static/readmission_by_outcome.png')
    if readmission_outcome_img:
        html_content += f'<img src="data:image/png;base64,{readmission_outcome_img}" class="img-fluid" alt="Readmission by Outcome">'
    else:
        html_content += '<p>Chart not available</p>'
    
    html_content += """
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="chart-container">
                                        <h4>Key Readmission Findings</h4>
                                        <ul class="list-group">
                                            <li class="list-group-item">Overall readmission rate: {stats['readmission_rate']}</li>
                                            <li class="list-group-item">Diagnosis with highest readmission rate: {df.groupby('PrimaryDiagnosis')['Readmitted'].mean().sort_values(ascending=False).index[0]} ({df.groupby('PrimaryDiagnosis')['Readmitted'].mean().sort_values(ascending=False).iloc[0]*100:.1f}%)</li>
                                            <li class="list-group-item">Age group with highest readmission rate: {df.groupby('AgeGroup')['Readmitted'].mean().sort_values(ascending=False).index[0]} ({df.groupby('AgeGroup')['Readmitted'].mean().sort_values(ascending=False).iloc[0]*100:.1f}%)</li>
                                            <li class="list-group-item">Outcome with highest readmission rate: {df.groupby('Outcome')['Readmitted'].mean().sort_values(ascending=False).index[0]} ({df.groupby('Outcome')['Readmitted'].mean().sort_values(ascending=False).iloc[0]*100:.1f}%)</li>
                                            <li class="list-group-item">Treatment with highest readmission rate: {df.groupby('Treatment')['Readmitted'].mean().sort_values(ascending=False).index[0]} ({df.groupby('Treatment')['Readmitted'].mean().sort_values(ascending=False).iloc[0]*100:.1f}%)</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Conclusions -->
            <div class="row">
                <div class="col-12">
                    <h2 class="section-title">Conclusions and Recommendations</h2>
                    <div class="chart-container">
                        <h4>Key Conclusions</h4>
                        <p>Based on the comprehensive analysis of hospital patient data, the following conclusions can be drawn:</p>
                        <ol>
                            <li>Patient demographics show a balanced distribution across age groups and genders, with a slight predominance of female patients.</li>
                            <li>The most common diagnoses include hypertension, type 2 diabetes, and back pain, which align with national health trends.</li>
                            <li>Emergency admissions constitute the largest portion of hospital visits, highlighting the importance of emergency department resources.</li>
                            <li>There is a strong positive correlation between length of stay and total cost, emphasizing the financial impact of extended hospitalizations.</li>
                            <li>Older patients (65+) generally have longer hospital stays, higher costs, and higher readmission rates compared to younger patients.</li>
                            <li>Insurance coverage varies significantly by type, with Medicare providing the highest average coverage rate.</li>
                            <li>Certain diagnoses show notably higher readmission rates, suggesting potential areas for improved follow-up care.</li>
                        </ol>
                        
                        <h4>Recommendations</h4>
                        <p>Based on the findings, the following recommendations are proposed:</p>
                        <ol>
                            <li><strong>Targeted Preventive Care:</strong> Implement targeted preventive care programs for the most common diagnoses to reduce hospital admissions.</li>
                            <li><strong>Length of Stay Optimization:</strong> Develop protocols to optimize length of stay for common diagnoses without compromising care quality.</li>
                            <li><strong>Readmission Reduction:</strong> Establish enhanced follow-up procedures for diagnoses with high readmission rates.</li>
                            <li><strong>Cost Management:</strong> Review treatment protocols for high-cost diagnoses to identify potential cost-saving opportunities.</li>
                            <li><strong>Insurance Coordination:</strong> Improve coordination with insurance providers to maximize coverage and minimize patient financial burden.</li>
                            <li><strong>Seasonal Planning:</strong> Adjust staffing and resource allocation based on identified seasonal admission patterns.</li>
                            <li><strong>Age-Specific Care:</strong> Develop specialized care protocols for elderly patients to address their unique needs and reduce complications.</li>
                        </ol>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="text-center">
            <div class="container">
                <p>Hospital Patient Data Analysis Dashboard</p>
                <p>Created for Data Science Project</p>
                <p>Report generated on {current_date}</p>
            </div>
        </footer>
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open('dashboard/hospital_patient_analysis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Copy interactive charts to the dashboard directory if they exist
    if os.path.exists('visualizations/interactive'):
        import shutil
        if not os.path.exists('dashboard/interactive'):
            os.makedirs('dashboard/interactive')
        
        for file in os.listdir('visualizations/interactive'):
            if file.endswith('.html'):
                shutil.copy2(os.path.join('visualizations/interactive', file), os.path.join('dashboard/interactive', file))
    
    print(f"Dashboard created successfully and saved to 'dashboard/hospital_patient_analysis.html'")

# Execute the dashboard creation
if __name__ == "__main__":
    create_dashboard()
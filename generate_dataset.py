import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the number of patients
num_patients = 1000

# Generate patient IDs
patient_ids = [f'P{i:04d}' for i in range(1, num_patients + 1)]

# Generate patient demographics
def generate_demographics(num_patients):
    # Age distribution (between 0 and 100 with more weight to middle ages)
    ages = np.random.normal(45, 20, num_patients).astype(int)
    ages = np.clip(ages, 0, 100)  # Clip ages to be between 0 and 100
    
    # Gender
    genders = np.random.choice(['Male', 'Female'], size=num_patients, p=[0.48, 0.52])
    
    # Blood types
    blood_types = np.random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'], 
                                  size=num_patients, 
                                  p=[0.34, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.07])
    
    # BMI (normal distribution around 26 with std of 5)
    bmis = np.random.normal(26, 5, num_patients)
    bmis = np.clip(bmis, 15, 45)  # Clip BMIs to realistic range
    bmis = np.round(bmis, 1)  # Round to 1 decimal place
    
    return ages, genders, blood_types, bmis

# Generate admission details
def generate_admission_details(num_patients, ages):
    # Current date for reference
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)  # 2 years of data
    
    # Random admission dates within the last 2 years
    admission_timestamps = [start_date + timedelta(days=random.randint(0, 365*2)) for _ in range(num_patients)]
    admission_dates = [ts.strftime('%Y-%m-%d') for ts in admission_timestamps]
    
    # Length of stay (correlated with age - older patients tend to stay longer)
    los_base = np.random.exponential(5, num_patients).astype(int) + 1  # Base length of stay
    age_factor = ages / 50  # Age factor (1.0 at age 50)
    length_of_stay = (los_base * age_factor).astype(int)
    length_of_stay = np.clip(length_of_stay, 1, 60)  # Clip to realistic range (1-60 days)
    
    # Admission types
    admission_types = np.random.choice(
        ['Emergency', 'Elective', 'Urgent', 'Observation', 'Maternity'],
        size=num_patients,
        p=[0.45, 0.30, 0.15, 0.05, 0.05]
    )
    
    # Calculate discharge dates
    discharge_dates = []
    for i in range(num_patients):
        discharge_date = admission_timestamps[i] + timedelta(days=int(length_of_stay[i]))
        discharge_dates.append(discharge_date.strftime('%Y-%m-%d'))
    
    return admission_dates, length_of_stay, admission_types, discharge_dates

# Generate medical conditions and diagnoses
def generate_diagnoses(num_patients, ages, genders):
    # Common diagnoses with ICD-10 codes
    common_diagnoses = [
        ('I10', 'Hypertension'),
        ('E11', 'Type 2 Diabetes'),
        ('J44', 'COPD'),
        ('I25', 'Coronary Artery Disease'),
        ('M54', 'Back Pain'),
        ('J45', 'Asthma'),
        ('F32', 'Depression'),
        ('M17', 'Osteoarthritis'),
        ('K21', 'GERD'),
        ('G47', 'Sleep Disorder'),
        ('N39', 'Urinary Tract Infection'),
        ('J01', 'Sinusitis'),
        ('L20', 'Atopic Dermatitis'),
        ('R07', 'Chest Pain'),
        ('R10', 'Abdominal Pain'),
        ('A09', 'Gastroenteritis'),
        ('J18', 'Pneumonia'),
        ('S06', 'Head Injury'),
        ('R50', 'Fever'),
        ('O80', 'Normal Delivery')
    ]
    
    # Assign primary diagnoses with age and gender considerations
    primary_diagnoses = []
    primary_diagnosis_codes = []
    
    for i in range(num_patients):
        age = ages[i]
        gender = genders[i]
        
        # Adjust probabilities based on age and gender
        weights = np.ones(len(common_diagnoses))
        
        # Age-related adjustments
        if age > 60:
            # Increase probability of chronic conditions for older patients
            for j, (code, name) in enumerate(common_diagnoses):
                if code in ['I10', 'E11', 'I25', 'M17']:
                    weights[j] *= 3.0
        elif age < 18:
            # Increase probability of certain conditions for younger patients
            for j, (code, name) in enumerate(common_diagnoses):
                if code in ['J45', 'L20', 'R50']:
                    weights[j] *= 2.5
                if code in ['I10', 'E11', 'I25', 'M17']:  # Reduce chronic adult conditions
                    weights[j] *= 0.2
        
        # Gender-related adjustments
        if gender == 'Female':
            # Increase probability of female-specific conditions
            for j, (code, name) in enumerate(common_diagnoses):
                if code in ['N39', 'O80']:
                    weights[j] *= 3.0
                    
            # Only females can have normal delivery
            if age < 15 or age > 50:  # Adjust for reproductive age
                for j, (code, name) in enumerate(common_diagnoses):
                    if code == 'O80':
                        weights[j] = 0.0
        else:  # Male
            # Males cannot have normal delivery
            for j, (code, name) in enumerate(common_diagnoses):
                if code == 'O80':
                    weights[j] = 0.0
        
        # Normalize weights to probabilities
        weights = weights / np.sum(weights)
        
        # Select diagnosis
        diagnosis_idx = np.random.choice(range(len(common_diagnoses)), p=weights)
        code, name = common_diagnoses[diagnosis_idx]
        
        primary_diagnosis_codes.append(code)
        primary_diagnoses.append(name)
    
    # Generate secondary diagnoses (some patients have comorbidities)
    has_secondary = np.random.binomial(1, 0.4, num_patients).astype(bool)  # 40% have secondary diagnosis
    secondary_diagnoses = []
    secondary_diagnosis_codes = []
    
    for i in range(num_patients):
        if has_secondary[i]:
            # Exclude the primary diagnosis from selection
            available_diagnoses = [d for d in common_diagnoses if d[0] != primary_diagnosis_codes[i]]
            diagnosis_idx = np.random.choice(range(len(available_diagnoses)))
            code, name = available_diagnoses[diagnosis_idx]
            secondary_diagnosis_codes.append(code)
            secondary_diagnoses.append(name)
        else:
            secondary_diagnosis_codes.append('')
            secondary_diagnoses.append('')
    
    return primary_diagnosis_codes, primary_diagnoses, secondary_diagnosis_codes, secondary_diagnoses

# Generate treatment and outcome data
def generate_treatments_and_outcomes(num_patients, primary_diagnoses, ages, length_of_stay):
    # Treatment types
    treatments = [
        'Medication',
        'Surgery',
        'Physical Therapy',
        'Radiation Therapy',
        'Chemotherapy',
        'Counseling',
        'Respiratory Therapy',
        'IV Fluids',
        'Pain Management',
        'Observation'
    ]
    
    # Assign treatments based on diagnosis and age
    assigned_treatments = []
    for i in range(num_patients):
        diagnosis = primary_diagnoses[i]
        age = ages[i]
        
        # Adjust probabilities based on diagnosis
        weights = np.ones(len(treatments))
        
        if 'Hypertension' in diagnosis or 'Diabetes' in diagnosis:
            for j, t in enumerate(treatments):
                if t == 'Medication':
                    weights[j] *= 5.0
                if t in ['Surgery', 'Radiation Therapy', 'Chemotherapy']:
                    weights[j] *= 0.2
        
        elif 'Coronary Artery Disease' in diagnosis or 'COPD' in diagnosis:
            for j, t in enumerate(treatments):
                if t in ['Medication', 'Respiratory Therapy']:
                    weights[j] *= 3.0
                if t == 'Surgery':
                    weights[j] *= 1.5
        
        elif 'Osteoarthritis' in diagnosis or 'Back Pain' in diagnosis:
            for j, t in enumerate(treatments):
                if t in ['Physical Therapy', 'Pain Management']:
                    weights[j] *= 4.0
                if t == 'Surgery' and age > 50:
                    weights[j] *= 2.0
        
        elif 'Depression' in diagnosis:
            for j, t in enumerate(treatments):
                if t in ['Counseling', 'Medication']:
                    weights[j] *= 5.0
                if t in ['Surgery', 'Radiation Therapy', 'Chemotherapy']:
                    weights[j] *= 0.1
        
        elif 'Pneumonia' in diagnosis:
            for j, t in enumerate(treatments):
                if t in ['Medication', 'Respiratory Therapy', 'IV Fluids']:
                    weights[j] *= 4.0
        
        # Normalize weights to probabilities
        weights = weights / np.sum(weights)
        
        # Select treatment
        treatment_idx = np.random.choice(range(len(treatments)), p=weights)
        assigned_treatments.append(treatments[treatment_idx])
    
    # Generate costs (correlated with length of stay and treatment type)
    base_costs = np.random.lognormal(mean=8.5, sigma=0.4, size=num_patients)  # Base daily cost
    
    # Adjust costs based on treatment
    treatment_multipliers = {
        'Medication': 1.0,
        'Surgery': 3.0,
        'Physical Therapy': 1.2,
        'Radiation Therapy': 2.5,
        'Chemotherapy': 2.8,
        'Counseling': 0.8,
        'Respiratory Therapy': 1.3,
        'IV Fluids': 1.1,
        'Pain Management': 1.2,
        'Observation': 0.7
    }
    
    costs = []
    for i in range(num_patients):
        treatment = assigned_treatments[i]
        los = length_of_stay[i]
        
        # Calculate total cost
        multiplier = treatment_multipliers.get(treatment, 1.0)
        daily_cost = base_costs[i]
        total_cost = daily_cost * los * multiplier
        
        # Add some random variation
        total_cost *= np.random.uniform(0.9, 1.1)
        
        costs.append(round(total_cost, 2))
    
    # Generate outcomes
    outcomes = np.random.choice(
        ['Recovered', 'Improved', 'Stable', 'Deteriorated', 'Deceased'],
        size=num_patients,
        p=[0.50, 0.30, 0.12, 0.05, 0.03]
    )
    
    # Adjust outcomes based on age, length of stay, and treatment
    for i in range(num_patients):
        age = ages[i]
        los = length_of_stay[i]
        
        # Elderly patients with long stays have higher chance of poor outcomes
        if age > 75 and los > 14 and random.random() < 0.4:
            outcomes[i] = np.random.choice(['Deteriorated', 'Deceased'], p=[0.7, 0.3])
        
        # Young patients with short stays have higher chance of good outcomes
        if age < 30 and los < 5 and random.random() < 0.7:
            outcomes[i] = np.random.choice(['Recovered', 'Improved'], p=[0.8, 0.2])
    
    # Generate readmission flags (correlated with outcome)
    readmission_probs = {
        'Recovered': 0.05,
        'Improved': 0.15,
        'Stable': 0.25,
        'Deteriorated': 0.40,
        'Deceased': 0.0  # No readmission if deceased
    }
    
    readmitted = []
    for i in range(num_patients):
        outcome = outcomes[i]
        prob = readmission_probs.get(outcome, 0.1)
        
        # Adjust for age
        if ages[i] > 70:
            prob *= 1.5
        
        readmitted.append(np.random.random() < prob)
    
    return assigned_treatments, costs, outcomes, readmitted

# Generate insurance and payment data
def generate_insurance_data(num_patients, ages, costs):
    # Insurance types
    insurance_types = np.random.choice(
        ['Medicare', 'Medicaid', 'Private', 'Self-Pay', 'Other'],
        size=num_patients,
        p=[0.30, 0.20, 0.35, 0.10, 0.05]
    )
    
    # Adjust insurance based on age
    for i in range(num_patients):
        if ages[i] >= 65 and random.random() < 0.8:
            insurance_types[i] = 'Medicare'
    
    # Calculate insurance coverage percentage
    coverage_pcts = []
    for i in range(num_patients):
        insurance = insurance_types[i]
        
        if insurance == 'Medicare':
            base_coverage = 0.80
        elif insurance == 'Medicaid':
            base_coverage = 0.90
        elif insurance == 'Private':
            base_coverage = np.random.uniform(0.70, 0.95)
        elif insurance == 'Self-Pay':
            base_coverage = 0.0
        else:  # Other
            base_coverage = np.random.uniform(0.50, 0.80)
        
        # Add some random variation
        coverage = base_coverage * np.random.uniform(0.95, 1.05)
        coverage = min(coverage, 1.0)  # Cap at 100%
        
        coverage_pcts.append(round(coverage, 2))
    
    # Calculate patient payment and insurance payment
    patient_payments = []
    insurance_payments = []
    
    for i in range(num_patients):
        total_cost = costs[i]
        coverage = coverage_pcts[i]
        
        insurance_payment = total_cost * coverage
        patient_payment = total_cost - insurance_payment
        
        patient_payments.append(round(patient_payment, 2))
        insurance_payments.append(round(insurance_payment, 2))
    
    return insurance_types, coverage_pcts, patient_payments, insurance_payments

# Generate all data
ages, genders, blood_types, bmis = generate_demographics(num_patients)
admission_dates, length_of_stay, admission_types, discharge_dates = generate_admission_details(num_patients, ages)
primary_diagnosis_codes, primary_diagnoses, secondary_diagnosis_codes, secondary_diagnoses = generate_diagnoses(num_patients, ages, genders)
treatments, costs, outcomes, readmitted = generate_treatments_and_outcomes(num_patients, primary_diagnoses, ages, length_of_stay)
insurance_types, coverage_pcts, patient_payments, insurance_payments = generate_insurance_data(num_patients, ages, costs)

# Create the DataFrame
patient_data = pd.DataFrame({
    'PatientID': patient_ids,
    'Age': ages,
    'Gender': genders,
    'BloodType': blood_types,
    'BMI': bmis,
    'AdmissionDate': admission_dates,
    'DischargeDate': discharge_dates,
    'LengthOfStay': length_of_stay,
    'AdmissionType': admission_types,
    'PrimaryDiagnosisCode': primary_diagnosis_codes,
    'PrimaryDiagnosis': primary_diagnoses,
    'SecondaryDiagnosisCode': secondary_diagnosis_codes,
    'SecondaryDiagnosis': secondary_diagnoses,
    'Treatment': treatments,
    'TotalCost': costs,
    'InsuranceType': insurance_types,
    'InsuranceCoverage': coverage_pcts,
    'PatientPayment': patient_payments,
    'InsurancePayment': insurance_payments,
    'Outcome': outcomes,
    'Readmitted': readmitted
})

# Save to CSV
patient_data.to_csv('hospital_patient_data.csv', index=False)

print(f"Dataset created with {num_patients} patient records and saved to 'hospital_patient_data.csv'")
print("\nDataset Preview:")
print(patient_data.head())

print("\nDataset Summary:")
print(patient_data.describe(include='all').T)
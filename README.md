# Hospital Patient Dataset Analysis Project

This project provides a comprehensive analysis of hospital patient data, including demographics, diagnoses, treatments, outcomes, and financial aspects. It generates synthetic data, performs exploratory data analysis, creates visualizations, presents findings in an interactive dashboard, and includes statistical analysis with predictive modeling for patient outcomes.

## Project Structure

```
├── generate_dataset.py        # Creates synthetic hospital patient data
├── exploratory_data_analysis.py  # Performs EDA on the dataset
├── visualizations.py          # Generates static and interactive visualizations
├── create_dashboard.py        # Creates an HTML dashboard with insights
├── statistical_analysis.py    # Performs statistical tests and builds predictive models
├── main.py                    # Main script to run all components
├── hospital_patient_data.csv  # Generated dataset
├── eda_outputs/               # Exploratory data analysis outputs
├── visualizations/            # Generated visualizations
│   ├── static/                # Static image visualizations
│   └── interactive/           # Interactive HTML visualizations
├── dashboard/                 # Dashboard files
│   └── hospital_patient_analysis.html  # Main dashboard file
└── statistical_analysis/      # Statistical analysis outputs
    └── statistical_analysis_report.md  # Statistical analysis report
```

## Features

1. **Data Generation**: Creates a synthetic dataset with realistic hospital patient data including:
   - Patient demographics (age, gender, blood type, BMI)
   - Admission details (date, type, length of stay)
   - Diagnoses (primary and secondary)
   - Treatments and procedures
   - Outcomes and readmission data
   - Financial information (costs, insurance)

2. **Exploratory Data Analysis**:
   - Data cleaning and preprocessing
   - Statistical summaries
   - Distribution analysis
   - Correlation analysis

3. **Visualizations**:
   - Demographic visualizations (age/gender distribution, BMI distribution)
   - Diagnosis and admission patterns
   - Treatment effectiveness and outcomes
   - Financial analysis (costs, insurance coverage)
   - Correlation analysis
   - Readmission analysis

4. **Interactive Dashboard**:
   - Executive summary with key statistics
   - Insights and findings
   - Tabbed sections for different analysis categories
   - Conclusions and recommendations

5. **Statistical Analysis**:
   - Hypothesis testing (t-tests, ANOVA, chi-square, correlation tests)
   - Statistical visualizations
   - Significance testing and interpretation
   - Relationship analysis between variables

6. **Predictive Modeling**:
   - Readmission prediction model
   - Length of stay prediction model
   - Mortality prediction model
   - Feature importance analysis
   - Model performance evaluation

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - scipy

## How to Run

1. Install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn plotly scipy
   ```

2. Run the main script to execute all components:
   ```
   python main.py
   ```

   This will:
   - Generate the dataset (if it doesn't exist)
   - Perform exploratory data analysis
   - Create visualizations
   - Build the dashboard

3. Open the dashboard in your web browser:
   ```
   dashboard/hospital_patient_analysis.html
   ```

## Individual Components

You can also run each component separately:

- Generate the dataset:
  ```
  python generate_dataset.py
  ```

- Perform exploratory data analysis:
  ```
  python exploratory_data_analysis.py
  ```

- Create visualizations:
  ```
  python visualizations.py
  ```

- Build the dashboard:
  ```
  python create_dashboard.py
  ```

## Project Insights

The dashboard provides various insights including:

- Patient demographic patterns
- Common diagnoses and their characteristics
- Treatment effectiveness analysis
- Financial patterns and insurance coverage
- Factors affecting readmission rates
- Correlations between patient attributes and outcomes

## Notes

- This project uses synthetic data generated with realistic distributions
- The dashboard is a static HTML file that can be viewed in any modern web browser
- All visualizations are saved in both static (PNG) and interactive (HTML) formats where applicable
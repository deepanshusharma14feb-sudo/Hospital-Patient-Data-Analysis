import os
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and display its output"""
    print(f"\n{'=' * 80}")
    print(f"Running: {description}")
    print(f"{'=' * 80}\n")
    
    try:
        # Run the script and capture its output
        process = subprocess.Popen(['python', script_name], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Check for errors
        return_code = process.poll()
        if return_code != 0:
            error_output = process.stderr.read()
            print(f"Error running {script_name}:\n{error_output}")
            return False
        return True
    except Exception as e:
        print(f"Exception running {script_name}: {e}")
        return False

def main():
    """Run all components of the hospital patient dataset analysis project"""
    print("\n" + "*" * 100)
    print("*" + " " * 98 + "*")
    print("*" + " " * 30 + "HOSPITAL PATIENT DATASET ANALYSIS PROJECT" + " " * 30 + "*")
    print("*" + " " * 98 + "*")
    print("*" * 100 + "\n")
    
    # Create directories if they don't exist
    for directory in ['eda_outputs', 'visualizations/static', 'visualizations/interactive', 'dashboard', 'statistical_analysis']:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Generate the dataset
    if not os.path.exists('hospital_patient_data.csv'):
        success = run_script('generate_dataset.py', 'Generating Hospital Patient Dataset')
        if not success:
            print("Failed to generate dataset. Exiting.")
            return
    else:
        print("Dataset already exists. Skipping generation step.")
    
    # Step 2: Perform exploratory data analysis
    success = run_script('exploratory_data_analysis.py', 'Performing Exploratory Data Analysis')
    if not success:
        print("Failed to complete exploratory data analysis. Exiting.")
        return
    
    # Step 3: Generate visualizations
    success = run_script('visualizations.py', 'Generating Visualizations')
    if not success:
        print("Failed to generate visualizations. Exiting.")
        return
    
    # Step 4: Create dashboard
    success = run_script('create_dashboard.py', 'Creating Dashboard')
    if not success:
        print("Failed to create dashboard. Exiting.")
        return
    
    # Step 5: Perform statistical analysis and predictive modeling
    success = run_script('statistical_analysis.py', 'Performing Statistical Analysis and Predictive Modeling')
    if not success:
        print("Failed to complete statistical analysis. Exiting.")
        return
    
    # Final message
    print("\n" + "*" * 100)
    print("Project execution completed successfully!")
    print("\nYou can view the dashboard by opening the following file in your web browser:")
    print(f"  {os.path.abspath('dashboard/hospital_patient_analysis.html')}")
    print("\nYou can also view the statistical analysis report at:")
    print(f"  {os.path.abspath('statistical_analysis/statistical_analysis_report.md')}")
    print("*" * 100 + "\n")

if __name__ == "__main__":
    main()
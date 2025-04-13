import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime
import re

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from script_dir)
project_root = os.path.dirname(script_dir)

# File paths
ENHANCED_DATA_PATH = os.path.join(script_dir, 'enhanced_data.xlsx')
REPORT_PATH = os.path.join(script_dir, 'comprehensive_data_cleaning_report.txt')

# Update this line to use the Mac file path
excel_path = '/Users/dameesh/Desktop/MinMUDAC2025/Data/Novice.xlsx'
# Or use this more flexible approach:
# excel_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'MinMUDAC2025', 'Data', 'Novice.xlsx')

def clean_column_name(col_name):
    """Clean column name by removing special characters and replacing spaces with underscores"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

def get_county_from_block_group(block_group):
    """Extract county name from Census Block Group ID."""
    # Census Block Group IDs are 12 digits:
    # First 2 digits: State (27 for Minnesota)
    # Next 3 digits: County FIPS code
    # Remaining 7 digits: Tract and Block Group
    
    try:
        block_group_str = str(int(block_group))  # Convert to int first to remove any decimal places
        if len(block_group_str) == 12:
            state_fips = block_group_str[:2]
            county_fips = block_group_str[2:5]
            
            # Only process Minnesota block groups
            if state_fips == "27":
                county_names = {
                    '001': 'Aitkin', '003': 'Anoka', '005': 'Becker', '007': 'Beltrami', 
                    '009': 'Benton', '011': 'Big Stone', '013': 'Blue Earth', '015': 'Brown', 
                    '017': 'Carlton', '019': 'Carver', '021': 'Cass', '023': 'Chippewa', 
                    '025': 'Chisago', '027': 'Clay', '029': 'Clearwater', '031': 'Cook', 
                    '033': 'Cottonwood', '035': 'Crow Wing', '037': 'Dakota', '039': 'Dodge', 
                    '041': 'Douglas', '043': 'Faribault', '045': 'Fillmore', '047': 'Freeborn', 
                    '049': 'Goodhue', '051': 'Grant', '053': 'Hennepin', '055': 'Houston', 
                    '057': 'Hubbard', '059': 'Isanti', '061': 'Itasca', '063': 'Jackson', 
                    '065': 'Kanabec', '067': 'Kandiyohi', '069': 'Kittson', '071': 'Koochiching', 
                    '073': 'Lac qui Parle', '075': 'Lake', '077': 'Lake of the Woods', '079': 'Le Sueur', 
                    '081': 'Lincoln', '083': 'Lyon', '085': 'McLeod', '087': 'Mahnomen', 
                    '089': 'Marshall', '091': 'Martin', '093': 'Meeker', '095': 'Mille Lacs', 
                    '097': 'Morrison', '099': 'Mower', '101': 'Murray', '103': 'Nicollet', 
                    '105': 'Nobles', '107': 'Norman', '109': 'Olmsted', '111': 'Otter Tail', 
                    '113': 'Pennington', '115': 'Pine', '117': 'Pipestone', '119': 'Polk', 
                    '121': 'Pope', '123': 'Ramsey', '125': 'Red Lake', '127': 'Redwood', 
                    '129': 'Renville', '131': 'Rice', '133': 'Rock', '135': 'Roseau', 
                    '137': 'St. Louis', '139': 'Scott', '141': 'Sherburne', '143': 'Sibley', 
                    '145': 'Stearns', '147': 'Steele', '149': 'Stevens', '151': 'Swift', 
                    '153': 'Todd', '155': 'Traverse', '157': 'Wabasha', '159': 'Wadena', 
                    '161': 'Waseca', '163': 'Washington', '165': 'Watonwan', '167': 'Wilkin', 
                    '169': 'Winona', '171': 'Wright', '173': 'Yellow Medicine'
                }
                return county_names.get(county_fips)
    except (ValueError, TypeError):
        pass
    return None

def print_stats_box(title, stats):
    """Print statistics in a formatted box."""
    width = max(len(title), max(len(f"{k}: {v}") for k, v in stats.items())) + 4
    print("+" + "=" * width + "+")
    print(f"|{title.center(width)}|")
    print("+" + "=" * width + "+")
    for key, value in stats.items():
        print(f"| {key}: {value}" + " " * (width - len(f"| {key}: {value}")) + "|")
    print("+" + "=" * width + "+")

def process_age_features(df, report_file=None):
    """Process age-related features from birthdates"""
    if report_file:
        report_file.write("\nAge Feature Processing:\n")
        report_file.write("-" * 50 + "\n")
    
    # Track initial missing values for birthdate columns
    initial_missing = {
        'Big_Birthdate': df['Big Birthdate'].isna().sum(),
        'Little_Birthdate': df['Little Birthdate'].isna().sum()
    }
    
    # Convert birthdates to datetime
    df['Big Birthdate'] = pd.to_datetime(df['Big Birthdate'], errors='coerce')
    df['Little Birthdate'] = pd.to_datetime(df['Little Birthdate'], errors='coerce')
    
    # Calculate ages
    today = pd.Timestamp.now()
    df['Big_Age'] = (today - df['Big Birthdate']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['Little_Age'] = (today - df['Little Birthdate']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    
    # Create age groups
    df['Big_Age_Group'] = pd.cut(df['Big_Age'], 
                                bins=[0, 18, 25, 35, 45, 55, 100],
                                labels=['Under 18', '18-25', '26-35', '36-45', '46-55', 'Over 55'])
    
    df['Little_Age_Group'] = pd.cut(df['Little_Age'],
                                   bins=[0, 5, 10, 15, 18],
                                   labels=['Under 5', '5-10', '11-15', '16-18'])
    
    # Calculate age compatibility score
    # Higher score means better age compatibility
    df['Age_Compatibility_Score'] = 0
    
    # Ideal age difference is 10-15 years
    age_diff = df['Big_Age'] - df['Little_Age']
    df.loc[age_diff.between(10, 15), 'Age_Compatibility_Score'] = 3
    df.loc[age_diff.between(8, 9), 'Age_Compatibility_Score'] = 2
    df.loc[age_diff.between(16, 20), 'Age_Compatibility_Score'] = 2
    df.loc[age_diff.between(6, 7), 'Age_Compatibility_Score'] = 1
    df.loc[age_diff.between(21, 25), 'Age_Compatibility_Score'] = 1
    
    # Calculate final missing values for age columns
    final_missing = {
        'Big_Age': df['Big_Age'].isna().sum(),
        'Little_Age': df['Little_Age'].isna().sum()
    }
    
    # Report improvements
    if report_file:
        report_file.write("\nAge Feature Improvements:\n")
        report_file.write(f"Big_Birthdate: {initial_missing['Big_Birthdate']} missing values\n")
        report_file.write(f"Little_Birthdate: {initial_missing['Little_Birthdate']} missing values\n")
        report_file.write(f"Big_Age: {final_missing['Big_Age']} missing values after processing\n")
        report_file.write(f"Little_Age: {final_missing['Little_Age']} missing values after processing\n")
        
        report_file.write("\nNew Age Features Created:\n")
        report_file.write("  - Big_Age: Calculated age of Big\n")
        report_file.write("  - Little_Age: Calculated age of Little\n")
        report_file.write("  - Big_Age_Group: Age group category for Big\n")
        report_file.write("  - Little_Age_Group: Age group category for Little\n")
        report_file.write("  - Age_Compatibility_Score: Score indicating age compatibility\n")
    
    return df

def fill_missing_geographic_data(df, report_file=None):
    """Fill missing geographic data using Census Block Group relationships"""
    if report_file:
        report_file.write("\nGeographic Data Imputation:\n")
        report_file.write("-" * 50 + "\n")
    
    # Track initial missing values
    initial_missing = {
        'Big_County': df['Big County'].isna().sum(),
        'Big_Race_Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little_Race_Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
    }
    
    # Create Census Block Group mapping
    block_group_to_county = {}
    for idx, row in df.iterrows():
        if pd.notna(row['Big Home Census Block Group']):
            block_group_to_county[row['Big Home Census Block Group']] = row['Big County']
        if pd.notna(row['Big Employer/School Census Block Group']):
            block_group_to_county[row['Big Employer/School Census Block Group']] = row['Big County']
        if pd.notna(row['Little Mailing Address Census Block Group']):
            block_group_to_county[row['Little Mailing Address Census Block Group']] = row['Big County']
    
    # Fill missing county data using block group relationships
    for idx, row in df.iterrows():
        if pd.isna(row['Big County']):
            # Try to find county from any available block group
            block_groups = [
                row['Big Home Census Block Group'],
                row['Big Employer/School Census Block Group'],
                row['Little Mailing Address Census Block Group']
            ]
            
            for bg in block_groups:
                if pd.notna(bg) and bg in block_group_to_county:
                    df.at[idx, 'Big County'] = block_group_to_county[bg]
                    break
    
    # Create and fill Little County column
    df['Little County'] = None
    for idx, row in df.iterrows():
        if pd.notna(row['Little Mailing Address Census Block Group']):
            county = get_county_from_block_group(row['Little Mailing Address Census Block Group'])
            if county:
                df.at[idx, 'Little County'] = county
    
    # Fill race/ethnicity data with confidence threshold
    confidence_threshold = 0.8
    
    # Create race/ethnicity mapping based on existing data
    race_mapping = {}
    for idx, row in df.iterrows():
        if pd.notna(row['Big Race/Ethnicity']) and pd.notna(row['Little Participant: Race/Ethnicity']):
            key = (row['Big Race/Ethnicity'], row['Little Participant: Race/Ethnicity'])
            race_mapping[key] = race_mapping.get(key, 0) + 1
    
    # Normalize race mapping probabilities
    total_matches = sum(race_mapping.values())
    race_mapping = {k: v/total_matches for k, v in race_mapping.items()}
    
    # Fill missing race/ethnicity data
    for idx, row in df.iterrows():
        if pd.isna(row['Big Race/Ethnicity']) or pd.isna(row['Little Participant: Race/Ethnicity']):
            # Find most likely race/ethnicity combination
            max_prob = 0
            best_match = None
            
            for (big_race, little_race), prob in race_mapping.items():
                if prob > max_prob and prob >= confidence_threshold:
                    max_prob = prob
                    best_match = (big_race, little_race)
            
            if best_match:
                if pd.isna(row['Big Race/Ethnicity']):
                    df.at[idx, 'Big Race/Ethnicity'] = best_match[0]
                if pd.isna(row['Little Participant: Race/Ethnicity']):
                    df.at[idx, 'Little Participant: Race/Ethnicity'] = best_match[1]
    
    # Calculate final missing values
    final_missing = {
        'Big_County': df['Big County'].isna().sum(),
        'Big_Race_Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little_Race_Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum(),
        'Little_County': df['Little County'].isna().sum()
    }
    
    # Report improvements
    if report_file:
        report_file.write("\nGeographic Data Improvements:\n")
        for field in initial_missing:
            improvement = initial_missing[field] - final_missing[field]
            report_file.write(f"{field}: {improvement} missing values filled\n")
        report_file.write(f"Little_County: {final_missing['Little_County']} missing values\n")
    
    return df

def fill_missing_occupation_data(df, report_file=None):
    """Fill missing occupation data based on employer and education correlations"""
    if report_file:
        report_file.write("\nOccupation Data Processing:\n")
        report_file.write("-" * 50 + "\n")
    
    # Track initial missing values
    initial_missing = {
        'Big_Occupation': df['Big Occupation'].isna().sum(),
        'Big_Education': df['Big Level of Education'].isna().sum()
    }
    
    # Create occupation mapping based on employer
    employer_to_occupation = {}
    for idx, row in df.iterrows():
        if pd.notna(row['Big Employer']) and pd.notna(row['Big Occupation']):
            employer_to_occupation[row['Big Employer']] = row['Big Occupation']
    
    # Fill missing occupations using employer mapping
    for idx, row in df.iterrows():
        if pd.isna(row['Big Occupation']) and pd.notna(row['Big Employer']):
            if row['Big Employer'] in employer_to_occupation:
                df.at[idx, 'Big Occupation'] = employer_to_occupation[row['Big Employer']]
    
    # Create education level mapping
    education_mapping = {
        'High School': 1,
        'Some College': 2,
        'Associate Degree': 3,
        'Bachelor Degree': 4,
        'Master Degree': 5,
        'Doctorate': 6
    }
    
    # Create occupation categories
    occupation_categories = {
        'Professional': ['Engineer', 'Doctor', 'Lawyer', 'Teacher', 'Accountant', 'Manager'],
        'Technical': ['Developer', 'Analyst', 'Technician', 'Specialist'],
        'Service': ['Sales', 'Customer Service', 'Retail', 'Food Service'],
        'Skilled': ['Mechanic', 'Electrician', 'Plumber', 'Carpenter'],
        'Other': []
    }
    
    # Categorize occupations
    df['Big_Occupation_Category'] = 'Other'
    for category, keywords in occupation_categories.items():
        for keyword in keywords:
            mask = df['Big Occupation'].str.contains(keyword, case=False, na=False)
            df.loc[mask, 'Big_Occupation_Category'] = category
    
    # Create education level numeric feature
    df['Big_Education_Level'] = df['Big Level of Education'].map(education_mapping)
    
    # Calculate final missing values
    final_missing = {
        'Big_Occupation': df['Big Occupation'].isna().sum(),
        'Big_Education': df['Big Level of Education'].isna().sum()
    }
    
    # Report improvements
    if report_file:
        report_file.write("\nOccupation Data Improvements:\n")
        for field in initial_missing:
            improvement = initial_missing[field] - final_missing[field]
            report_file.write(f"{field}: {improvement} missing values filled\n")
        
        report_file.write("\nNew Occupation Features Created:\n")
        report_file.write("  - Big_Occupation_Category: Categorized occupation type\n")
        report_file.write("  - Big_Education_Level: Numeric representation of education level\n")
    
    return df

def create_demographic_compatibility_features(df, report_file=None):
    """Create demographic compatibility features"""
    if report_file:
        report_file.write("\nDemographic Compatibility Features:\n")
        report_file.write("-" * 50 + "\n")
    
    # Track initial missing values
    initial_missing = {
        'Big_Gender': df['Big Gender'].isna().sum(),
        'Little_Gender': df['Little Gender'].isna().sum(),
        'Big_Race_Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little_Race_Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
    }
    
    # Create gender compatibility score
    df['Gender_Compatibility_Score'] = 0
    same_gender_mask = df['Big Gender'] == df['Little Gender']
    df.loc[same_gender_mask, 'Gender_Compatibility_Score'] = 1
    
    # Create race/ethnicity compatibility score
    df['Race_Ethnicity_Compatibility_Score'] = 0
    same_race_mask = df['Big Race/Ethnicity'] == df['Little Participant: Race/Ethnicity']
    df.loc[same_race_mask, 'Race_Ethnicity_Compatibility_Score'] = 1
    
    # Create language compatibility score
    df['Language_Compatibility_Score'] = 0
    for idx, row in df.iterrows():
        if pd.notna(row['Big Languages']) and pd.notna(row['Little Contact: Language(s) Spoken']):
            big_languages = set(str(row['Big Languages']).lower().split(','))
            little_languages = set(str(row['Little Contact: Language(s) Spoken']).lower().split(','))
            if big_languages & little_languages:  # If there's any overlap
                df.at[idx, 'Language_Compatibility_Score'] = 1
    
    # Create overall demographic compatibility score
    df['Demographic_Compatibility_Score'] = (
        df['Gender_Compatibility_Score'] +
        df['Race_Ethnicity_Compatibility_Score'] +
        df['Language_Compatibility_Score']
    ) / 3  # Normalize to 0-1 range
    
    # Calculate final missing values
    final_missing = {
        'Big_Gender': df['Big Gender'].isna().sum(),
        'Little_Gender': df['Little Gender'].isna().sum(),
        'Big_Race_Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little_Race_Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
    }
    
    # Report improvements
    if report_file:
        report_file.write("\nDemographic Compatibility Features Created:\n")
        report_file.write("  - Gender_Compatibility_Score: Binary score for gender match\n")
        report_file.write("  - Race_Ethnicity_Compatibility_Score: Binary score for race/ethnicity match\n")
        report_file.write("  - Language_Compatibility_Score: Binary score for language match\n")
        report_file.write("  - Demographic_Compatibility_Score: Overall demographic compatibility score\n")
    
    return df

def create_process_efficiency_features(df, report_file=None):
    """Create features related to process efficiency"""
    if report_file:
        report_file.write("\nProcess Efficiency Features:\n")
        report_file.write("-" * 50 + "\n")
    
    # Track initial missing values
    initial_missing = {
        'Big_Approved_Date': df['Big Approved Date'].isna().sum(),
        'Big_Acceptance_Date': df['Big Acceptance Date'].isna().sum(),
        'Little_Acceptance_Date': df['Little Acceptance Date'].isna().sum(),
        'Match_Activation_Date': df['Match Activation Date'].isna().sum()
    }
    
    # Convert date columns to datetime
    date_columns = [
        'Big Approved Date',
        'Big Acceptance Date',
        'Little Acceptance Date',
        'Match Activation Date',
        'Big Contact: Created Date',
        'Big Enrollment: Created Date'
    ]
    
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate time-to-match metrics
    df['Days_Approval_to_Acceptance'] = (
        df['Big Acceptance Date'] - df['Big Approved Date']
    ).dt.total_seconds() / (24 * 60 * 60)
    
    df['Days_Acceptance_to_Match'] = (
        df['Match Activation Date'] - df['Big Acceptance Date']
    ).dt.total_seconds() / (24 * 60 * 60)
    
    # Create process completeness indicators
    df['Has_Approval_Date'] = df['Big Approved Date'].notna().astype(int)
    df['Has_Acceptance_Date'] = df['Big Acceptance Date'].notna().astype(int)
    df['Has_Match_Date'] = df['Match Activation Date'].notna().astype(int)
    
    # Create volunteer readiness score
    df['Volunteer_Readiness_Score'] = (
        df['Has_Approval_Date'] +
        df['Has_Acceptance_Date'] +
        df['Has_Match_Date']
    ) / 3  # Normalize to 0-1 range
    
    # Calculate application and interview timelines
    df['Days_Created_to_Approval'] = (
        df['Big Approved Date'] - df['Big Contact: Created Date']
    ).dt.total_seconds() / (24 * 60 * 60)
    
    df['Days_Enrollment_to_Approval'] = (
        df['Big Approved Date'] - df['Big Enrollment: Created Date']
    ).dt.total_seconds() / (24 * 60 * 60)
    
    # Calculate final missing values
    final_missing = {
        'Big_Approved_Date': df['Big Approved Date'].isna().sum(),
        'Big_Acceptance_Date': df['Big Acceptance Date'].isna().sum(),
        'Little_Acceptance_Date': df['Little Acceptance Date'].isna().sum(),
        'Match_Activation_Date': df['Match Activation Date'].isna().sum()
    }
    
    # Report improvements
    if report_file:
        report_file.write("\nProcess Efficiency Features Created:\n")
        report_file.write("  - Days_Approval_to_Acceptance: Time between approval and acceptance\n")
        report_file.write("  - Days_Acceptance_to_Match: Time between acceptance and match\n")
        report_file.write("  - Has_Approval_Date: Binary indicator for approval date presence\n")
        report_file.write("  - Has_Acceptance_Date: Binary indicator for acceptance date presence\n")
        report_file.write("  - Has_Match_Date: Binary indicator for match date presence\n")
        report_file.write("  - Volunteer_Readiness_Score: Overall readiness score\n")
        report_file.write("  - Days_Created_to_Approval: Time from creation to approval\n")
        report_file.write("  - Days_Enrollment_to_Approval: Time from enrollment to approval\n")
    
    return df

def enhance_important_features(df, report_file=None):
    """Enhance important features with additional derived features"""
    if report_file:
        report_file.write("\nEnhancing Important Features\n")
        report_file.write("===========================\n\n")
    
    # Track initial shape
    initial_columns = len(df.columns)
    
    # 1. Geographic Features (must come first for interactions)
    if report_file:
        report_file.write("1. Geographic Features\n")
        report_file.write("----------------------\n")
    
    # Calculate distance between Big and Little locations
    # Using Census Block Group as a proxy for location
    df['Geographic_Distance'] = abs(df['Big Home Census Block Group'] - df['Little Mailing Address Census Block Group'])
    
    # Create urban/rural classification based on Census Block Group
    # This is a simplified version - in practice, you'd want to use actual census data
    df['Urban_Rural_Score'] = df['Big Home Census Block Group'].apply(
        lambda x: 1 if x > 270000000000 else 0.5 if x > 27000000000 else 0
    )
    
    # 2. Age-Related Features
    if report_file:
        report_file.write("\n2. Age-Related Features\n")
        report_file.write("----------------------\n")
    
    # Calculate age difference
    df['Age_Difference'] = df['Big_Age'] - df['Little_Age']
    
    # Create age compatibility score (0-1)
    ideal_age_diff = 20  # Assuming ideal age difference is 20 years
    df['Age_Compatibility_Score'] = 1 - (abs(df['Age_Difference'] - ideal_age_diff) / 40)
    
    # Create age group interactions (convert categorical to string first)
    df['Age_Group_Interaction'] = df['Big_Age_Group'].astype(str) + '_' + df['Little_Age_Group'].astype(str)
    
    # 3. Match Activation Date Enhancements
    if report_file:
        report_file.write("\n3. Match Activation Date Enhancements\n")
        report_file.write("--------------------------------\n")
    
    # Convert to datetime if not already
    df['Match Activation Date'] = pd.to_datetime(df['Match Activation Date'])
    
    # Add season
    df['Activation_Season'] = df['Match Activation Date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Add day of week (0-6, where 0 is Monday)
    df['Activation_Day_of_Week'] = df['Match Activation Date'].dt.dayofweek
    
    # Add time of year score (0-1, where 0 is January 1st)
    df['Time_of_Year_Score'] = df['Match Activation Date'].dt.dayofyear / 365
    
    # Add program maturity (years since program start)
    program_start_date = df['Match Activation Date'].min()
    df['Program_Maturity_Years'] = (df['Match Activation Date'] - program_start_date).dt.days / 365
    
    # 4. Program Type Enhancements and Interactions
    if report_file:
        report_file.write("\n4. Program Type Enhancements and Interactions\n")
        report_file.write("------------------------------------------\n")
    
    # Create program type interaction with season
    df['Program_Season_Interaction'] = df['Program Type'] + '_' + df['Activation_Season']
    
    # Create interaction features with Program Type
    df['Program_Maturity_Type_Interaction'] = df['Program_Maturity_Years'] * (df['Program Type'] == 'Community').astype(int)
    df['Geographic_Type_Interaction'] = df['Geographic_Distance'] * (df['Program Type'] == 'Community').astype(int)
    df['Age_Type_Interaction'] = df['Age_Difference'] * (df['Program Type'] == 'Community').astype(int)
    
    # Report new features
    if report_file:
        report_file.write("\nNew Features Created:\n")
        report_file.write("--------------------\n")
        new_features = [
            'Geographic_Distance', 'Urban_Rural_Score',
            'Age_Difference', 'Age_Compatibility_Score', 'Age_Group_Interaction',
            'Activation_Season', 'Activation_Day_of_Week', 'Time_of_Year_Score',
            'Program_Maturity_Years', 'Program_Season_Interaction',
            'Program_Maturity_Type_Interaction', 'Geographic_Type_Interaction',
            'Age_Type_Interaction'
        ]
        for feature in new_features:
            report_file.write(f"- {feature}\n")
        
        report_file.write(f"\nTotal new features added: {len(new_features)}\n")
        report_file.write(f"Total columns in dataset: {len(df.columns)}\n")
    
    return df

def main():
    print("===== COMPREHENSIVE DATA PREPROCESSING PIPELINE =====")
    
    # Create a comprehensive report file
    with open(REPORT_PATH, 'w') as report_file:
        report_file.write("Comprehensive Data Cleaning and Imputation Report\n")
        report_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write("="*80 + "\n\n")
        
        # Step 1: Initial Data Loading
        report_file.write("Step 1: Initial Data Loading\n")
        report_file.write("==========================\n\n")
        print("\nStep 1: Initial Data Loading")
        print("==========================")
        
        # Read the Excel file using an absolute path
        print(f"Reading dataset: {excel_path}")
        report_file.write(f"Reading dataset: {excel_path}\n")
        
        df = pd.read_excel(excel_path)
        print(f"Dataset shape: {df.shape}")
        report_file.write(f"Dataset shape: {df.shape}\n\n")
        
        # Track initial missing data
        total_missing_before = df.isna().sum().sum()
        total_cells = df.size
        report_file.write(f"Initial Missing Data:\n")
        report_file.write(f"Total missing values: {total_missing_before} ({total_missing_before/total_cells*100:.2f}%)\n\n")
        
        # Step 2: Age Feature Processing
        report_file.write("\nStep 2: Age Feature Processing\n")
        report_file.write("============================\n\n")
        print("\nStep 2: Age Feature Processing")
        print("============================")
        
        # Process age features
        df = process_age_features(df, report_file)
        
        # Step 3: Geographic Data Processing
        report_file.write("\nStep 3: Geographic Data Processing\n")
        report_file.write("================================\n\n")
        print("\nStep 3: Geographic Data Processing")
        print("=================================")
        
        # Process geographic data
        df = fill_missing_geographic_data(df, report_file)
        
        # Step 4: Occupation Data Processing
        report_file.write("\nStep 4: Occupation Data Processing\n")
        report_file.write("================================\n\n")
        print("\nStep 4: Occupation Data Processing")
        print("=================================")
        
        # Process occupation data
        df = fill_missing_occupation_data(df, report_file)
        
        # Step 5: Create Demographic Compatibility Features
        report_file.write("\nStep 5: Creating Demographic Compatibility Features\n")
        report_file.write("=================================================\n\n")
        print("\nStep 5: Creating Demographic Compatibility Features")
        print("=================================================")
        
        # Create demographic compatibility features
        df = create_demographic_compatibility_features(df, report_file)
        
        # Step 6: Create Process Efficiency Features
        report_file.write("\nStep 6: Creating Process Efficiency Features\n")
        report_file.write("==========================================\n\n")
        print("\nStep 6: Creating Process Efficiency Features")
        print("==========================================")
        
        # Create process efficiency features
        df = create_process_efficiency_features(df, report_file)
        
        # Step 7: Enhance Important Features
        report_file.write("\nStep 7: Enhancing Important Features\n")
        report_file.write("==================================\n\n")
        print("\nStep 7: Enhancing Important Features")
        print("==================================")
        
        # Enhance important features
        df = enhance_important_features(df, report_file)
        
        # Final Summary
        report_file.write("\nFinal Data Quality Summary\n")
        report_file.write("========================\n\n")
        
        # Calculate final missing data
        total_missing_after = df.isna().sum().sum()
        cells_filled = total_missing_before - total_missing_after
        
        # Track missing values by column
        missing_by_column = df.isna().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0]
        
        report_file.write("Missing Values by Column:\n")
        report_file.write("----------------------\n")
        for col, missing in columns_with_missing.items():
            percentage = (missing/len(df))*100
            report_file.write(f"{col}: {missing} missing ({percentage:.2f}%)\n")
        
        report_file.write("\nOverall Statistics:\n")
        report_file.write("-----------------\n")
        report_file.write(f"Initial missing values: {total_missing_before} ({total_missing_before/total_cells*100:.2f}%)\n")
        report_file.write(f"Final missing values: {total_missing_after} ({total_missing_after/total_cells*100:.2f}%)\n")
        report_file.write(f"Total cells filled: {cells_filled} ({cells_filled/total_missing_before*100:.2f}% of initial missing)\n")
        
        # Save the processed dataset
        print(f"\nSaving processed dataset to: {ENHANCED_DATA_PATH}")
        df.to_excel(ENHANCED_DATA_PATH, index=False)
        report_file.write(f"\nProcessed dataset saved to: {ENHANCED_DATA_PATH}\n")
        
    print("\nFiles saved:")
    print(f"- Comprehensive data cleaning report: {REPORT_PATH}")
    print(f"- Processed data: {ENHANCED_DATA_PATH}")
    print("\nComprehensive data cleaning complete!")

if __name__ == "__main__":
    main() 
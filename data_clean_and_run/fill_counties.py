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
FILLED_DATA_PATH = os.path.join(script_dir, 'filled_data.xlsx')
REPORT_PATH = os.path.join(script_dir, 'data_cleaning_report.txt')

def clean_column_name(col_name):
    """Clean column name by removing special characters and replacing spaces with underscores"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

def process_age_features(df, report_file=None):
    """Process age-related features from birthdates"""
    message = "\nProcessing age features:\n"
    print(message)
    if report_file:
        report_file.write(message)
    
    # Convert birthdate columns to datetime
    date_columns = ['Big_Birthdate', 'Little_Birthdate']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate ages
    current_year = datetime.now().year
    
    # Big age calculation
    df['Big_Age'] = current_year - df['Big_Birthdate'].dt.year
    df['Big_Age'] = df['Big_Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan)
    
    # Little age calculation (using birthdate year)
    df['Little_Birthdate_year'] = df['Little_Birthdate'].dt.year
    df['Little_Birthdate_year'] = df['Little_Birthdate_year'].apply(lambda x: x if 1900 <= x <= current_year else np.nan)
    
    # Calculate age difference
    df['Age_Difference'] = df['Big_Age'] - (current_year - df['Little_Birthdate_year'])
    
    # Create age groups for Bigs
    age_bins = [0, 25, 35, 45, 55, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '55+']
    df['Big_Age_Group'] = pd.cut(df['Big_Age'], bins=age_bins, labels=age_labels)
    
    # Track missing values
    missing_stats = {
        'Big_Age': df['Big_Age'].isna().sum(),
        'Little_Birthdate_year': df['Little_Birthdate_year'].isna().sum(),
        'Age_Difference': df['Age_Difference'].isna().sum()
    }
    
    # Report missing values
    if report_file:
        report_file.write("\nAge Feature Missing Values:\n")
        for feature, count in missing_stats.items():
            report_file.write(f"- {feature}: {count} missing values\n")
    
    return df

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

def fill_missing_geographic_data(df, report_file=None):
    """Fill missing values using Census Block Group relationships"""
    message = "Performing geographic data imputation:\n"
    print(message)
    if report_file:
        report_file.write(message)
        
    # Track initial missing counts
    initial_missing = {
        'Big_County': df['Big_County'].isna().sum(),
        'Big_Race_Ethnicity': df['Big_Race_Ethnicity'].isna().sum(),
        'Little_Participant_Race_Ethnicity': df['Little_Participant_Race_Ethnicity'].isna().sum()
    }
    
    # Map Census Block Groups to counties where known
    block_to_county = {}
    for idx, row in df.dropna(subset=['Big_County', 'Big_Home_Census_Block_Group']).iterrows():
        block_to_county[row['Big_Home_Census_Block_Group']] = row['Big_County']
    
    # Fill missing counties using the mapping
    county_mask = df['Big_County'].isna() & df['Big_Home_Census_Block_Group'].notna()
    df.loc[county_mask, 'Big_County'] = df.loc[county_mask, 'Big_Home_Census_Block_Group'].map(block_to_county)
    
    # Map Census Block Groups to race/ethnicity where known
    block_to_race = {}
    for idx, row in df.dropna(subset=['Big_Race_Ethnicity', 'Big_Home_Census_Block_Group']).iterrows():
        block_to_race[row['Big_Home_Census_Block_Group']] = row['Big_Race_Ethnicity']
    
    # Fill missing race/ethnicity using the mapping (only if we have high confidence)
    race_counts = {}
    for block, races in block_to_race.items():
        if block not in race_counts:
            race_counts[block] = {}
        if races in race_counts[block]:
            race_counts[block][races] += 1
        else:
            race_counts[block][races] = 1
    
    # Only use blocks where we have a dominant race pattern
    reliable_block_to_race = {}
    for block, counts in race_counts.items():
        if len(counts) > 0:
            dominant_race = max(counts.items(), key=lambda x: x[1])[0]
            dominant_count = counts[dominant_race]
            total = sum(counts.values())
            if dominant_count / total > 0.7 and total >= 3:  # 70% threshold and at least 3 samples
                reliable_block_to_race[block] = dominant_race
    
    # Apply the reliable mapping
    race_mask = df['Big_Race_Ethnicity'].isna() & df['Big_Home_Census_Block_Group'].notna()
    df.loc[race_mask, 'Big_Race_Ethnicity'] = df.loc[race_mask, 'Big_Home_Census_Block_Group'].map(reliable_block_to_race)
    
    # Apply similar logic for Little's race/ethnicity using Little's Census Block Group
    little_block_to_race = {}
    for idx, row in df.dropna(subset=['Little_Participant_Race_Ethnicity', 'Little_Mailing_Address_Census_Block_Group']).iterrows():
        little_block_to_race[row['Little_Mailing_Address_Census_Block_Group']] = row['Little_Participant_Race_Ethnicity']
    
    # Same reliability filtering as above
    race_counts = {}
    for block, races in little_block_to_race.items():
        if block not in race_counts:
            race_counts[block] = {}
        if races in race_counts[block]:
            race_counts[block][races] += 1
        else:
            race_counts[block][races] = 1
    
    reliable_little_block_to_race = {}
    for block, counts in race_counts.items():
        if len(counts) > 0:
            dominant_race = max(counts.items(), key=lambda x: x[1])[0]
            dominant_count = counts[dominant_race]
            total = sum(counts.values())
            if dominant_count / total > 0.7 and total >= 3:
                reliable_little_block_to_race[block] = dominant_race
    
    little_race_mask = df['Little_Participant_Race_Ethnicity'].isna() & df['Little_Mailing_Address_Census_Block_Group'].notna()
    df.loc[little_race_mask, 'Little_Participant_Race_Ethnicity'] = df.loc[little_race_mask, 'Little_Mailing_Address_Census_Block_Group'].map(reliable_little_block_to_race)
    
    # Track improvements
    final_missing = {
        'Big_County': df['Big_County'].isna().sum(),
        'Big_Race_Ethnicity': df['Big_Race_Ethnicity'].isna().sum(),
        'Little_Participant_Race_Ethnicity': df['Little_Participant_Race_Ethnicity'].isna().sum()
    }
    
    # Report improvements
    for col in initial_missing:
        filled = initial_missing[col] - final_missing[col]
        message = ""
        if filled > 0:
            pct_improved = (filled / initial_missing[col]) * 100
            message = f"  - {col}: Filled {filled} missing values ({pct_improved:.1f}% improvement)\n"
        else:
            message = f"  - {col}: No improvement\n"
        print(message, end="")
        if report_file:
            report_file.write(message)
                
    if report_file:
        report_file.write("\n")
    print()
    
    return df

def fill_missing_occupation_data(df, report_file=None):
    """Fill missing occupation data using employer and education correlations"""
    message = "Performing occupation data imputation:\n"
    print(message)
    if report_file:
        report_file.write(message)
    
    initial_missing = {
        'Big_Occupation': df['Big_Occupation'].isna().sum(),
        'Big_Level_of_Education': df['Big_Level_of_Education'].isna().sum(),
    }
    
    # Map employer to most common occupation
    employer_to_occupation = {}
    for employer, group in df.dropna(subset=['Big_Employer', 'Big_Occupation']).groupby('Big_Employer'):
        if len(group) >= 3:  # Only consider employers with at least 3 records
            occupation_counts = group['Big_Occupation'].value_counts()
            if not occupation_counts.empty:
                employer_to_occupation[employer] = occupation_counts.index[0]
    
    # Fill missing occupations using employer mapping
    occupation_mask = df['Big_Occupation'].isna() & df['Big_Employer'].notna()
    occupation_before = df['Big_Occupation'].isna().sum()
    df.loc[occupation_mask, 'Big_Occupation'] = df.loc[occupation_mask, 'Big_Employer'].map(employer_to_occupation)
    occupation_after = df['Big_Occupation'].isna().sum()
    
    # Fill missing education levels based on occupation
    occupation_to_education = {}
    for occupation, group in df.dropna(subset=['Big_Occupation', 'Big_Level_of_Education']).groupby('Big_Occupation'):
        if len(group) >= 3:  # Only consider occupations with at least 3 records
            education_counts = group['Big_Level_of_Education'].value_counts()
            if not education_counts.empty:
                occupation_to_education[occupation] = education_counts.index[0]
    
    # Apply education mapping
    education_mask = df['Big_Level_of_Education'].isna() & df['Big_Occupation'].notna()
    education_before = df['Big_Level_of_Education'].isna().sum()
    df.loc[education_mask, 'Big_Level_of_Education'] = df.loc[education_mask, 'Big_Occupation'].map(occupation_to_education)
    education_after = df['Big_Level_of_Education'].isna().sum()
    
    # Report improvements
    occupation_filled = occupation_before - occupation_after
    education_filled = education_before - education_after
    
    message = ""
    if occupation_filled > 0:
        pct_improved = (occupation_filled / initial_missing['Big_Occupation']) * 100
        message = f"  - Big_Occupation: Filled {occupation_filled} missing values ({pct_improved:.1f}% improvement)\n"
    else:
        message = f"  - Big_Occupation: No improvement\n"
    print(message, end="")
    if report_file:
        report_file.write(message)
        
    message = ""
    if education_filled > 0:
        pct_improved = (education_filled / initial_missing['Big_Level_of_Education']) * 100
        message = f"  - Big_Level_of_Education: Filled {education_filled} missing values ({pct_improved:.1f}% improvement)\n"
    else:
        message = f"  - Big_Level_of_Education: No improvement\n"
    print(message, end="")
    if report_file:
        report_file.write(message)
        
    if report_file:
        report_file.write("\n")
    print()
            
    return df

def main():
    print("===== DATA PREPROCESSING PIPELINE =====")
    
    # Create a comprehensive report file
    with open(REPORT_PATH, 'w') as report_file:
        report_file.write("Data Cleaning and Imputation Report\n")
        report_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write("="*80 + "\n\n")
        
        # Step 1: Initial Data Loading
        report_file.write("Step 1: Initial Data Loading\n")
        report_file.write("==========================\n\n")
        print("\nStep 1: Initial Data Loading")
        print("==========================")
        
        # Read the Excel file using an absolute path
        excel_path = os.path.join(project_root, 'Data', 'Novice.xlsx')
        print(f"Reading dataset: {excel_path}")
        report_file.write(f"Reading dataset: {excel_path}\n")
        
        df = pd.read_excel(excel_path)
        print(f"Dataset shape: {df.shape}")
        report_file.write(f"Dataset shape: {df.shape}\n\n")
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]
        
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
        
        # Track age-related columns before processing
        age_columns = ['Big_Birthdate', 'Little_Birthdate']
        initial_age_missing = {col: df[col].isna().sum() for col in age_columns}
        
        # Process age features
        df = process_age_features(df, report_file)
        
        # Track improvements in age features
        new_age_columns = ['Big_Age', 'Little_Birthdate_year', 'Age_Difference', 'Big_Age_Group']
        age_improvements = {}
        
        for col in new_age_columns:
            if col in df.columns:
                missing = df[col].isna().sum()
                age_improvements[col] = {
                    'Missing': missing,
                    'Total': len(df),
                    'Percentage': (missing/len(df))*100
                }
        
        # Report age feature improvements
        report_file.write("\nAge Feature Processing Results:\n")
        report_file.write("-----------------------------\n")
        for col, stats in age_improvements.items():
            report_file.write(f"{col}:\n")
            report_file.write(f"  - Missing values: {stats['Missing']}\n")
            report_file.write(f"  - Total records: {stats['Total']}\n")
            report_file.write(f"  - Missing percentage: {stats['Percentage']:.2f}%\n")
        report_file.write("\n")
        
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
        print(f"\nSaving processed dataset to: {FILLED_DATA_PATH}")
        df.to_excel(FILLED_DATA_PATH, index=False)
        report_file.write(f"\nCleaned dataset saved to: {FILLED_DATA_PATH}\n")
        
    print("\nFiles saved:")
    print(f"- Data cleaning report: {REPORT_PATH}")
    print(f"- Cleaned data: {FILLED_DATA_PATH}")
    print("\nData cleaning complete!")

if __name__ == "__main__":
    main() 
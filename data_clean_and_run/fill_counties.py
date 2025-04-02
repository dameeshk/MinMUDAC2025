import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
from datetime import datetime

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from script_dir)
project_root = os.path.dirname(script_dir)

# File paths
FILLED_DATA_PATH = os.path.join(script_dir, 'filled_data.xlsx')
REPORT_PATH = os.path.join(script_dir, 'data_cleaning_report.txt')

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
        'Big County': df['Big County'].isna().sum(),
        'Big Race/Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little Participant: Race/Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
    }
    
    # Map Census Block Groups to counties where known
    block_to_county = {}
    for idx, row in df.dropna(subset=['Big County', 'Big Home Census Block Group']).iterrows():
        block_to_county[row['Big Home Census Block Group']] = row['Big County']
    
    # Fill missing counties using the mapping
    county_mask = df['Big County'].isna() & df['Big Home Census Block Group'].notna()
    df.loc[county_mask, 'Big County'] = df.loc[county_mask, 'Big Home Census Block Group'].map(block_to_county)
    
    # Map Census Block Groups to race/ethnicity where known
    block_to_race = {}
    for idx, row in df.dropna(subset=['Big Race/Ethnicity', 'Big Home Census Block Group']).iterrows():
        block_to_race[row['Big Home Census Block Group']] = row['Big Race/Ethnicity']
    
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
    race_mask = df['Big Race/Ethnicity'].isna() & df['Big Home Census Block Group'].notna()
    df.loc[race_mask, 'Big Race/Ethnicity'] = df.loc[race_mask, 'Big Home Census Block Group'].map(reliable_block_to_race)
    
    # Apply similar logic for Little's race/ethnicity using Little's Census Block Group
    little_block_to_race = {}
    for idx, row in df.dropna(subset=['Little Participant: Race/Ethnicity', 'Little Mailing Address Census Block Group']).iterrows():
        little_block_to_race[row['Little Mailing Address Census Block Group']] = row['Little Participant: Race/Ethnicity']
    
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
    
    little_race_mask = df['Little Participant: Race/Ethnicity'].isna() & df['Little Mailing Address Census Block Group'].notna()
    df.loc[little_race_mask, 'Little Participant: Race/Ethnicity'] = df.loc[little_race_mask, 'Little Mailing Address Census Block Group'].map(reliable_little_block_to_race)
    
    # Track improvements
    final_missing = {
        'Big County': df['Big County'].isna().sum(),
        'Big Race/Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
        'Little Participant: Race/Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
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
        'Big Occupation': df['Big Occupation'].isna().sum(),
        'Big Level of Education': df['Big Level of Education'].isna().sum(),
    }
    
    # Map employer to most common occupation
    employer_to_occupation = {}
    for employer, group in df.dropna(subset=['Big Employer', 'Big Occupation']).groupby('Big Employer'):
        if len(group) >= 3:  # Only consider employers with at least 3 records
            occupation_counts = group['Big Occupation'].value_counts()
            if not occupation_counts.empty:
                employer_to_occupation[employer] = occupation_counts.index[0]
    
    # Fill missing occupations using employer mapping
    occupation_mask = df['Big Occupation'].isna() & df['Big Employer'].notna()
    occupation_before = df['Big Occupation'].isna().sum()
    df.loc[occupation_mask, 'Big Occupation'] = df.loc[occupation_mask, 'Big Employer'].map(employer_to_occupation)
    occupation_after = df['Big Occupation'].isna().sum()
    
    # Fill missing education levels based on occupation
    occupation_to_education = {}
    for occupation, group in df.dropna(subset=['Big Occupation', 'Big Level of Education']).groupby('Big Occupation'):
        if len(group) >= 3:  # Only consider occupations with at least 3 records
            education_counts = group['Big Level of Education'].value_counts()
            if not education_counts.empty:
                occupation_to_education[occupation] = education_counts.index[0]
    
    # Apply education mapping
    education_mask = df['Big Level of Education'].isna() & df['Big Occupation'].notna()
    education_before = df['Big Level of Education'].isna().sum()
    df.loc[education_mask, 'Big Level of Education'] = df.loc[education_mask, 'Big Occupation'].map(occupation_to_education)
    education_after = df['Big Level of Education'].isna().sum()
    
    # Report improvements
    occupation_filled = occupation_before - occupation_after
    education_filled = education_before - education_after
    
    message = ""
    if occupation_filled > 0:
        pct_improved = (occupation_filled / initial_missing['Big Occupation']) * 100
        message = f"  - Big Occupation: Filled {occupation_filled} missing values ({pct_improved:.1f}% improvement)\n"
    else:
        message = f"  - Big Occupation: No improvement\n"
    print(message, end="")
    if report_file:
        report_file.write(message)
        
    message = ""
    if education_filled > 0:
        pct_improved = (education_filled / initial_missing['Big Level of Education']) * 100
        message = f"  - Big Level of Education: Filled {education_filled} missing values ({pct_improved:.1f}% improvement)\n"
    else:
        message = f"  - Big Level of Education: No improvement\n"
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
        
        # Step 1: County Filling
        report_file.write("Step 1: Automatic County Filling\n")
        report_file.write("================================\n\n")
        print("\nStep 1: Automatic County Filling")
        print("================================")
        
        # Read the Excel file using an absolute path
        excel_path = os.path.join(project_root, 'Data', 'Novice.xlsx')
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
        
        # Create a dictionary to store census block to county mappings
        block_to_county = {}
        
        # First pass: Build mapping from known relationships in the data
        for idx, row in df.iterrows():
            if pd.notna(row['Big County']) and pd.notna(row['Big Home Census Block Group']):
                block_to_county[row['Big Home Census Block Group']] = row['Big County']
        
        # Count missing counties before filling
        missing_before = df['Big County'].isna().sum()
        total_records = len(df)
        
        # Second pass: Fill in missing counties using block group IDs
        filled_from_block_ids = 0
        missing_blocks = set()
        
        print("\nLooking up missing counties using Census Block Group IDs...")
        report_file.write("Looking up missing counties using Census Block Group IDs...\n")
        
        for idx, row in df.iterrows():
            if pd.isna(row['Big County']) and pd.notna(row['Big Home Census Block Group']):
                block_group = row['Big Home Census Block Group']
                county = get_county_from_block_group(block_group)
                if county:
                    df.at[idx, 'Big County'] = county
                    filled_from_block_ids += 1
                    if idx % 100 == 0:  # Progress update every 100 records
                        print(f"Processed {idx+1}/{total_records} records...")
                else:
                    missing_blocks.add(str(block_group))
        
        # Count remaining missing counties after automatic filling
        missing_final = df['Big County'].isna().sum()
        
        # Print final statistics
        final_stats = {
            "Total Records": total_records,
            "Initially Missing": missing_before,
            "Filled from Known Data": len(block_to_county),
            "Filled from Block Group IDs": filled_from_block_ids,
            "Total Filled": (missing_before - missing_final),
            "Still Missing": missing_final,
            "Success Rate": f"{((missing_before - missing_final)/missing_before*100):.1f}% of missing filled"
        }
        print_stats_box("County Filling Results", final_stats)
        
        # Write stats to report
        report_file.write("\nCounty Filling Results:\n")
        report_file.write("----------------------\n")
        for key, value in final_stats.items():
            report_file.write(f"{key}: {value}\n")
        report_file.write("\n")
        
        # Step 2: Enhanced Data Imputation
        report_file.write("\nStep 2: Additional Data Imputation\n")
        report_file.write("===============================\n\n")
        print("\nStep 2: Additional Data Imputation")
        print("===============================")
        
        # Calculate initial missing data for this step
        interim_missing = df.isna().sum().sum()
        report_file.write(f"Missing Data Before Additional Imputation:\n")
        report_file.write(f"Total missing values: {interim_missing} ({interim_missing/total_cells*100:.2f}%)\n\n")
        
        # Perform geographic data imputation
        df = fill_missing_geographic_data(df, report_file)
        
        # Perform occupation data imputation
        df = fill_missing_occupation_data(df, report_file)
        
        # Calculate final missing data
        total_missing_after = df.isna().sum().sum()
        cells_filled = total_missing_before - total_missing_after
        report_file.write(f"Final Missing Data Summary:\n")
        report_file.write(f"Total missing values: {total_missing_after} ({total_missing_after/total_cells*100:.2f}%)\n")
        report_file.write(f"Total cells filled: {cells_filled} ({cells_filled/total_missing_before*100:.2f}% of initial missing)\n\n")
        
        # Save the completely filled dataset
        df.to_excel(FILLED_DATA_PATH, index=False)
        report_file.write(f"Cleaned dataset saved to: {FILLED_DATA_PATH}\n\n")
        
        # Summary 
        report_file.write("\nSummary of Improvements:\n")
        report_file.write("=======================\n")
        report_file.write(f"Initial missing values: {total_missing_before} ({total_missing_before/total_cells*100:.2f}%)\n")
        report_file.write(f"Final missing values: {total_missing_after} ({total_missing_after/total_cells*100:.2f}%)\n")
        report_file.write(f"Overall improvement: {cells_filled} cells filled ({cells_filled/total_missing_before*100:.2f}%)\n")
        
    print("\nFiles saved:")
    print(f"- Data cleaning report: {REPORT_PATH}")
    print(f"- Cleaned data: {FILLED_DATA_PATH}")
    print("\nData cleaning complete!")

if __name__ == "__main__":
    main() 
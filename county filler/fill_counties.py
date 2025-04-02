import pandas as pd
import numpy as np
import json
from pathlib import Path
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from script_dir)
project_root = os.path.dirname(script_dir)

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

def main():
    print("Step 1: Automatic County Filling")
    print("================================")
    
    # Read the Excel file using an absolute path
    excel_path = os.path.join(project_root, 'Data', 'Novice.xlsx')
    df = pd.read_excel(excel_path)
    
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
    print_stats_box("Final Results", final_stats)
    
    # Save all results to a single comprehensive report
    report_path = Path('county filler/county_filling_report.txt')
    with open(report_path, 'w') as f:
        # Write summary statistics
        f.write("County Filling Report\n")
        f.write("===================\n\n")
        
        f.write("Final Results\n")
        f.write("------------\n")
        for key, value in final_stats.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Write remaining missing blocks
        f.write("Remaining Unmapped Census Block Groups:\n")
        f.write("====================================\n")
        for block in sorted(missing_blocks):
            f.write(f"{block}\n")
    
    # Save the updated dataframe
    output_path = os.path.join(script_dir, 'filled_data.xlsx')
    df.to_excel(output_path, index=False)
    
    print("\nFiles saved:")
    print(f"- Comprehensive report: {report_path}")
    print(f"- Updated data: {output_path}")

if __name__ == "__main__":
    main() 
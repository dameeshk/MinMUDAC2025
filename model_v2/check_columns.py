import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up from script_dir)
project_root = os.path.dirname(script_dir)

# Read the test Excel file
excel_path = os.path.join(project_root, 'Data', 'Test-Truncated-Restated.xlsx')
print(f"Reading test dataset: {excel_path}")

df = pd.read_excel(excel_path)
print(f"Test dataset shape: {df.shape}")
print("\nColumns in test dataset:")
for col in df.columns:
    print(f"- {col}")

# Load the enhanced data
df_enhanced = pd.read_excel('enhanced_data.xlsx')

# Print all column names
print("\nAll columns in the dataset:")
for col in df_enhanced.columns:
    print(f"- {col}")

# Print shape of the dataset
print(f"\nDataset shape: {df_enhanced.shape}") 
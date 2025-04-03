import pandas as pd
import os

# Get file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(script_dir, 'enhanced_data.xlsx')

# Load the data
print("Loading data...")
df = pd.read_excel(TRAIN_DATA_PATH)

# Get numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nNumeric columns:")
for col in numeric_cols:
    print(f"- {col}")

# Check for duplicate columns
print("\nChecking for duplicate columns...")
duplicates = df[numeric_cols].columns[df[numeric_cols].columns.duplicated()]
if len(duplicates) > 0:
    print("\nDuplicate columns found:")
    for col in duplicates:
        print(f"- {col}")
else:
    print("\nNo duplicate columns found.") 
import pandas as pd
import os

# Define the path to the test file
test_file_path = r"C:\Users\solo\Desktop\Repos\MinMUDAC2025\Data\Test-Truncated-Restated.xlsx"

def main():
    """Read the test file and print its columns"""
    print(f"Reading test file: {test_file_path}")
    
    # Check if the file exists
    if not os.path.exists(test_file_path):
        print(f"Error: File not found at {test_file_path}")
        return
    
    try:
        # Read the Excel file
        df = pd.read_excel(test_file_path)
        
        # Get and print the columns
        columns = df.columns.tolist()
        print(f"\nFound {len(columns)} columns in the test file:")
        print("=" * 50)
        for i, col in enumerate(columns, 1):
            print(f"{i}. {col}")
        
        # Print some basic stats
        print("\nDataset Statistics:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(columns)}")
        
        # Check for any missing values
        missing_values = df.isnull().sum().sum()
        print(f"Total missing values: {missing_values}")
        
        # Save columns to a text file for reference
        output_file = "test_file_columns.txt"
        with open(output_file, "w") as f:
            f.write("Test File Columns\n")
            f.write("=" * 50 + "\n")
            for i, col in enumerate(columns, 1):
                f.write(f"{i}. {col}\n")
        print(f"\nColumns saved to {output_file}")
        
    except Exception as e:
        print(f"Error reading the file: {str(e)}")

if __name__ == "__main__":
    main() 
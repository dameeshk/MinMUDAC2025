import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def analyze_excel(file_path):
    """
    Analyze an Excel file and output information about its structure
    """
    print(f"Analyzing Excel file: {file_path}")
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80 + "\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return
    
    try:
        # Read the Excel file
        print("Reading Excel file...")
        df = pd.read_excel(file_path)
        
        # Basic information
        print("\n## BASIC INFORMATION ##")
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        
        # Column names and data types
        print("\n## COLUMN INFORMATION ##")
        print("\nColumn Names:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        
        print("\nData Types:")
        for col in df.columns:
            dtype = df[col].dtype
            print(f"  {col}: {dtype}")
        
        # Missing values
        print("\n## MISSING VALUES ##")
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        print("\nMissing Values by Column:")
        for col in df.columns:
            count = missing_counts[col]
            percentage = missing_percentages[col]
            print(f"  {col}: {count} missing values ({percentage:.2f}%)")
        
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        print(f"\nTotal missing values: {total_missing} out of {total_cells} cells ({(total_missing/total_cells)*100:.2f}%)")
        
        # Statistical summary
        print("\n## STATISTICAL SUMMARY ##")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            print("\nNumeric Columns Summary:")
            for col in numeric_cols:
                print(f"\n  {col}:")
                print(f"    Min: {df[col].min()}")
                print(f"    Max: {df[col].max()}")
                print(f"    Mean: {df[col].mean()}")
                print(f"    Median: {df[col].median()}")
                print(f"    Standard Deviation: {df[col].std()}")
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) > 0:
            print("\n## CATEGORICAL COLUMNS ##")
            for col in categorical_cols:
                unique_values = df[col].nunique()
                print(f"\n  {col}:")
                print(f"    Unique values: {unique_values}")
                
                # Show value counts for columns with fewer than 20 unique values
                if unique_values < 20:
                    print("    Value counts:")
                    value_counts = df[col].value_counts()
                    for value, count in value_counts.items():
                        print(f"      {value}: {count} ({count/len(df)*100:.2f}%)")
        
        # Date columns
        date_cols = df.select_dtypes(include=['datetime64']).columns
        
        if len(date_cols) > 0:
            print("\n## DATE COLUMNS ##")
            for col in date_cols:
                print(f"\n  {col}:")
                print(f"    Earliest date: {df[col].min()}")
                print(f"    Latest date: {df[col].max()}")
                print(f"    Date range: {(df[col].max() - df[col].min()).days} days")
        
        # Potential data cleaning issues
        print("\n## POTENTIAL DATA CLEANING ISSUES ##")
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicate_rows} ({duplicate_rows/len(df)*100:.2f}%)")
        
        # Check for columns with high correlation
        if len(numeric_cols) > 1:
            print("\nHigh Correlation between Numeric Columns (>0.8):")
            corr_matrix = df[numeric_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                        for col1 in upper_tri.index 
                        for col2 in upper_tri.columns 
                        if corr_matrix.loc[col1, col2] > 0.8]
            
            if high_corr:
                for col1, col2, corr_val in high_corr:
                    print(f"  {col1} and {col2}: {corr_val:.3f}")
            else:
                print("  None found.")
        
        print("\n## RECOMMENDATIONS FOR DATA CLEANING ##")
        recommendations = []
        
        # Check for columns with high missing values
        high_missing = [col for col in df.columns if missing_percentages[col] > 10]
        if high_missing:
            recommendations.append(f"Consider handling or removing columns with high missing values: {', '.join(high_missing)}")
        
        # Check for outliers in numeric columns
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_pct = len(outliers) / len(df) * 100
                if outlier_pct > 5:
                    recommendations.append(f"Check for outliers in '{col}' ({len(outliers)} potential outliers, {outlier_pct:.2f}%)")
        
        if duplicate_rows > 0:
            recommendations.append("Remove duplicate rows")
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No significant data cleaning issues identified.")
        
        print("\n" + "="*80)
        print("Analysis complete!")
        
        return df
        
    except Exception as e:
        print(f"Error analyzing Excel file: {str(e)}")
        return None

if __name__ == "__main__":
    # Define the file path
    file_path = r"C:\Users\solo\Desktop\Repos\MinMUDAC2025\Data\Novice.xlsx"
    
    # Analyze the file and save the output to a text file
    original_stdout = sys.stdout
    output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "excel_analysis_results.txt")
    
    with open(output_file_path, 'w') as f:
        sys.stdout = f
        analyze_excel(file_path)
        sys.stdout = original_stdout
    
    print(f"Analysis complete! Results saved to: {output_file_path}") 
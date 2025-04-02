"""
Missing Data Percentage Analyzer

This script analyzes and compares the missing data percentages across 
different Excel files in the MinMUDAC2025 project:
1. The original novice.xlsx file
2. The filled_data.xlsx file (after county filling)
3. The enhanced_filled_data.xlsx file (after all imputations)

It generates a comprehensive report with statistics on missing data.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

# Set file paths
NOVICE_PATH = os.path.join('Data', 'novice.xlsx')
FILLED_PATH = os.path.join('county filler', 'filled_data.xlsx')
ENHANCED_PATH = os.path.join('county filler', 'enhanced_filled_data.xlsx')
OUTPUT_FILE = 'missing_data_comparison.txt'

def analyze_missing_data(file_path, file_description):
    """Analyze missing data in an Excel file and return statistics"""
    print(f"Analyzing {file_description}...")
    
    try:
        # Track loading time for large files
        start_time = time.time()
        df = pd.read_excel(file_path)
        load_time = time.time() - start_time
        
        # Basic information
        rows, cols = df.shape
        total_cells = rows * cols
        
        # Missing data analysis
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        # Missing by column
        missing_by_col = df.isnull().sum()
        missing_percentage_by_col = (missing_by_col / rows) * 100
        
        # High missing columns (>50%)
        high_missing_cols = missing_percentage_by_col[missing_percentage_by_col > 50]
        
        # Completely filled columns
        complete_cols = missing_percentage_by_col[missing_percentage_by_col == 0]
        
        return {
            'file_path': file_path,
            'description': file_description,
            'rows': rows,
            'columns': cols,
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': missing_percentage,
            'load_time': load_time,
            'missing_by_col': missing_by_col,
            'missing_percentage_by_col': missing_percentage_by_col,
            'high_missing_cols': high_missing_cols,
            'complete_cols': complete_cols
        }
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def write_report(results, output_file):
    """Write comprehensive report comparing missing data across files"""
    with open(output_file, 'w') as f:
        f.write("Missing Data Percentage Comparison Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary table
        f.write("## SUMMARY TABLE ##\n\n")
        f.write(f"{'File':<25} {'Rows':<8} {'Columns':<8} {'Missing %':<10} {'Missing Cells':<15}\n")
        f.write("-"*70 + "\n")
        
        for result in results:
            if result:
                f.write(f"{result['description']:<25} {result['rows']:<8} {result['columns']:<8} "
                        f"{result['missing_percentage']:.2f}%  {result['missing_cells']:<15}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        
        # Detailed analysis for each file
        for result in results:
            if result:
                f.write(f"DETAILED ANALYSIS: {result['description']}\n")
                f.write(f"File path: {result['file_path']}\n")
                f.write("-"*80 + "\n\n")
                
                f.write("Basic Information:\n")
                f.write(f"  - Rows: {result['rows']}\n")
                f.write(f"  - Columns: {result['columns']}\n")
                f.write(f"  - Total cells: {result['total_cells']}\n")
                f.write(f"  - Missing cells: {result['missing_cells']} ({result['missing_percentage']:.2f}%)\n")
                f.write(f"  - Analysis time: {result['load_time']:.2f} seconds\n\n")
                
                f.write("Column Completeness:\n")
                f.write(f"  - Completely filled columns: {len(result['complete_cols'])} ({len(result['complete_cols'])/result['columns']*100:.2f}%)\n")
                f.write(f"  - Columns with >50% missing: {len(result['high_missing_cols'])} ({len(result['high_missing_cols'])/result['columns']*100:.2f}%)\n\n")
                
                # Top 10 most missing columns
                f.write("Top 10 columns with most missing values:\n")
                top_missing = result['missing_percentage_by_col'].sort_values(ascending=False).head(10)
                for col, pct in top_missing.items():
                    f.write(f"  - {col}: {pct:.2f}% missing\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        # Improvement analysis
        if len(results) > 1 and all(results):
            f.write("IMPROVEMENT ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            # Calculate improvements
            orig_missing = results[0]['missing_cells']
            orig_pct = results[0]['missing_percentage']
            
            for i in range(1, len(results)):
                curr_missing = results[i]['missing_cells']
                curr_pct = results[i]['missing_percentage']
                
                abs_improvement = orig_missing - curr_missing
                pct_improvement = (orig_pct - curr_pct)
                rel_improvement = (abs_improvement / orig_missing) * 100
                
                f.write(f"Improvement from {results[0]['description']} to {results[i]['description']}:\n")
                f.write(f"  - Absolute reduction: {abs_improvement} cells filled\n")
                f.write(f"  - Percentage point reduction: {pct_improvement:.2f}%\n")
                f.write(f"  - Relative improvement: {rel_improvement:.2f}% of missing values filled\n\n")
                
                # Column-specific improvements
                f.write("Column-specific improvements (top 10):\n")
                improvements = {}
                
                for col in results[0]['missing_by_col'].index:
                    if col in results[i]['missing_by_col'].index:
                        orig_col_missing = results[0]['missing_by_col'][col]
                        curr_col_missing = results[i]['missing_by_col'][col]
                        
                        if orig_col_missing > 0:
                            col_improvement = orig_col_missing - curr_col_missing
                            col_improvement_pct = (col_improvement / orig_col_missing) * 100
                            improvements[col] = (col_improvement, col_improvement_pct)
                
                # Sort by absolute improvement
                sorted_improvements = sorted(improvements.items(), key=lambda x: x[1][0], reverse=True)
                
                for col, (abs_imp, rel_imp) in sorted_improvements[:10]:
                    f.write(f"  - {col}: {abs_imp} values filled ({rel_imp:.2f}%)\n")
                
                f.write("\n")
        
        f.write("Analysis complete!\n")

def main():
    print("Starting missing data analysis...")
    
    # Analyze each file
    results = []
    
    novice_result = analyze_missing_data(NOVICE_PATH, "Original novice.xlsx")
    results.append(novice_result)
    
    filled_result = analyze_missing_data(FILLED_PATH, "Filled data.xlsx")
    results.append(filled_result)
    
    enhanced_result = analyze_missing_data(ENHANCED_PATH, "Enhanced filled data.xlsx")
    results.append(enhanced_result)
    
    # Write the report
    write_report(results, OUTPUT_FILE)
    
    print(f"Analysis complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 
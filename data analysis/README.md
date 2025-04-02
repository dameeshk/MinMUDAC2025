# Data Analysis for MinMUDAC2025

## Project Overview
This project analyzes the MUDAC 2025 Novice dataset to prepare for building a predictive algorithm. The data is stored in an Excel file and contains information about mentorship matches.

## Analysis Results

We've performed initial data analysis on the MUDAC Novice dataset. Here's what we've discovered:

### Basic Information
- Number of rows: 3,275
- Number of columns: 66

### Data Structure
- The dataset contains information about mentor-mentee matching (Bigs and Littles)
- Includes demographic information, interests, match dates, and outcomes
- Contains a mix of categorical, numerical, and date data

### Key Findings
1. **High Missing Values**: The dataset has significant missing data (54.64% of all cells are missing)
   - Many columns have over 90% missing values
   - This will require careful handling during model development

2. **Categorical Data**: There are numerous categorical columns, many with multiple possible values
   - These will need encoding for model development

3. **Date Features**: The dataset contains 13 date columns
   - These can be transformed into useful features like match duration, wait times, etc.

4. **No Duplicates**: There are no duplicate rows in the dataset

5. **Potential Target Variables**:
   - "Match Length" (already present, complete data)
   - "Closure Reason" (categorical, 24% missing values)
   - "Stage" (categorical, complete data)

## Next Steps for Building a Predictive Algorithm

1. **Data Cleaning**:
   - Handle missing values (imputation or removal)
   - Convert date columns to useful features (e.g., duration, age)
   - Encode categorical variables
   - Normalize/standardize numeric features

2. **Feature Engineering**:
   - Create meaningful features from existing data
   - Calculate derived metrics like time periods between events
   - Group sparse categorical values

3. **Feature Selection**:
   - Identify most important features for prediction
   - Remove highly correlated features
   - Reduce dimensionality if needed

4. **Model Development**:
   - Define prediction targets (e.g., match length, match success)
   - Split data into training and testing sets
   - Try various algorithms (e.g., Random Forest, XGBoost, Neural Networks)
   - Perform cross-validation

5. **Model Evaluation**:
   - Use appropriate metrics (RMSE, MAE for regression; accuracy, F1 for classification)
   - Compare model performance
   - Perform feature importance analysis

6. **Model Refinement**:
   - Hyperparameter tuning
   - Ensemble methods
   - Address any issues with class imbalance

## Getting Started

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the data analysis script:
   ```
   python analyze_excel.py
   ```

3. Review the analysis results in the `excel_analysis_results.txt` file

## Files
- `analyze_excel.py`: Script to analyze the Excel dataset
- `excel_analysis_results.txt`: Detailed output of the analysis
- `requirements.txt`: Required Python packages 
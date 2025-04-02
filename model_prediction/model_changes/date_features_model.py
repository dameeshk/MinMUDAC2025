"""
Enhanced model evaluation script with date-based feature engineering
This script will run regression models with additional time-based features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
from datetime import datetime

# Set paths
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Data', 'Novice.xlsx')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'date_features_results.txt')

def create_date_features(df):
    """Create time-based features from date columns"""
    # Convert date columns to datetime if they aren't already
    date_columns = [
        'Big Approved Date', 'Big Birthdate', 'Match Activation Date',
        'Big Assessment Uploaded', 'Big Acceptance Date', 'Big Contact: Created Date',
        'Big Enrollment: Created Date', 'Little RTBM Date in MF',
        'Little Application Received', 'Little Interview Date',
        'Little Acceptance Date', 'Little Birthdate'
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Create time-based features
    features = pd.DataFrame()
    
    # Time to match features
    if all(col in df.columns for col in ['Big Approved Date', 'Match Activation Date']):
        features['days_approval_to_match'] = (df['Match Activation Date'] - df['Big Approved Date']).dt.days
    
    if all(col in df.columns for col in ['Big Acceptance Date', 'Match Activation Date']):
        features['days_acceptance_to_match'] = (df['Match Activation Date'] - df['Big Acceptance Date']).dt.days
    
    # Age features
    if all(col in df.columns for col in ['Big Birthdate', 'Little Birthdate']):
        features['big_age'] = (datetime.now() - df['Big Birthdate']).dt.days / 365.25
        features['little_age'] = (datetime.now() - df['Little Birthdate']).dt.days / 365.25
        features['age_difference'] = features['big_age'] - features['little_age']
    
    # Process time features
    if all(col in df.columns for col in ['Big Application Received', 'Big Interview Date']):
        features['days_application_to_interview'] = (df['Big Interview Date'] - df['Big Application Received']).dt.days
    
    if all(col in df.columns for col in ['Big Interview Date', 'Big Acceptance Date']):
        features['days_interview_to_acceptance'] = (df['Big Acceptance Date'] - df['Big Interview Date']).dt.days
    
    # Seasonality features
    if 'Match Activation Date' in df.columns:
        features['match_month'] = df['Match Activation Date'].dt.month
        features['match_quarter'] = df['Match Activation Date'].dt.quarter
        features['match_year'] = df['Match Activation Date'].dt.year
    
    # Handle missing values in new features
    features = features.fillna(features.mean())
    
    return features

# Create a results file
with open(OUTPUT_FILE, 'w') as f:
    f.write("Match Length Prediction Model Evaluation with Date Features\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    f.write(f"Reading data from: {DATA_PATH}\n\n")

# Load the dataset
try:
    df = pd.read_excel(DATA_PATH)
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Data loaded successfully. Shape: {df.shape}\n\n")
        f.write("Data Overview:\n")
        f.write(f"Total rows: {df.shape[0]}\n")
        f.write(f"Total columns: {df.shape[1]}\n\n")
except Exception as e:
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Error loading data: {str(e)}\n")
    exit(1)

# Check if 'Match Length' exists in the dataframe
if 'Match Length' not in df.columns:
    with open(OUTPUT_FILE, 'a') as f:
        f.write("Error: 'Match Length' column not found in the dataset\n")
    exit(1)

# Data preprocessing
with open(OUTPUT_FILE, 'a') as f:
    f.write("Data Preprocessing:\n")

# Keep track of the target variable
y = df['Match Length']
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Target variable: Match Length\n")
    f.write(f"Match Length statistics: Mean={y.mean():.2f}, Median={y.median():.2f}, Min={y.min():.2f}, Max={y.max():.2f}\n\n")

# Create date-based features
date_features = create_date_features(df)
with open(OUTPUT_FILE, 'a') as f:
    f.write("Created date-based features:\n")
    for col in date_features.columns:
        f.write(f"- {col}\n")
    f.write(f"\nTotal date features: {len(date_features.columns)}\n\n")

# Remove columns with high missing values (>80%)
missing_percentage = df.isnull().mean() * 100
high_missing_cols = missing_percentage[missing_percentage > 80].index.tolist()
df_filtered = df.drop(columns=high_missing_cols)

# Identify numeric and categorical columns
numeric_features = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('Match Length')  # Remove target variable

categorical_features = df_filtered.select_dtypes(include=['object']).columns.tolist()

# Remove ID columns and columns with too many unique values (>50)
id_cols = [col for col in df_filtered.columns if 'ID' in col]
high_cardinality_cols = [col for col in categorical_features 
                        if df_filtered[col].nunique() > 50]

categorical_features = [col for col in categorical_features 
                      if col not in id_cols and col not in high_cardinality_cols]

# Combine original features with date features
X = pd.concat([df_filtered[numeric_features + categorical_features], date_features], axis=1)

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Original numeric features: {len(numeric_features)}\n")
    f.write(f"Original categorical features: {len(categorical_features)}\n")
    f.write(f"Date-based features: {len(date_features.columns)}\n")
    f.write(f"Total features: {X.shape[1]}\n\n")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Data split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples\n\n")

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features + list(date_features.columns)),
        ('cat', categorical_transformer, categorical_features)
    ])

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100)
}

# Results storage
results = {}

# Model evaluation section
with open(OUTPUT_FILE, 'a') as f:
    f.write("Model Evaluation:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Model':<25} {'RMSE':<10} {'R²':<10}\n")
    f.write("-" * 80 + "\n")

# Train and evaluate each model
for name, model in models.items():
    # Create the pipeline with preprocessor and model
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'rmse': rmse,
        'r2': r2
    }
    
    # Log results
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"{name:<25} {rmse:<10.4f} {r2:<10.4f}\n")

# Sort models by RMSE
sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])

with open(OUTPUT_FILE, 'a') as f:
    f.write("\nRanked Models by RMSE (lower is better):\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Rank':<5} {'Model':<25} {'RMSE':<10} {'R²':<10}\n")
    f.write("-" * 80 + "\n")
    
    for i, (name, result) in enumerate(sorted_models, 1):
        f.write(f"{i:<5} {name:<25} {result['rmse']:<10.4f} {result['r2']:<10.4f}\n")

# Best model details
best_model_name = sorted_models[0][0]
best_rmse = results[best_model_name]['rmse']
best_r2 = results[best_model_name]['r2']

with open(OUTPUT_FILE, 'a') as f:
    f.write("\nBest Model: " + best_model_name + "\n")
    f.write(f"RMSE: {best_rmse:.4f}\n")
    f.write(f"R²: {best_r2:.4f}\n\n")
    f.write("Analysis complete.\n")

print(f"Analysis complete. Results saved to {OUTPUT_FILE}") 
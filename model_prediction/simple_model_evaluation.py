"""
Simplified model evaluation script for match length prediction
This script will run 5 regression models and write the results to a file.
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
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'Novice.xlsx')
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_results.txt')

# Create a results file
with open(OUTPUT_FILE, 'w') as f:
    f.write("Match Length Prediction Model Evaluation\n")
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

# Remove columns with high missing values (>80%)
missing_percentage = df.isnull().mean() * 100
high_missing_cols = missing_percentage[missing_percentage > 80].index.tolist()
df_filtered = df.drop(columns=high_missing_cols)
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Removed {len(high_missing_cols)} columns with >80% missing values\n")
    f.write(f"Remaining columns: {df_filtered.shape[1]}\n\n")

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

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Numeric features: {len(numeric_features)}\n")
    f.write(f"Categorical features: {len(categorical_features)}\n")
    f.write(f"Removed {len(id_cols)} ID columns and {len(high_cardinality_cols)} high cardinality columns\n\n")

# Create X dataframe with selected features
X = df_filtered[numeric_features + categorical_features]
with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Final feature set: {X.shape[1]} columns\n\n")

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
        ('num', numeric_transformer, numeric_features),
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
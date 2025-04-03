import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess the data for modeling"""
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying the original dataframe
    X = df.copy()
    
    # Remove columns that are likely to cause issues
    cols_to_drop = [
        'Stage',
        'Closure Reason',
        'Closure Details',
        'Match Closure Meeting Date',
        'Match Length',  # Target variable
        'Match ID 18Char',  # Identifier column
        'Little ID',  # Identifier column
        'Big ID'  # Identifier column
    ]
    
    # Drop columns
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    # Get target variable
    y = df['Match Length'].copy()
    
    # Age-related feature engineering
    print("Creating age-related features...")
    
    # Calculate age difference
    X['Age_Difference'] = abs(X['Big_Age'] - X['Little_Age'])
    
    # Create age gap categories
    X['Age_Gap_Category'] = pd.cut(X['Age_Difference'], 
                                  bins=[0, 5, 10, 15, float('inf')],
                                  labels=['Small (0-5)', 'Medium (6-10)', 'Large (11-15)', 'Very Large (16+)'])
    
    # Age group compatibility
    X['Same_Age_Group'] = (X['Big_Age_Group'] == X['Little_Age_Group']).astype(int)
    
    # Create age compatibility score (inverse of age difference, normalized)
    max_age_diff = X['Age_Difference'].max()
    X['Age_Compatibility_Score'] = 1 - (X['Age_Difference'] / max_age_diff)
    
    # Handle date columns
    date_columns = X.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        X[f'{col}_Year'] = X[col].dt.year
        X[f'{col}_Month'] = X[col].dt.month
        X = X.drop(columns=[col])
    
    # Convert boolean columns to integers
    bool_columns = X.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        X[col] = X[col].astype(int)
    
    # Convert categorical columns to strings and handle missing values
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        X[col] = X[col].fillna('Unknown')
        X[col] = X[col].astype(str)
    
    # Handle numeric missing values
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Create dummy variables for categorical columns
    X = pd.get_dummies(X, dummy_na=True)
    
    print(f"Final feature count: {X.shape[1]}")
    return X, y

def train_and_evaluate(model, model_name, results_file):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Training {model_name}...\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Write results
    results = f"""
{model_name} Results:
----------------
Mean Squared Error: {mse:.2f}
Root Mean Squared Error: {rmse:.2f}
R2 Score: {r2:.2f}
"""
    print(results)
    with open(results_file, 'a') as f:
        f.write(results)
    
    # Feature importance for RandomForest
    if model_name == "RandomForest":
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_text = "\nTop 20 most important features:\n"
        importance_text += feature_importance.head(20).to_string()
        importance_text += "\n\n"
        
        print(importance_text)
        with open(results_file, 'a') as f:
            f.write(importance_text)
    
    return model

# Create results file with timestamp
results_file = f'model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# Initialize results file with header
with open(results_file, 'w') as f:
    f.write("Match Length Prediction Model Results\n")
    f.write("===================================\n")
    f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

# Load the enhanced data
print("Loading data...")
df = pd.read_excel('enhanced_data.xlsx')

# Write dataset info
with open(results_file, 'a') as f:
    f.write(f"Dataset Information:\n")
    f.write(f"------------------\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Original features: {df.shape[1]}\n\n")

# Preprocess the data
X, y = preprocess_data(df)

# Write preprocessing info
with open(results_file, 'a') as f:
    f.write(f"Preprocessing Information:\n")
    f.write(f"------------------------\n")
    f.write(f"Final feature count: {X.shape[1]}\n")
    f.write(f"Target variable: Match Length\n\n")

# Split the data
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open(results_file, 'a') as f:
    f.write(f"Model Training Information:\n")
    f.write(f"-------------------------\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Testing samples: {len(X_test)}\n")
    f.write(f"Test size: 20%\n\n")

# Train and evaluate RandomForest
print("\nTraining Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model = train_and_evaluate(rf_model, "RandomForest", results_file)

# Train and evaluate XGBoost
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
xgb_model = train_and_evaluate(xgb_model, "XGBoost", results_file)

print(f"\nModel training complete! Results saved to: {results_file}") 
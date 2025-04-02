"""
Enhanced model evaluation script with date-based feature engineering
This script will run regression models with additional time-based features,
improved imputation of missing values, and optimized feature interactions.

NOTE: This script uses the filled_data.xlsx file, which is created by the 
fill_counties.py script. Run fill_counties.py first.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.feature_selection import SelectFromModel, RFE
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'filled_data.xlsx')
OUTPUT_FILE = os.path.join(current_dir, 'optimized_features_results.txt')
FEATURE_IMPORTANCE_PATH = os.path.join(current_dir, 'feature_importance.png')

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
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create time-based features
    features = pd.DataFrame()
    
    # Early Stage Process Features
    if all(col in df.columns for col in ['Little Application Received', 'Little Interview Date']):
        features['days_application_to_interview'] = (df['Little Interview Date'] - df['Little Application Received']).dt.days
        features['days_application_to_interview'] = features['days_application_to_interview'].clip(lower=0)
        features['log_days_application_to_interview'] = np.log1p(features['days_application_to_interview'])
        
        # Add process pace indicators
        features['is_fast_application_process'] = (features['days_application_to_interview'] < 30).astype(int)
        features['is_slow_application_process'] = (features['days_application_to_interview'] > 90).astype(int)
        
        # Add seasonal application process features
        if 'Match Activation Date' in df.columns:
            features['application_month'] = df['Little Application Received'].dt.month
            features['is_summer_application'] = features['application_month'].isin([6, 7, 8]).astype(int)
            features['is_school_year_application'] = (~features['is_summer_application']).astype(int)
    
    # Big Process Features
    if all(col in df.columns for col in ['Big Contact: Created Date', 'Big Interview Date']):
        features['days_contact_to_interview'] = (df['Big Interview Date'] - df['Big Contact: Created Date']).dt.days
        features['days_contact_to_interview'] = features['days_contact_to_interview'].clip(lower=0)
        features['log_days_contact_to_interview'] = np.log1p(features['days_contact_to_interview'])
        
        # Add process pace indicators for Big
        features['is_fast_big_process'] = (features['days_contact_to_interview'] < 30).astype(int)
        features['is_slow_big_process'] = (features['days_contact_to_interview'] > 90).astype(int)
    
    # Combined Process Features
    if all(col in features.columns for col in ['days_application_to_interview', 'days_contact_to_interview']):
        features['process_speed_ratio'] = (features['days_application_to_interview'] / 
                                         features['days_contact_to_interview'].replace(0, 1)).clip(0, 10)
        features['process_speed_difference'] = (features['days_application_to_interview'] - 
                                              features['days_contact_to_interview']).clip(-90, 90)
        features['is_synchronized_process'] = (abs(features['process_speed_difference']) < 30).astype(int)
    
    # Time to match features (existing)
    if all(col in df.columns for col in ['Big Approved Date', 'Match Activation Date']):
        features['days_approval_to_match'] = (df['Match Activation Date'] - df['Big Approved Date']).dt.days
        features['days_approval_to_match'] = features['days_approval_to_match'].clip(lower=0)
        features['log_days_approval_to_match'] = np.log1p(features['days_approval_to_match'])
    
    if all(col in df.columns for col in ['Big Acceptance Date', 'Match Activation Date']):
        features['days_acceptance_to_match'] = (df['Match Activation Date'] - df['Big Acceptance Date']).dt.days
        features['days_acceptance_to_match'] = features['days_acceptance_to_match'].clip(lower=0)
        features['log_days_acceptance_to_match'] = np.log1p(features['days_acceptance_to_match'])
    
    # Age features (existing)
    if all(col in df.columns for col in ['Big Birthdate', 'Little Birthdate']):
        features['big_age'] = (datetime.now() - df['Big Birthdate']).dt.days / 365.25
        features['little_age'] = (datetime.now() - df['Little Birthdate']).dt.days / 365.25
        
        # Clip ages to reasonable ranges
        features['big_age'] = features['big_age'].clip(lower=18, upper=100)
        features['little_age'] = features['little_age'].clip(lower=5, upper=21)
        
        features['age_difference'] = features['big_age'] - features['little_age']
        
        # Add age interaction features with safety checks
        features['age_ratio'] = (features['big_age'] / features['little_age']).clip(lower=1, upper=10)
        features['age_product'] = (features['big_age'] * features['little_age'] / 100).clip(upper=1000)
        
        # Add squared terms with scaling to prevent overflow
        features['big_age_squared'] = (features['big_age'] ** 2) / 100
        features['little_age_squared'] = (features['little_age'] ** 2) / 100
        features['age_difference_squared'] = (features['age_difference'] ** 2) / 100
        
        # Add scaled polynomial features
        features['big_age_cubed'] = (features['big_age'] ** 3) / 10000
        features['little_age_cubed'] = (features['little_age'] ** 3) / 10000
        features['age_diff_cubed'] = (features['age_difference'] ** 3) / 10000
        features['age_ratio_squared'] = (features['age_ratio'] ** 2).clip(upper=100)
        
        # Add logarithmic transformations of age features
        features['log_big_age'] = np.log1p(features['big_age'])
        features['log_little_age'] = np.log1p(features['little_age'])
        features['log_age_diff'] = np.log1p(abs(features['age_difference']))
    
    # Process time features (existing)
    if all(col in df.columns for col in ['Big Interview Date', 'Big Acceptance Date']):
        features['days_interview_to_acceptance'] = (df['Big Acceptance Date'] - df['Big Interview Date']).dt.days
        features['days_interview_to_acceptance'] = features['days_interview_to_acceptance'].clip(lower=0)
        features['log_days_interview_to_acceptance'] = np.log1p(features['days_interview_to_acceptance'])
    
    # Add more time sequence features with validation
    if all(col in df.columns for col in ['Little Application Received', 'Match Activation Date']):
        features['days_application_to_match'] = (df['Match Activation Date'] - df['Little Application Received']).dt.days
        features['days_application_to_match'] = features['days_application_to_match'].clip(lower=0)
        features['log_days_application_to_match'] = np.log1p(features['days_application_to_match'])
        
        # Add process efficiency indicators
        if 'days_application_to_interview' in features.columns:
            features['interview_to_match_ratio'] = (features['days_application_to_match'] / 
                                                  features['days_application_to_interview'].replace(0, 1)).clip(0, 10)
            features['is_efficient_process'] = (features['interview_to_match_ratio'] < 2).astype(int)
    
    if all(col in df.columns for col in ['Little Interview Date', 'Match Activation Date']):
        features['days_interview_to_match'] = (df['Match Activation Date'] - df['Little Interview Date']).dt.days
        features['days_interview_to_match'] = features['days_interview_to_match'].clip(lower=0)
        features['log_days_interview_to_match'] = np.log1p(features['days_interview_to_match'])
    
    if all(col in df.columns for col in ['Little Acceptance Date', 'Match Activation Date']):
        features['days_little_acceptance_to_match'] = (df['Match Activation Date'] - df['Little Acceptance Date']).dt.days
        features['days_little_acceptance_to_match'] = features['days_little_acceptance_to_match'].clip(lower=0)
        features['log_days_little_acceptance_to_match'] = np.log1p(features['days_little_acceptance_to_match'])
    
    # Add time sequence ratios with safety checks
    if all(col in features.columns for col in ['days_acceptance_to_match', 'days_interview_to_acceptance']):
        denominator = features['days_interview_to_acceptance'].replace(0, 1)
        features['interview_acceptance_match_ratio'] = (features['days_acceptance_to_match'] / denominator).clip(0, 10)
    
    # Seasonality features (existing)
    if 'Match Activation Date' in df.columns:
        features['match_month'] = df['Match Activation Date'].dt.month
        features['match_quarter'] = df['Match Activation Date'].dt.quarter
        features['match_year'] = df['Match Activation Date'].dt.year
        features['match_day_of_week'] = df['Match Activation Date'].dt.dayofweek
        
        # Add cyclical encodings (these are already bounded by sin/cos)
        features['month_sin'] = np.sin(2 * np.pi * features['match_month']/12)
        features['month_cos'] = np.cos(2 * np.pi * features['match_month']/12)
        features['quarter_sin'] = np.sin(2 * np.pi * features['match_quarter']/4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['match_quarter']/4)
        features['day_of_week_sin'] = np.sin(2 * np.pi * features['match_day_of_week']/7)
        features['day_of_week_cos'] = np.cos(2 * np.pi * features['match_day_of_week']/7)
        
        # Add year features with reasonable bounds
        current_year = datetime.now().year
        features['years_since_match'] = (current_year - features['match_year']).clip(0, 20)
        features['log_years_since_match'] = np.log1p(features['years_since_match'])
        
        # Binary seasonal indicators (these are already 0/1)
        features['is_summer'] = features['match_month'].isin([6, 7, 8]).astype(int)
        features['is_winter'] = features['match_month'].isin([12, 1, 2]).astype(int)
        features['is_spring'] = features['match_month'].isin([3, 4, 5]).astype(int)
        features['is_fall'] = features['match_month'].isin([9, 10, 11]).astype(int)
        features['is_school_year'] = (~features['is_summer']).astype(int)
        features['is_school_start'] = features['match_month'].isin([8, 9]).astype(int)
        features['is_school_end'] = features['match_month'].isin([5, 6]).astype(int)
        features['is_weekend'] = features['match_day_of_week'].isin([5, 6]).astype(int)
        
        # Year-based features with scaling
        features['match_year_centered'] = (features['match_year'] - 2020)  # Center around a reasonable year
        features['match_year_squared'] = (features['match_year_centered'] ** 2) / 100  # Scale down
    
    # Replace any remaining infinities with NaN
    features = features.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values - use median for numeric features
    features = features.fillna(features.median())
    
    return features

def create_optimized_interactions(df, numeric_features, categorical_features, date_features):
    """Create targeted interaction features based on domain knowledge"""
    interactions = pd.DataFrame(index=df.index)
    
    # Create combined features DataFrame for easier access
    combined_df = pd.concat([
        df[numeric_features + categorical_features],
        date_features
    ], axis=1)
    
    # Get top categories by frequency
    top_counties = df['Big County'].value_counts().nlargest(5).index.tolist() if 'Big County' in df.columns else []
    
    # 1. Gender match interactions (already binary)
    if all(col in df.columns for col in ['Big Gender', 'Little Gender']):
        interactions['gender_match'] = (df['Big Gender'] == df['Little Gender']).astype(int)
        
        if 'big_age' in date_features.columns:
            # Scale interactions to prevent overflow
            interactions['gender_match_x_big_age'] = (interactions['gender_match'] * date_features['big_age']).clip(0, 100)
            interactions['gender_match_x_age_difference'] = (interactions['gender_match'] * date_features['age_difference']).clip(-50, 50)
            interactions['gender_match_x_age_product'] = (interactions['gender_match'] * date_features['age_product']).clip(0, 1000)
            interactions['gender_match_x_age_ratio'] = (interactions['gender_match'] * date_features['age_ratio']).clip(1, 10)
    
    # 2. Program type interactions (binary features)
    if 'Program Type' in df.columns:
        interactions['is_community'] = (df['Program Type'] == 'Community').astype(int)
        
        if 'big_age' in date_features.columns:
            interactions['community_x_big_age'] = (interactions['is_community'] * date_features['big_age']).clip(0, 100)
            interactions['community_x_age_difference'] = (interactions['is_community'] * date_features['age_difference']).clip(-50, 50)
            interactions['community_x_age_ratio'] = (interactions['is_community'] * date_features['age_ratio']).clip(1, 10)
        
        # Binary interactions don't need clipping
        if 'is_summer' in date_features.columns:
            interactions['community_x_summer'] = interactions['is_community'] * date_features['is_summer']
            interactions['community_x_school_year'] = interactions['is_community'] * date_features['is_school_year']
            
            if 'is_winter' in date_features.columns:
                interactions['community_x_winter'] = interactions['is_community'] * date_features['is_winter']
            if 'is_school_start' in date_features.columns:
                interactions['community_x_school_start'] = interactions['is_community'] * date_features['is_school_start']
                interactions['community_x_school_end'] = interactions['is_community'] * date_features['is_school_end']
    
    # 3. County interactions (binary features)
    for county in top_counties:
        if 'Big Age' in numeric_features:
            interactions[f'county_{county}'] = (df['Big County'] == county).astype(int)
            interactions[f'county_{county}_x_big_age'] = (interactions[f'county_{county}'] * df['Big Age']).clip(0, 100)
        
        if 'is_summer' in date_features.columns:
            interactions[f'county_{county}_x_summer'] = interactions[f'county_{county}'] * date_features['is_summer']
            
            if 'is_school_start' in date_features.columns:
                interactions[f'county_{county}_x_school_start'] = interactions[f'county_{county}'] * date_features['is_school_start']
    
    # 4. Age difference threshold interactions (binary features)
    if 'age_difference' in date_features.columns:
        interactions['large_age_diff'] = (date_features['age_difference'] > 20).astype(int)
        interactions['medium_age_diff'] = ((date_features['age_difference'] >= 10) & 
                                          (date_features['age_difference'] <= 20)).astype(int)
        interactions['small_age_diff'] = (date_features['age_difference'] < 10).astype(int)
        interactions['very_large_age_diff'] = (date_features['age_difference'] > 30).astype(int)
        
        if 'is_summer' in date_features.columns:
            interactions['large_age_diff_x_summer'] = interactions['large_age_diff'] * date_features['is_summer']
            interactions['large_age_diff_x_school_year'] = interactions['large_age_diff'] * date_features['is_school_year']
    
    # 5. Year since match interactions
    if 'years_since_match' in date_features.columns:
        if 'Big Age' in numeric_features:
            interactions['years_since_x_big_age'] = (date_features['years_since_match'] * df['Big Age'] / 10).clip(0, 100)
            interactions['years_since_squared'] = (date_features['years_since_match'] ** 2 / 100).clip(0, 100)
            interactions['years_since_cubed'] = (date_features['years_since_match'] ** 3 / 1000).clip(0, 100)
    
    # Replace any remaining infinities with NaN
    interactions = interactions.replace([np.inf, -np.inf], np.nan)
    
    # Handle missing values - use 0 for interaction features
    interactions = interactions.fillna(0)
    
    return interactions

def select_optimal_features(X_train, y_train, X_test, feature_names, threshold='median'):
    """Select features using a more targeted approach combining importance and domain knowledge"""
    
    with open(OUTPUT_FILE, 'a') as f:
        f.write("Performing optimized feature selection:\n")
    
    # Train RF model for feature importance
    feature_selector = RandomForestRegressor(n_estimators=100, random_state=42)
    feature_selector.fit(X_train, y_train)
    
    # Get feature importances
    importances = feature_selector.feature_importances_
    
    # Create DataFrame of features and importance scores
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance for top features
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH)
    plt.close()
    
    # Use a less aggressive threshold to keep more features
    # Options: 'mean', 'median', or a float value like 0.001
    selector = SelectFromModel(feature_selector, threshold=threshold)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]
    
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"  - Selected {len(selected_features)} out of {len(feature_names)} features\n")
        f.write("  - Top 10 most important features:\n")
        for i, (feature, importance) in enumerate(feature_importance.head(10).values):
            f.write(f"    {i+1}. {feature}: {importance:.6f}\n")
        f.write(f"  - Feature importance plot saved to: {FEATURE_IMPORTANCE_PATH}\n\n")
    
    return X_train_selected, X_test_selected, selected_features, feature_importance

# Create a results file
with open(OUTPUT_FILE, 'w') as f:
    f.write("Match Length Prediction Model with Optimized Feature Engineering\n")
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
        f.write(f"Please run fill_counties.py first to generate the filled_data.xlsx file\n")
    print(f"Error: {str(e)}")
    print("Please run fill_counties.py first to generate the filled_data.xlsx file")
    exit(1)

# Check if 'Match Length' exists in the dataframe
if 'Match Length' not in df.columns:
    with open(OUTPUT_FILE, 'a') as f:
        f.write("Error: 'Match Length' column not found in the dataset\n")
    print("Error: 'Match Length' column not found in the dataset")
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

# Use different missing value thresholds for different column types
# Keep more categorical columns that might be important
missing_percentage = df_filtered.isnull().mean() * 100
cat_high_missing = [col for col in df_filtered.select_dtypes(include=['object']).columns
                   if missing_percentage[col] > 60 and col not in ['Big County', 'Program Type', 'Stage']]
df_filtered = df_filtered.drop(columns=cat_high_missing)

# Identify numeric and categorical columns
numeric_features = df_filtered.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('Match Length')  # Remove target variable

categorical_features = df_filtered.select_dtypes(include=['object']).columns.tolist()

# Remove ID columns and columns with too many unique values (>50)
id_cols = [col for col in df_filtered.columns if 'ID' in col]
high_cardinality_cols = [col for col in categorical_features 
                        if df_filtered[col].nunique() > 50 and col not in ['Big County']]

categorical_features = [col for col in categorical_features 
                      if col not in id_cols and col not in high_cardinality_cols]

# Create optimized interaction features
with open(OUTPUT_FILE, 'a') as f:
    f.write("Creating optimized interaction features:\n")

interaction_features = create_optimized_interactions(df_filtered, numeric_features, categorical_features, date_features)

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"  - Created {interaction_features.shape[1]} optimized interaction features\n\n")

# Combine original features with date features and interactions
X = pd.concat([df_filtered[numeric_features + categorical_features], 
              date_features, 
              interaction_features], axis=1)

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Original numeric features: {len(numeric_features)}\n")
    f.write(f"Original categorical features: {len(categorical_features)}\n")
    f.write(f"Date-based features: {len(date_features.columns)}\n")
    f.write(f"Interaction features: {len(interaction_features.columns)}\n")
    f.write(f"Total features before selection: {X.shape[1]}\n\n")

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

# Get the list of all feature names for feature selection
feature_names = list(X.columns)
date_features_cols = list(date_features.columns)
interaction_features_cols = list(interaction_features.columns)

# Preprocess the data for feature selection
num_cols = numeric_features + date_features_cols + interaction_features_cols
cat_cols = categorical_features

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ], remainder='passthrough')

# Preprocess data for feature selection
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get the feature names after preprocessing
cat_feature_names = []
for name, _, cols in preprocessor.transformers_:
    if name == 'cat':
        encoder = preprocessor.named_transformers_[name].named_steps['onehot']
        cat_indices = []
        for i, col in enumerate(cols):
            col_idx = X_train.columns.get_loc(col)
            cat_indices.append(col_idx)
        cat_feature_names = encoder.get_feature_names_out(
            [X_train.columns[idx] for idx in cat_indices])

processed_feature_names = list(num_cols) + list(cat_feature_names)

# Perform feature selection with a less aggressive threshold (0.005 or 'median')
X_train_selected, X_test_selected, selected_features, feature_importance = select_optimal_features(
    X_train_processed, y_train, X_test_processed, processed_feature_names, threshold=0.005
)

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"Total features after selection: {X_train_selected.shape[1]}\n\n")

# Models to evaluate with hyperparameter optimization
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    ),
    "LightGBM": lgb.LGBMRegressor(
        objective='regression',
        random_state=42
    ),
}

# Define parameter grids for hyperparameter optimization
param_grids = {
    "Random Forest": {
        'n_estimators': [100, 150, 200],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0]
    },
    "XGBoost": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5, 6, 7],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    "LightGBM": {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'max_depth': [4, 5, 6, 7],
        'min_child_samples': [15, 20, 25],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
}

# Results storage
results = {}

# Model evaluation section
with open(OUTPUT_FILE, 'a') as f:
    f.write("Model Evaluation with Optimized Features:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Model':<25} {'RMSE':<10} {'R²':<10}\n")
    f.write("-" * 80 + "\n")

# Train and evaluate each model
for name, model in models.items():
    if name in param_grids:
        # Perform hyperparameter optimization
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_selected, y_train)
        best_model = grid_search.best_estimator_
        
        # Log best parameters
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"\nBest parameters for {name}:\n")
            for param, value in grid_search.best_params_.items():
                f.write(f"  {param}: {value}\n")
    else:
        best_model = model
        best_model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = best_model.predict(X_test_selected)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'rmse': rmse,
        'r2': r2,
        'model': best_model
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
best_model = results[best_model_name]['model']

with open(OUTPUT_FILE, 'a') as f:
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"RMSE: {best_rmse:.4f}\n")
    f.write(f"R²: {best_r2:.4f}\n\n")
    f.write("Analysis complete.\n")

print(f"Analysis complete. Results saved to {OUTPUT_FILE}")
print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
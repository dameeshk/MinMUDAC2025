"""
Enhanced model evaluation script with date-based feature engineering
This script will run regression models with additional time-based features,
improved imputation of missing values, and optimized feature interactions.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, 'filled_data.xlsx')
OUTPUT_FILE = os.path.join(current_dir, 'optimized_features_results.txt')
ENHANCED_DATA_PATH = os.path.join(current_dir, 'enhanced_filled_data.xlsx')
FEATURE_IMPORTANCE_PATH = os.path.join(current_dir, 'feature_importance.png')

def fill_missing_geographic_data(df):
    """Fill missing values using Census Block Group relationships"""
    with open(OUTPUT_FILE, 'a') as f:
        f.write("Performing geographic data imputation:\n")
        
        # Track initial missing counts
        initial_missing = {
            'Big County': df['Big County'].isna().sum(),
            'Big Race/Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
            'Little Participant: Race/Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
        }
        
        # Map Census Block Groups to counties where known
        block_to_county = {}
        for idx, row in df.dropna(subset=['Big County', 'Big Home Census Block Group']).iterrows():
            block_to_county[row['Big Home Census Block Group']] = row['Big County']
        
        # Fill missing counties using the mapping
        county_mask = df['Big County'].isna() & df['Big Home Census Block Group'].notna()
        df.loc[county_mask, 'Big County'] = df.loc[county_mask, 'Big Home Census Block Group'].map(block_to_county)
        
        # Map Census Block Groups to race/ethnicity where known
        block_to_race = {}
        for idx, row in df.dropna(subset=['Big Race/Ethnicity', 'Big Home Census Block Group']).iterrows():
            block_to_race[row['Big Home Census Block Group']] = row['Big Race/Ethnicity']
        
        # Fill missing race/ethnicity using the mapping (only if we have high confidence)
        race_counts = {}
        for block, races in block_to_race.items():
            if block not in race_counts:
                race_counts[block] = {}
            if races in race_counts[block]:
                race_counts[block][races] += 1
            else:
                race_counts[block][races] = 1
        
        # Only use blocks where we have a dominant race pattern
        reliable_block_to_race = {}
        for block, counts in race_counts.items():
            if len(counts) > 0:
                dominant_race = max(counts.items(), key=lambda x: x[1])[0]
                dominant_count = counts[dominant_race]
                total = sum(counts.values())
                if dominant_count / total > 0.7 and total >= 3:  # 70% threshold and at least 3 samples
                    reliable_block_to_race[block] = dominant_race
        
        # Apply the reliable mapping
        race_mask = df['Big Race/Ethnicity'].isna() & df['Big Home Census Block Group'].notna()
        df.loc[race_mask, 'Big Race/Ethnicity'] = df.loc[race_mask, 'Big Home Census Block Group'].map(reliable_block_to_race)
        
        # Apply similar logic for Little's race/ethnicity using Little's Census Block Group
        little_block_to_race = {}
        for idx, row in df.dropna(subset=['Little Participant: Race/Ethnicity', 'Little Mailing Address Census Block Group']).iterrows():
            little_block_to_race[row['Little Mailing Address Census Block Group']] = row['Little Participant: Race/Ethnicity']
        
        # Same reliability filtering as above
        race_counts = {}
        for block, races in little_block_to_race.items():
            if block not in race_counts:
                race_counts[block] = {}
            if races in race_counts[block]:
                race_counts[block][races] += 1
            else:
                race_counts[block][races] = 1
        
        reliable_little_block_to_race = {}
        for block, counts in race_counts.items():
            if len(counts) > 0:
                dominant_race = max(counts.items(), key=lambda x: x[1])[0]
                dominant_count = counts[dominant_race]
                total = sum(counts.values())
                if dominant_count / total > 0.7 and total >= 3:
                    reliable_little_block_to_race[block] = dominant_race
        
        little_race_mask = df['Little Participant: Race/Ethnicity'].isna() & df['Little Mailing Address Census Block Group'].notna()
        df.loc[little_race_mask, 'Little Participant: Race/Ethnicity'] = df.loc[little_race_mask, 'Little Mailing Address Census Block Group'].map(reliable_little_block_to_race)
        
        # Track improvements
        final_missing = {
            'Big County': df['Big County'].isna().sum(),
            'Big Race/Ethnicity': df['Big Race/Ethnicity'].isna().sum(),
            'Little Participant: Race/Ethnicity': df['Little Participant: Race/Ethnicity'].isna().sum()
        }
        
        # Report improvements
        for col in initial_missing:
            filled = initial_missing[col] - final_missing[col]
            if filled > 0:
                pct_improved = (filled / initial_missing[col]) * 100
                f.write(f"  - {col}: Filled {filled} missing values ({pct_improved:.1f}% improvement)\n")
            else:
                f.write(f"  - {col}: No improvement\n")
                
        f.write("\n")
    
    return df

def fill_missing_occupation_data(df):
    """Fill missing occupation data using employer and education correlations"""
    with open(OUTPUT_FILE, 'a') as f:
        f.write("Performing occupation data imputation:\n")
        
        initial_missing = {
            'Big Occupation': df['Big Occupation'].isna().sum(),
            'Big Level of Education': df['Big Level of Education'].isna().sum(),
        }
        
        # Map employer to most common occupation
        employer_to_occupation = {}
        for employer, group in df.dropna(subset=['Big Employer', 'Big Occupation']).groupby('Big Employer'):
            if len(group) >= 3:  # Only consider employers with at least 3 records
                occupation_counts = group['Big Occupation'].value_counts()
                if not occupation_counts.empty:
                    employer_to_occupation[employer] = occupation_counts.index[0]
        
        # Fill missing occupations using employer mapping
        occupation_mask = df['Big Occupation'].isna() & df['Big Employer'].notna()
        occupation_before = df['Big Occupation'].isna().sum()
        df.loc[occupation_mask, 'Big Occupation'] = df.loc[occupation_mask, 'Big Employer'].map(employer_to_occupation)
        occupation_after = df['Big Occupation'].isna().sum()
        
        # Fill missing education levels based on occupation
        occupation_to_education = {}
        for occupation, group in df.dropna(subset=['Big Occupation', 'Big Level of Education']).groupby('Big Occupation'):
            if len(group) >= 3:  # Only consider occupations with at least 3 records
                education_counts = group['Big Level of Education'].value_counts()
                if not education_counts.empty:
                    occupation_to_education[occupation] = education_counts.index[0]
        
        # Apply education mapping
        education_mask = df['Big Level of Education'].isna() & df['Big Occupation'].notna()
        education_before = df['Big Level of Education'].isna().sum()
        df.loc[education_mask, 'Big Level of Education'] = df.loc[education_mask, 'Big Occupation'].map(occupation_to_education)
        education_after = df['Big Level of Education'].isna().sum()
        
        # Report improvements
        occupation_filled = occupation_before - occupation_after
        education_filled = education_before - education_after
        
        if occupation_filled > 0:
            pct_improved = (occupation_filled / initial_missing['Big Occupation']) * 100
            f.write(f"  - Big Occupation: Filled {occupation_filled} missing values ({pct_improved:.1f}% improvement)\n")
        else:
            f.write(f"  - Big Occupation: No improvement\n")
            
        if education_filled > 0:
            pct_improved = (education_filled / initial_missing['Big Level of Education']) * 100
            f.write(f"  - Big Level of Education: Filled {education_filled} missing values ({pct_improved:.1f}% improvement)\n")
        else:
            f.write(f"  - Big Level of Education: No improvement\n")
            
        f.write("\n")
            
    return df

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
        
        # Add age interaction features
        features['age_ratio'] = features['big_age'] / features['little_age']
        features['age_product'] = features['big_age'] * features['little_age'] / 100  # Scaled down to avoid large numbers
        
        # Add squared terms
        features['big_age_squared'] = features['big_age'] ** 2
        features['little_age_squared'] = features['little_age'] ** 2
        features['age_difference_squared'] = features['age_difference'] ** 2
        
        # Add age cubic terms (helps capture more complex relationships)
        features['big_age_cubed'] = features['big_age'] ** 3 / 1000  # Scale down
    
    # Process time features
    if all(col in df.columns for col in ['Big Interview Date', 'Big Acceptance Date']):
        features['days_interview_to_acceptance'] = (df['Big Acceptance Date'] - df['Big Interview Date']).dt.days
    
    # Seasonality features
    if 'Match Activation Date' in df.columns:
        features['match_month'] = df['Match Activation Date'].dt.month
        features['match_quarter'] = df['Match Activation Date'].dt.quarter
        features['match_year'] = df['Match Activation Date'].dt.year
        
        # Add cyclical month encoding (better captures seasons)
        features['month_sin'] = np.sin(2 * np.pi * features['match_month']/12)
        features['month_cos'] = np.cos(2 * np.pi * features['match_month']/12)
        
        # Quarter encoding
        features['quarter_sin'] = np.sin(2 * np.pi * features['match_quarter']/4)
        features['quarter_cos'] = np.cos(2 * np.pi * features['match_quarter']/4)
        
        # Add year interaction features
        current_year = datetime.now().year
        features['years_since_match'] = current_year - features['match_year']
        features['log_years_since_match'] = np.log1p(features['years_since_match'])
        
        # Season indicators
        features['is_summer'] = features['match_month'].isin([6, 7, 8]).astype(int)
        features['is_winter'] = features['match_month'].isin([12, 1, 2]).astype(int)
        features['is_school_year'] = (~features['is_summer']).astype(int)
        
        # School year transitions
        features['is_school_start'] = features['match_month'].isin([8, 9]).astype(int)
        features['is_school_end'] = features['match_month'].isin([5, 6]).astype(int)
    
    # Duration buckets (helps capture non-linear patterns in match length)
    if 'Match Length' in df.columns:
        # Create buckets by year
        features['length_1yr'] = (df['Match Length'] <= 12).astype(int)
        features['length_1_2yr'] = ((df['Match Length'] > 12) & (df['Match Length'] <= 24)).astype(int)
        features['length_2_3yr'] = ((df['Match Length'] > 24) & (df['Match Length'] <= 36)).astype(int)
        features['length_3yr_plus'] = (df['Match Length'] > 36).astype(int)
    
    # Handle missing values in new features
    features = features.fillna(features.mean())
    
    return features

def create_optimized_interactions(df, numeric_features, categorical_features, date_features):
    """Create targeted interaction features based on domain knowledge"""
    interactions = pd.DataFrame(index=df.index)
    
    # Create combined features DataFrame for easier access
    combined_df = pd.concat([
        df[numeric_features + categorical_features],
        date_features
    ], axis=1)
    
    # Get top categories by frequency to avoid creating too many sparse features
    top_counties = df['Big County'].value_counts().nlargest(3).index.tolist() if 'Big County' in df.columns else []
    
    # 1. Gender match interactions
    if all(col in df.columns for col in ['Big Gender', 'Little Gender']):
        interactions['gender_match'] = (df['Big Gender'] == df['Little Gender']).astype(int)
        
        # Add gender match interaction with age
        if 'big_age' in date_features.columns:
            interactions['gender_match_x_big_age'] = interactions['gender_match'] * date_features['big_age']
            interactions['gender_match_x_age_difference'] = interactions['gender_match'] * date_features['age_difference']
    
    # 2. Program type interactions
    if 'Program Type' in df.columns:
        # Community program is the most common
        interactions['is_community'] = (df['Program Type'] == 'Community').astype(int)
        
        if 'big_age' in date_features.columns:
            interactions['community_x_big_age'] = interactions['is_community'] * date_features['big_age']
        
        if 'is_summer' in date_features.columns:
            interactions['community_x_summer'] = interactions['is_community'] * date_features['is_summer']
            interactions['community_x_school_year'] = interactions['is_community'] * date_features['is_school_year']
    
    # 3. County and geographic interactions
    for county in top_counties:
        if 'Big Age' in numeric_features:
            interactions[f'county_{county}'] = (df['Big County'] == county).astype(int)
            interactions[f'county_{county}_x_big_age'] = interactions[f'county_{county}'] * df['Big Age']
        
        if 'is_summer' in date_features.columns:
            interactions[f'county_{county}_x_summer'] = interactions[f'county_{county}'] * date_features['is_summer']
    
    # 4. Age difference threshold interactions 
    if 'age_difference' in date_features.columns:
        interactions['large_age_diff'] = (date_features['age_difference'] > 20).astype(int)
        interactions['medium_age_diff'] = ((date_features['age_difference'] >= 10) & 
                                          (date_features['age_difference'] <= 20)).astype(int)
    
    # 5. Year since match interactions
    if 'years_since_match' in date_features.columns:
        if 'Big Age' in numeric_features:
            interactions['years_since_x_big_age'] = date_features['years_since_match'] * df['Big Age'] / 10
    
    # 6. Match month and program type interactions
    if 'match_month' in date_features.columns and 'Program Type' in df.columns:
        for program in df['Program Type'].unique():
            if len(df[df['Program Type'] == program]) > 50:  # Only for common programs
                is_program = (df['Program Type'] == program).astype(int)
                interactions[f'{program}_summer'] = is_program * date_features['is_summer']
                interactions[f'{program}_school_start'] = is_program * date_features['is_school_start']
    
    # 7. Special closure reason features
    if 'Closure Reason' in df.columns:
        # Focus on key reasons that appeared in feature importance
        key_reasons = ['Child: Graduated', 'Successful match closure', 
                      'Child/Family: Lost contact with agency']
        
        for reason in key_reasons:
            if reason in df['Closure Reason'].unique():
                interactions[f'closure_{reason}'] = (df['Closure Reason'] == reason).astype(int)
    
    # 8. Stage and time interactions
    if 'Stage' in df.columns and 'match_year' in date_features.columns:
        interactions['closed_stage'] = (df['Stage'] == 'Closed').astype(int)
        interactions['closed_x_match_year'] = interactions['closed_stage'] * date_features['match_year']
        
        if 'years_since_match' in date_features.columns:
            interactions['closed_x_years_since'] = interactions['closed_stage'] * date_features['years_since_match']
    
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
    f.write(f"Reading data from: {ENHANCED_DATA_PATH}\n\n")

# Load the enhanced dataset
try:
    df = pd.read_excel(ENHANCED_DATA_PATH)
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Data loaded successfully. Shape: {df.shape}\n\n")
        f.write("Data Overview:\n")
        f.write(f"Total rows: {df.shape[0]}\n")
        f.write(f"Total columns: {df.shape[1]}\n\n")
except Exception as e:
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"Error loading enhanced data, falling back to original: {str(e)}\n")
    try:
        df = pd.read_excel(DATA_PATH)
        df = fill_missing_geographic_data(df)
        df = fill_missing_occupation_data(df)
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

# Models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=150, min_samples_leaf=2),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=150),
}

# Add a stacking ensemble
base_models = [
    ('rf', RandomForestRegressor(random_state=42, n_estimators=150)),
    ('gb', GradientBoostingRegressor(random_state=42, n_estimators=150))
]
models["Stacking Ensemble"] = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(),
    cv=5
)

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
    # Train the model on selected features
    model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    
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
    f.write(f"\nBest Model: {best_model_name}\n")
    f.write(f"RMSE: {best_rmse:.4f}\n")
    f.write(f"R²: {best_r2:.4f}\n\n")
    
    # Compare with previous results
    previous_rmse = 9.8005  # From the original date_features_results.txt
    previous_r2 = 0.7550    # From the original date_features_results.txt
    
    rmse_change = ((previous_rmse - best_rmse) / previous_rmse) * 100
    r2_change = ((best_r2 - previous_r2) / previous_r2) * 100
    
    f.write("Comparison with Original Model (9.8005 RMSE):\n")
    f.write(f"Original Best Model (Random Forest):\n")
    f.write(f"  - RMSE: {previous_rmse:.4f}\n")
    f.write(f"  - R²: {previous_r2:.4f}\n\n")
    
    f.write(f"Improvement:\n")
    f.write(f"  - RMSE: {rmse_change:.2f}% {'better' if rmse_change > 0 else 'worse'}\n")
    f.write(f"  - R²: {r2_change:.2f}% {'better' if r2_change > 0 else 'worse'}\n\n")
    
    f.write("Analysis complete.\n")

print(f"Analysis complete. Results saved to {OUTPUT_FILE}")
print(f"Feature importance plot saved to {FEATURE_IMPORTANCE_PATH}")
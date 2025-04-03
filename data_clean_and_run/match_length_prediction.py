import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib
import re
import sys
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

def log_message(message, log_file=None):
    """Print message to console and write to log file if provided"""
    print(message)
    if log_file:
        log_file.write(message + "\n")

def clean_column_name(col_name):
    """Clean column name by removing special characters and replacing spaces with underscores"""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '_', col_name)
    cleaned = re.sub(r'_+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

def clean_data(df):
    """Clean and validate the data"""
    # Clean column names
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Handle age-related issues
    df['Big_Age'] = df['Big_Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan)
    df['Little_Age'] = df['Little_Age'].apply(lambda x: x if 5 <= x <= 18 else np.nan)
    
    # Clean categorical variables
    categorical_cols = ['Program_Type', 'Big_Level_of_Education', 'Big_Gender', 
                       'Big_Race_Ethnicity', 'Big_Occupation', 'Big_Military',
                       'Big_Contact_Marital_Status', 'Little_Gender',
                       'Little_Participant_Race_Ethnicity']
    
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string and clean
            df[col] = df[col].astype(str).str.strip()
            # Replace empty strings and 'nan' with NaN
            df[col] = df[col].replace(['', 'nan', 'None', 'null'], np.nan)
    
    # Clean numeric variables
    numeric_cols = ['Days_Approval_to_Acceptance', 'Days_Application_to_Interview',
                   'Days_Interview_to_Acceptance', 'Days_Acceptance_to_Match']
    
    for col in numeric_cols:
        if col in df.columns:
            # Remove negative values
            df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)
            # Remove extreme outliers (values > 3 standard deviations)
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].apply(lambda x: x if abs(x - mean) <= 3*std else np.nan)
    
    # Clean location data
    if 'Big_County' in df.columns:
        df['Big_County'] = df['Big_County'].str.title().str.strip()
        # Remove any non-alphabetic characters
        df['Big_County'] = df['Big_County'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    return df

def process_age_features(df):
    """Process age-related features with enhanced validation"""
    # Convert birthdate columns to datetime
    date_columns = ['Big_Birthdate', 'Little_Birthdate']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate ages with validation
    current_year = datetime.now().year
    
    # Big age calculation with validation
    df['Big_Age'] = current_year - df['Big_Birthdate'].dt.year
    df['Big_Age'] = df['Big_Age'].apply(lambda x: x if 18 <= x <= 100 else np.nan)
    
    # Little age calculation with validation
    df['Little_Age'] = current_year - df['Little_Birthdate'].dt.year
    df['Little_Age'] = df['Little_Age'].apply(lambda x: x if 5 <= x <= 18 else np.nan)
    
    # Calculate age difference with validation
    df['Age_Difference'] = df['Big_Age'] - df['Little_Age']
    df['Age_Difference'] = df['Age_Difference'].apply(lambda x: x if 13 <= x <= 50 else np.nan)
    
    # Create more detailed age groups
    big_age_bins = [0, 22, 25, 30, 35, 40, 45, 50, 55, 100]
    big_age_labels = ['18-22', '23-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '55+']
    df['Big_Age_Group_Detailed'] = pd.cut(df['Big_Age'], bins=big_age_bins, labels=big_age_labels)
    
    little_age_bins = [0, 5, 8, 11, 14, 18]
    little_age_labels = ['0-5', '6-8', '9-11', '12-14', '15-18']
    df['Little_Age_Group'] = pd.cut(df['Little_Age'], bins=little_age_bins, labels=little_age_labels)
    
    # Create age combination groups
    df['Age_Group_Combination'] = df['Big_Age_Group_Detailed'].astype(str) + '_' + df['Little_Age_Group'].astype(str)
    
    # Calculate age-related ratios with validation
    df['Age_Ratio'] = np.where(
        (df['Big_Age'] > 0) & (df['Little_Age'] > 0),
        df['Big_Age'] / df['Little_Age'],
        np.nan
    )
    
    # Add age gap categories
    df['Age_Gap_Category'] = pd.cut(df['Age_Difference'], 
                                   bins=[0, 15, 20, 25, 30, 50],
                                   labels=['Small', 'Medium', 'Large', 'Very Large', 'Extreme'])
    
    return df

def process_date_features(df):
    """Create time-based features from date columns"""
    # Convert date columns to datetime
    date_columns = [
        'Big_Approved_Date', 'Big_Acceptance_Date', 'Big_Assessment_Uploaded',
        'Little_Application_Received', 'Little_Interview_Date', 'Little_Acceptance_Date',
        'Match_Activation_Date'
    ]
    
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Calculate time differences
    df['Days_Approval_to_Acceptance'] = (df['Big_Acceptance_Date'] - df['Big_Approved_Date']).dt.days
    df['Days_Application_to_Interview'] = (df['Little_Interview_Date'] - df['Little_Application_Received']).dt.days
    df['Days_Interview_to_Acceptance'] = (df['Little_Acceptance_Date'] - df['Little_Interview_Date']).dt.days
    df['Days_Acceptance_to_Match'] = (df['Match_Activation_Date'] - df['Big_Acceptance_Date']).dt.days
    
    return df

def process_location_features(df):
    """Process location-based features with enhanced categorization"""
    # Metro area counties (Minnesota specific)
    metro_counties = ['Hennepin', 'Ramsey', 'Dakota', 'Washington', 'Anoka', 'Scott', 'Carver']
    
    # Create location type features
    df['Is_Metro'] = df['Big_County'].isin(metro_counties)
    
    # Group counties by region (Minnesota specific)
    north_counties = ['St. Louis', 'Carlton', 'Lake', 'Cook', 'Itasca', 'Koochiching']
    south_counties = ['Olmsted', 'Rice', 'Goodhue', 'Steele', 'Waseca', 'Blue Earth']
    central_counties = ['Stearns', 'Wright', 'Sherburne', 'Benton', 'Morrison', 'Todd']
    
    df['County_Region'] = 'Other'
    df.loc[df['Big_County'].isin(metro_counties), 'County_Region'] = 'Metro'
    df.loc[df['Big_County'].isin(north_counties), 'County_Region'] = 'North'
    df.loc[df['Big_County'].isin(south_counties), 'County_Region'] = 'South'
    df.loc[df['Big_County'].isin(central_counties), 'County_Region'] = 'Central'
    
    # Create county population density groups
    urban_counties = metro_counties
    suburban_counties = ['Wright', 'Sherburne', 'Chisago', 'Isanti', 'Scott', 'Carver']
    
    df['County_Density'] = 'Rural'
    df.loc[df['Big_County'].isin(urban_counties), 'County_Density'] = 'Urban'
    df.loc[df['Big_County'].isin(suburban_counties), 'County_Density'] = 'Suburban'
    
    # Create location-program type interaction
    df['Location_Program'] = df['County_Region'] + '_' + df['Program_Type']
    
    # Create metro-nonmetro program type feature
    df['Metro_Program'] = df['Is_Metro'].astype(str) + '_' + df['Program_Type']
    
    return df

def process_seasonal_features(df):
    """Process seasonal features with enhanced event detection"""
    # Extract date components
    df['Match_Month'] = df['Match_Activation_Date'].dt.month
    df['Match_Day_Of_Week'] = df['Match_Activation_Date'].dt.dayofweek
    df['Match_Day_Of_Year'] = df['Match_Activation_Date'].dt.dayofyear
    
    # Define seasons
    df['Match_Season'] = pd.cut(df['Match_Month'], 
                               bins=[0, 3, 6, 9, 12], 
                               labels=['Winter', 'Spring', 'Summer', 'Fall'],
                               include_lowest=True)
    
    # School year indicators (Minnesota specific)
    df['Is_School_Start'] = df['Match_Month'].isin([8, 9])  # August-September
    df['Is_School_End'] = df['Match_Month'].isin([5, 6])    # May-June
    df['Is_Summer_Break'] = df['Match_Month'].isin([6, 7, 8])  # June-August
    
    # Holiday season indicators
    df['Is_Holiday_Season'] = df['Match_Month'].isin([11, 12, 1])  # November-January
    df['Is_Christmas_Season'] = df['Match_Month'].isin([12])  # December
    df['Is_New_Year'] = (df['Match_Month'] == 1) & (df['Match_Day_Of_Year'] <= 7)  # First week of January
    
    # Weekend vs weekday
    df['Is_Weekend_Match'] = df['Match_Day_Of_Week'].isin([5, 6])  # Saturday-Sunday
    
    # Create season-location combinations
    df['Season_Region'] = df['Match_Season'].astype(str) + '_' + df['County_Region'].astype(str)
    df['Season_Density'] = df['Match_Season'].astype(str) + '_' + df['County_Density'].astype(str)
    
    # Create holiday-location combinations
    df['Holiday_Region'] = df['Is_Holiday_Season'].astype(str) + '_' + df['County_Region'].astype(str)
    
    return df

def advanced_feature_engineering(df):
    """Create advanced features through transformations and interactions"""
    # Process location features
    df = process_location_features(df)
    
    # Process seasonal features
    df = process_seasonal_features(df)
    
    # Age-related features
    df['Age_Difference_Squared'] = df['Age_Difference'] ** 2
    
    # Time-based features with proper handling
    df['Total_Processing_Time'] = (
        df['Days_Approval_to_Acceptance'].fillna(0) + 
        df['Days_Application_to_Interview'].fillna(0) + 
        df['Days_Interview_to_Acceptance'].fillna(0) + 
        df['Days_Acceptance_to_Match'].fillna(0)
    )
    
    # Calculate Processing_Time_Ratio safely
    df['Processing_Time_Ratio'] = np.where(
        df['Total_Processing_Time'] > 0,
        df['Days_Acceptance_to_Match'].fillna(0) / df['Total_Processing_Time'],
        0
    )
    
    # Create binary features for important categories
    df['Is_Community_Program'] = df['Program_Type'] == 'Community'
    df['Is_Site_Based'] = df['Program_Type'] == 'Site'
    
    # Create features based on education level
    education_levels = {
        'High School Graduate': 1,
        'Some College': 2,
        'Associate Degree': 3,
        'Bachelors Degree': 4,
        'Masters Degree': 5,
        'Doctor of Medicine (MD)': 6,
        'Juris Doctorate (JD)': 6,
        'PHD': 6
    }
    df['Education_Level_Numeric'] = df['Big_Level_of_Education'].map(education_levels)
    
    # Create features based on military status
    df['Has_Military_Background'] = df['Big_Military'].isin(['Yes - Retired/Vet', 'Yes - Active'])
    
    # Create features based on race/ethnicity
    df['Is_Multiracial'] = df['Big_Race_Ethnicity'].str.contains(';', na=False)
    df['Is_White'] = df['Big_Race_Ethnicity'].str.contains('White', na=False)
    
    # Create features based on gender
    df['Same_Gender_Match'] = df['Big_Gender'] == df['Little_Gender']
    
    # Create features based on marital status
    df['Is_Married'] = df['Big_Contact_Marital_Status'] == 'Married'
    df['Is_Single'] = df['Big_Contact_Marital_Status'] == 'Single'
    
    # Create interaction features
    df['Age_Gap_Season'] = df['Age_Group_Combination'].astype(str) + '_' + df['Match_Season'].astype(str)
    df['Location_Program'] = df['County_Region'].astype(str) + '_' + df['Program_Type'].astype(str)
    
    # Create complex interaction features
    df['Age_Gap_Location'] = df['Age_Gap_Category'].astype(str) + '_' + df['County_Region'].astype(str)
    df['Education_Location'] = df['Education_Level_Numeric'].astype(str) + '_' + df['County_Region'].astype(str)
    df['Season_Program_Type'] = df['Match_Season'].astype(str) + '_' + df['Program_Type'].astype(str)
    
    # Replace any remaining infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def select_features(X, y, importance_threshold=0.01):
    """Select features based on Random Forest importance"""
    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline for feature selection
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', TargetEncoder())
            ]), categorical_cols)
        ])
    
    # Create a pipeline with preprocessing and random forest
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Get feature importances from the random forest
    rf_model = pipeline.named_steps['rf']
    
    # Get feature names after preprocessing
    feature_names = (
        list(numeric_cols) +
        list(categorical_cols)
    )
    
    # Create importance DataFrame
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    })
    
    # Sort by importance
    importances = importances.sort_values('Importance', ascending=False)
    
    # Select features above threshold
    selected_features = importances[importances['Importance'] > importance_threshold]['Feature'].tolist()
    
    return selected_features, importances

def preprocess_data(df, is_training=True, target_col='Match_Length'):
    """Preprocess the data for model training or prediction"""
    # Clean column names first
    df.columns = [clean_column_name(col) for col in df.columns]
    
    # Process age features first
    df = process_age_features(df)
    
    # Process date features
    df = process_date_features(df)
    
    # Now clean the data after features are created
    df = clean_data(df)
    
    # Apply advanced feature engineering
    df = advanced_feature_engineering(df)
    
    # Define columns to keep
    features_to_keep = [
        # Original features
        'Big_Age', 'Big_Age_Group_Detailed', 'Little_Age_Group', 'Age_Difference',
        'Little_Birthdate_year', 'Big_Level_of_Education', 'Big_Gender',
        'Big_Race_Ethnicity', 'Big_Occupation', 'Big_Military', 'Big_Car_Access',
        'Big_Open_to_Cross_Gender_Match', 'Big_Contact_Marital_Status',
        'Big_Contact_Volunteer_Availability', 'Big_County',
        'Days_Approval_to_Acceptance', 'Days_Application_to_Interview',
        'Days_Interview_to_Acceptance', 'Days_Acceptance_to_Match',
        'Program', 'Program_Type', 'Little_Gender',
        'Little_Participant_Race_Ethnicity',
        
        # Age-related features
        'Age_Difference_Squared', 'Age_Ratio', 'Age_Group_Combination',
        'Little_Age', 'Age_Gap_Category',
        
        # Location features
        'Is_Metro', 'County_Region', 'County_Density', 'Location_Program',
        'Metro_Program',
        
        # Seasonal features
        'Match_Month', 'Match_Season', 'Is_School_Start',
        'Is_School_End', 'Is_Summer_Break', 'Is_Holiday_Season',
        'Is_Weekend_Match', 'Season_Region', 'Season_Density',
        'Holiday_Region', 'Is_Christmas_Season', 'Is_New_Year',
        
        # Time features
        'Total_Processing_Time', 'Processing_Time_Ratio',
        
        # Program features
        'Is_Community_Program', 'Is_Site_Based',
        
        # Demographic features
        'Education_Level_Numeric', 'Has_Military_Background',
        'Is_Multiracial', 'Is_White', 'Same_Gender_Match',
        'Is_Married', 'Is_Single',
        
        # Interaction features
        'Age_Gap_Season', 'Age_Gap_Location', 'Education_Location',
        'Season_Program_Type'
    ]
    
    # Add target column if in training mode
    if is_training and target_col in df.columns:
        features_to_keep.append(target_col)
    
    # Keep only relevant columns
    df = df[features_to_keep]
    
    return df

def train_models(X, y, output_dir, log_file=None):
    """Train and evaluate models with feature selection and hyperparameter tuning"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', TargetEncoder())
            ]), categorical_cols)
        ])
    
    # Define cross-validation strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define parameter grids for each model
    rf_param_grid = {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [15, 20, 25],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__max_features': ['sqrt', 'log2', None]
    }
    
    xgb_param_grid = {
        'model__n_estimators': [200, 300, 400],
        'model__max_depth': [4, 6, 8],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.8, 0.9, 1.0],
        'model__min_child_weight': [1, 3, 5],
        'model__max_features': ['sqrt', 'log2', None]  # Added max_features for XGBoost
    }
    
    # Define base models
    base_models = {
        'Random Forest': (RandomForestRegressor(random_state=42), rf_param_grid),
        'XGBoost': (xgb.XGBRegressor(random_state=42), xgb_param_grid)
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    # Train and evaluate each model
    for name, (model, param_grid) in base_models.items():
        if log_file:
            log_message(f"\nTuning {name} hyperparameters...", log_file)
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_pipeline = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_pipeline.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': best_pipeline,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'best_params': grid_search.best_params_,
            'cv_rmse': -grid_search.best_score_  # Convert negative RMSE back to positive
        }
        
        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = name
        
        # Log results
        log_message(f"\n{name} Results:", log_file)
        log_message(f"Best Parameters: {grid_search.best_params_}", log_file)
        log_message(f"Cross-validation RMSE: {-grid_search.best_score_:.4f}", log_file)
        log_message(f"Test R2 Score: {r2:.4f}", log_file)
        log_message(f"Test RMSE: {rmse:.4f}", log_file)
        log_message(f"Test MSE: {mse:.4f}", log_file)
        
        # Get feature importances for both models
        if name in ['Random Forest', 'XGBoost']:
            model = best_pipeline.named_steps['model']
            # Get feature names after preprocessing
            feature_names = []
            for name, _, cols in best_pipeline.named_steps['preprocessor'].transformers_:
                if name == 'num':
                    feature_names.extend(cols)
                elif name == 'cat':
                    # For categorical features, we need to get the feature names after one-hot encoding
                    encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_features = encoder.get_feature_names_out(cols)
                        feature_names.extend(cat_features)
                    else:
                        feature_names.extend([f"{col}_{val}" for col in cols for val in encoder.categories_[0]])
            
            feature_importances = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            })
            feature_importances = feature_importances.sort_values('Importance', ascending=False)
            log_message(f"\nTop 20 Most Important Features for {name}:", log_file)
            log_message(feature_importances.head(20).to_string(), log_file)
    
    return results, best_model, X.columns.tolist()

def save_model_results(results, output_dir, log_file=None, selected_features=None):
    """Save model results to a text file"""
    results_path = os.path.join(output_dir, 'model_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("Model Training Results\n")
        f.write("=====================\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Find the best model
        best_model_name = max(results, key=lambda k: results[k]['r2'])
        
        f.write("Best Model: " + best_model_name + "\n")
        f.write("=====================\n")
        f.write(f"R2 Score: {results[best_model_name]['r2']:.4f}\n")
        f.write(f"RMSE: {results[best_model_name]['rmse']:.4f}\n")
        f.write(f"MSE: {results[best_model_name]['mse']:.4f}\n")
        f.write(f"Cross-validation RMSE: {results[best_model_name]['cv_rmse']:.4f}\n")
        f.write(f"Best Parameters: {results[best_model_name]['best_params']}\n\n")
        
        f.write("Detailed Results for All Models\n")
        f.write("=============================\n\n")
        
        for name, result in results.items():
            f.write(f"{name} Results:\n")
            f.write("-" * (len(name) + 9) + "\n")
            f.write(f"R2 Score: {result['r2']:.4f}\n")
            f.write(f"RMSE: {result['rmse']:.4f}\n")
            f.write(f"MSE: {result['mse']:.4f}\n")
            f.write(f"Cross-validation RMSE: {result['cv_rmse']:.4f}\n")
            f.write(f"Best Parameters: {result['best_params']}\n\n")
            
            # Save feature importance for Random Forest
            if name == 'Random Forest' and selected_features is not None:
                pipeline = result['model']
                rf_model = pipeline.named_steps['model']
                
                # Get feature names after preprocessing
                feature_names = []
                for name, _, cols in pipeline.named_steps['preprocessor'].transformers_:
                    if name == 'num':
                        feature_names.extend(cols)
                    elif name == 'cat':
                        # For categorical features, we need to get the feature names after one-hot encoding
                        encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
                        if hasattr(encoder, 'get_feature_names_out'):
                            cat_features = encoder.get_feature_names_out(cols)
                            feature_names.extend(cat_features)
                        else:
                            feature_names.extend([f"{col}_{val}" for col in cols for val in encoder.categories_[0]])
                
                feature_importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                f.write("Top 20 Most Important Features:\n")
                f.write("-" * 30 + "\n")
                f.write(feature_importances.head(20).to_string() + "\n\n")
    
    if log_file:
        log_message(f"\nDetailed results saved to {results_path}", log_file)

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_path = os.path.join(output_dir, 'model_training.log')
    with open(log_path, 'w') as log_file:
        log_message("Starting model training process...", log_file)
        log_message(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file)
        
        # File paths
        train_data_path = os.path.join(script_dir, 'filled_data.xlsx')
        
        # Check if file exists
        if not os.path.exists(train_data_path):
            log_message(f"Error: Training data file not found at {train_data_path}", log_file)
            sys.exit(1)
        
        # Load and preprocess training data
        log_message("\nLoading and preprocessing training data...", log_file)
        train_df = pd.read_excel(train_data_path)
        train_df = preprocess_data(train_df, is_training=True)
        
        # Separate features and target
        X = train_df.drop('Match_Length', axis=1)
        y = train_df['Match_Length']
        
        # Train models
        log_message("\nTraining models...", log_file)
        results, best_model, selected_features = train_models(X, y, output_dir, log_file)
        
        # Save detailed results to text file
        save_model_results(results, output_dir, log_file, selected_features)
        
        # Save the best model
        best_model_name = best_model
        best_model = results[best_model_name]['model']
        model_path = os.path.join(output_dir, 'best_model.joblib')
        joblib.dump(best_model, model_path)
        log_message(f"\nBest model ({best_model_name}) saved to {model_path}", log_file)
        
        # Make predictions on training data (since we don't have test data yet)
        log_message("\nMaking predictions on training data...", log_file)
        predictions = best_model.predict(X)
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Actual_Match_Length': y,
            'Predicted_Match_Length': predictions
        })
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        log_message(f"Predictions saved to {predictions_path}", log_file)
        
        log_message("\nProcess completed successfully!", log_file)

if __name__ == "__main__":
    main() 
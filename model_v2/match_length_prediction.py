import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import geopandas as gpd
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import silhouette_score
warnings.filterwarnings('ignore')

def select_features(X, y, importance_threshold=0.001):
    """Select features based on importance scores from both models"""
    print("Selecting important features...")
    
    # Train both models to get feature importance
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    
    # Get feature importance from both models
    rf_model.fit(X, y)
    xgb_model.fit(X, y)
    
    # Combine importance scores
    rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    
    # Normalize importance scores
    rf_importance = rf_importance / rf_importance.sum()
    xgb_importance = xgb_importance / xgb_importance.sum()
    
    # Average importance scores
    combined_importance = (rf_importance + xgb_importance) / 2
    
    # Select features above threshold
    selected_features = combined_importance[combined_importance > importance_threshold].index
    
    print(f"Selected {len(selected_features)} features out of {X.shape[1]} total features")
    print("\nTop 20 most important features:")
    print(combined_importance.nlargest(20))
    
    return X[selected_features]

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
        'Big ID',  # Identifier column
        # Remove highly missing columns
        'Big Level of Education',
        'Big Car Access',
        'Big Open to Cross-Gender Match',
        'Big Contact: Preferred Communication Type',
        'Big Contact: Former Big/Little',
        'Big Contact: Interest Finder - Sports',
        'Big Contact: Interest Finder - Places To Go',
        'Big Contact: Interest Finder - Hobbies',
        'Big Contact: Interest Finder - Entertainment',
        'Big Contact: Volunteer Availability',
        'Little Contact: Language(s) Spoken',
        'Little Contact: Interest Finder - Sports',
        'Little Contact: Interest Finder - Outdoors',
        'Little Contact: Interest Finder - Arts',
        'Little Contact: Interest Finder - Places To Go',
        'Little Contact: Interest Finder - Hobbies',
        'Little Contact: Interest Finder - Entertainment',
        'Little Contact: Interest Finder - Other Interests',
        'Little Contact: Interest Finder - Career',
        'Little Contact: Interest Finder - Personality',
        'Little Contact: Interest Finder - Three Wishes',
        'Big Employer/School Census Block Group',
        # Remove duplicate age columns
        'Big Age',  # Duplicate of Big_Age
        'Little Age'  # Duplicate of Little_Age
    ]
    
    # Drop columns
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
    
    # Get target variable
    y = df['Match Length'].copy()
    
    # Age-related feature engineering
    print("Creating age-related features...")
    
    # Calculate age difference
    X['Age_Difference'] = abs(X['Big_Age'] - X['Little_Age'])
    
    # Create age compatibility score (inverse of age difference, normalized)
    max_age_diff = X['Age_Difference'].max()
    X['Age_Compatibility_Score'] = 1 - (X['Age_Difference'] / max_age_diff)
    
    # Enhanced Demographic Feature Engineering
    print("Creating enhanced demographic features...")
    
    # Gender compatibility features
    X['Gender_Match'] = (X['Big Gender'] == X['Little Gender']).astype(int)
    
    # Race/Ethnicity compatibility features
    X['Race_Ethnicity_Match'] = (X['Big Race/Ethnicity'] == X['Little Participant: Race/Ethnicity']).astype(int)
    
    # Create simplified demographic score
    X['Demographic_Similarity_Score'] = (
        X['Gender_Match'] * 0.5 +
        X['Race_Ethnicity_Match'] * 0.5
    )
    
    # Geographic feature engineering
    print("Creating geographic features...")
    
    # Simple geographic features
    X['Same_Census_Block'] = (X['Big Home Census Block Group'] == X['Little Mailing Address Census Block Group']).astype(int)
    
    # County comparison features
    X['Same_County'] = (X['Big County'] == X['Little County']).astype(int)
    X['County_Match_Score'] = X['Same_County'] * 0.5 + X['Same_Census_Block'] * 0.5
    
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
    
    print(f"Initial feature count: {X.shape[1]}")
    
    # Return the full dataset without feature selection for clustering
    return X, y

def cluster_data(X, results_file, n_clusters=4):
    """
    Cluster the data into distinct match profiles
    """
    print(f"\nClustering data into {n_clusters} distinct match profiles...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Clustering Data into {n_clusters} Match Profiles\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    # Select core features for clustering (not using the target variable)
    clustering_features = [
        # Program characteristics
        'Program Type_Community', 'Program Type_School', 'Program_Maturity_Years', 
        
        # Match demographics
        'Age_Difference', 'Gender_Match', 'Race_Ethnicity_Match',
        'Big_Age', 'Little_Age', 'Demographic_Similarity_Score',
        
        # Geographic factors
        'Geographic_Distance', 'Same_County', 'Same_Census_Block',
        
        # Temporal factors
        'Time_of_Year_Score', 'Activation_Day_of_Week'
    ]
    
    # Filter existing columns that are in the dataset
    clustering_features = [col for col in clustering_features if col in X.columns]
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(X[clustering_features])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster)
    
    # Add cluster labels to the dataset
    X['Cluster'] = cluster_labels
    
    # Visualize clusters with PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)
    
    plt.figure(figsize=(10, 8))
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster_id}', alpha=0.7)
    plt.title('Match Profiles Clusters (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('cluster_visualization.png')
    
    # Analyze cluster characteristics
    cluster_analysis = pd.DataFrame()
    for cluster_id in range(n_clusters):
        cluster_data = X[X['Cluster'] == cluster_id]
        cluster_profile = {}
        
        # Calculate mean values for key features
        for feature in clustering_features:
            cluster_profile[feature] = cluster_data[feature].mean()
        
        # Add additional metadata
        cluster_profile['Size'] = len(cluster_data)
        cluster_profile['Proportion'] = len(cluster_data) / len(X)
        
        # If we have the target variable in this context
        cluster_profile['Average Match Length'] = df.loc[cluster_data.index, 'Match Length'].mean()
        
        # Add to analysis dataframe
        cluster_analysis[f'Cluster {cluster_id}'] = pd.Series(cluster_profile)
    
    # Write cluster analysis to results file
    with open(results_file, 'a') as f:
        f.write("Cluster Analysis:\n")
        f.write("-----------------\n\n")
        f.write(cluster_analysis.to_string())
        f.write("\n\n")
        
        # Describe clusters in words
        f.write("Cluster Descriptions:\n")
        f.write("--------------------\n\n")
        
        for cluster_id in range(n_clusters):
            f.write(f"Cluster {cluster_id}:\n")
            
            # Get top 5 defining features
            cluster_features = cluster_analysis[f'Cluster {cluster_id}'].sort_values(ascending=False).head(5)
            
            f.write(f"- Size: {cluster_analysis.loc['Size', f'Cluster {cluster_id}']:.0f} matches ")
            f.write(f"({cluster_analysis.loc['Proportion', f'Cluster {cluster_id}']:.1%} of total)\n")
            f.write(f"- Average Match Length: {cluster_analysis.loc['Average Match Length', f'Cluster {cluster_id}']:.1f} months\n")
            f.write("- Key characteristics:\n")
            
            for feature, value in cluster_features.items():
                if feature not in ['Size', 'Proportion', 'Average Match Length']:
                    f.write(f"  * {feature}: {value:.2f}\n")
            
            f.write("\n")
    
    print("Clustering complete. Results saved to file and visualization saved as 'cluster_visualization.png'")
    return X, cluster_labels

def train_cluster_models(X, y, cluster_labels, n_clusters, results_file):
    """
    Train separate models for each cluster
    """
    print("\nTraining cluster-specific models...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Training Cluster-Specific Models\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    # Store models for each cluster
    cluster_models = {}
    cluster_performances = {}
    
    # Track which clusters are too small
    small_clusters = []
    
    # Train models for each cluster
    for cluster_id in range(n_clusters):
        print(f"\nTraining models for Cluster {cluster_id}...")
        with open(results_file, 'a') as f:
            f.write(f"\nModels for Cluster {cluster_id}:\n")
            f.write(f"{'-'*30}\n")
        
        # Get data for this cluster
        cluster_mask = cluster_labels == cluster_id
        X_cluster = X[cluster_mask].drop(columns=['Cluster'])
        y_cluster = y[cluster_mask]
        
        # Skip if cluster is too small
        if len(X_cluster) < 50:
            print(f"Cluster {cluster_id} has only {len(X_cluster)} samples, skipping...")
            with open(results_file, 'a') as f:
                f.write(f"Cluster {cluster_id} has only {len(X_cluster)} samples, skipping...\n\n")
            small_clusters.append(cluster_id)
            continue
        
        # Apply feature selection for this specific cluster
        X_cluster = select_features(X_cluster, y_cluster)
        
        # Split data
        X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
            X_cluster, y_cluster, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=300, max_depth=20, min_samples_split=2, 
                min_samples_leaf=1, random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
            )
        }
        
        cluster_models[cluster_id] = {}
        cluster_performances[cluster_id] = {}
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"Training {model_name} for Cluster {cluster_id}...")
            
            # Train model
            model.fit(X_train_cluster, y_train_cluster)
            
            # Make predictions
            y_pred = model.predict(X_test_cluster)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_cluster, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_cluster, y_pred)
            
            # Store model and performance
            cluster_models[cluster_id][model_name] = model
            cluster_performances[cluster_id][model_name] = {
                'mse': mse, 'rmse': rmse, 'r2': r2
            }
            
            # Write results
            results = f"""
{model_name} Results for Cluster {cluster_id}:
----------------
Mean Squared Error: {mse:.2f}
Root Mean Squared Error: {rmse:.2f}
R2 Score: {r2:.2f}
Cluster Size: {len(X_cluster)}
"""
            print(results)
            with open(results_file, 'a') as f:
                f.write(results)
            
            # Write feature importance for RandomForest
            if model_name == "RandomForest":
                feature_importance = pd.DataFrame({
                    'feature': X_cluster.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_text = f"\nTop 10 most important features for Cluster {cluster_id}:\n"
                importance_text += feature_importance.head(10).to_string()
                importance_text += "\n\n"
                
                print(importance_text)
                with open(results_file, 'a') as f:
                    f.write(importance_text)
    
    # Write overall performance summary
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write("Overall Cluster Model Performance Summary:\n")
        f.write(f"{'-'*80}\n\n")
        
        # Create a summary table
        summary_data = []
        for cluster_id in cluster_performances:
            for model_name, metrics in cluster_performances[cluster_id].items():
                summary_data.append({
                    'Cluster': cluster_id,
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            f.write(summary_df.to_string(index=False))
            f.write("\n\n")
        
        # Write which clusters were skipped
        if small_clusters:
            f.write(f"Skipped clusters due to small size: {small_clusters}\n\n")
    
    return cluster_models, cluster_performances, small_clusters

def train_and_evaluate_baseline(X, y, results_file):
    """Train and evaluate baseline models for comparison"""
    print("\nTraining baseline models for comparison...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Training Baseline Models\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    # Apply feature selection
    X = select_features(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_split=2, 
            min_samples_leaf=1, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1
        )
    }
    
    baseline_models = {}
    baseline_performances = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and performance
        baseline_models[model_name] = model
        baseline_performances[model_name] = {
            'mse': mse, 'rmse': rmse, 'r2': r2
        }
        
        # Write results
        results = f"""
{model_name} Baseline Results:
----------------
Mean Squared Error: {mse:.2f}
Root Mean Squared Error: {rmse:.2f}
R2 Score: {r2:.2f}
"""
        print(results)
        with open(results_file, 'a') as f:
            f.write(results)
        
        # Write feature importance for RandomForest
        if model_name == "RandomForest":
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_text = "\nTop 20 most important features (baseline):\n"
            importance_text += feature_importance.head(20).to_string()
            importance_text += "\n\n"
            
            print(importance_text)
            with open(results_file, 'a') as f:
                f.write(importance_text)
    
    return baseline_models, baseline_performances

def evaluate_ensemble_predictions(X, y, cluster_labels, cluster_models, baseline_models, small_clusters, results_file):
    """Evaluate the ensemble of cluster-specific models"""
    print("\nEvaluating ensemble predictions...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Evaluating Ensemble of Cluster Models\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    # Get data for prediction
    X_pred = X.drop(columns=['Cluster'])
    
    # Split data consistently
    X_train, X_test, y_train, y_test = train_test_split(X_pred, y, test_size=0.2, random_state=42)
    
    # Get cluster assignments for test samples
    test_indices = X_test.index
    
    # Handle cluster_labels properly based on its type
    if isinstance(cluster_labels, np.ndarray):
        # If it's a NumPy array, we need to get the indices
        # We can map X's index to array positions
        indices_map = {idx: i for i, idx in enumerate(X.index)}
        test_clusters = np.array([cluster_labels[indices_map[idx]] for idx in test_indices])
    else:
        # If it's already a pandas Series/DataFrame
        test_clusters = cluster_labels.loc[test_indices].values
    
    # Track predictions by source
    prediction_sources = {'cluster_specific': 0, 'baseline': 0, 'fallback': 0}
    
    # Make predictions for each model type
    ensemble_predictions = {}
    
    for model_type in ['RandomForest', 'XGBoost']:
        print(f"\nMaking predictions with {model_type}...")
        
        # Initialize predictions array
        y_pred = np.zeros(len(y_test))
        
        # Reset prediction source counters
        prediction_sources = {'cluster_specific': 0, 'baseline': 0, 'fallback': 0}
        
        # Get baseline model
        baseline_model = baseline_models[model_type]
        
        # Make baseline predictions for all samples first (fallback)
        baseline_features = baseline_model.feature_names_in_
        baseline_X_test = X_test[baseline_features]
        baseline_predictions = baseline_model.predict(baseline_X_test)
        
        # Now try to use cluster-specific models where possible
        for i, (idx, row) in enumerate(X_test.iterrows()):
            cluster_id = int(test_clusters[i])
            
            # Skip small clusters or clusters without models
            if (cluster_id in small_clusters or 
                cluster_id not in cluster_models or 
                model_type not in cluster_models[cluster_id]):
                # Use baseline prediction
                y_pred[i] = baseline_predictions[i]
                prediction_sources['baseline'] += 1
                continue
            
            # Try to use the cluster-specific model
            try:
                # Get the model and its features
                cluster_model = cluster_models[cluster_id][model_type]
                model_features = cluster_model.feature_names_in_
                
                # Check if we have all required features
                if all(feature in X_test.columns for feature in model_features):
                    # Use cluster-specific model
                    cluster_X = row[model_features].values.reshape(1, -1)
                    y_pred[i] = cluster_model.predict(cluster_X)[0]
                    prediction_sources['cluster_specific'] += 1
                else:
                    # Fall back to baseline
                    y_pred[i] = baseline_predictions[i]
                    prediction_sources['baseline'] += 1
            except Exception as e:
                # Log the error and use baseline
                print(f"Error with cluster {cluster_id}: {str(e)}")
                y_pred[i] = baseline_predictions[i]
                prediction_sources['baseline'] += 1
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        ensemble_predictions[model_type] = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'prediction_sources': prediction_sources
        }
        
        # Write results
        results = f"""
{model_type} Ensemble Results:
----------------
Mean Squared Error: {mse:.2f}
Root Mean Squared Error: {rmse:.2f}
R2 Score: {r2:.2f}
Cluster-specific predictions: {prediction_sources['cluster_specific']}/{len(y_test)} ({prediction_sources['cluster_specific']/len(y_test)*100:.1f}%)
Baseline predictions: {prediction_sources['baseline']}/{len(y_test)} ({prediction_sources['baseline']/len(y_test)*100:.1f}%)
"""
        print(results)
        with open(results_file, 'a') as f:
            f.write(results)
    
    # Compare with baseline
    with open(results_file, 'a') as f:
        f.write("\nComparison: Ensemble vs. Baseline:\n")
        f.write("-------------------------------\n\n")
        
        comparison_data = []
        for model_type in ['RandomForest', 'XGBoost']:
            baseline_metrics = baseline_performances[model_type]
            ensemble_metrics = ensemble_predictions[model_type]
            
            rmse_diff = baseline_metrics['rmse'] - ensemble_metrics['rmse']
            r2_diff = ensemble_metrics['r2'] - baseline_metrics['r2']
            
            comparison_data.append({
                'Model': model_type,
                'Baseline RMSE': baseline_metrics['rmse'],
                'Ensemble RMSE': ensemble_metrics['rmse'],
                'RMSE Improvement': rmse_diff,
                'Baseline R2': baseline_metrics['r2'],
                'Ensemble R2': ensemble_metrics['r2'],
                'R2 Improvement': r2_diff
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
    
    return ensemble_predictions

def find_optimal_clusters(X, y, results_file, min_clusters=4, max_clusters=7):
    """
    Find the optimal number of clusters by evaluating model performance across different cluster counts
    """
    print(f"\nFinding optimal number of clusters ({min_clusters}-{max_clusters})...")
    
    # Write to results file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Finding Optimal Number of Clusters ({min_clusters}-{max_clusters})\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-'*80}\n\n")
    
    # Select core features for clustering
    clustering_features = [
        # Program characteristics
        'Program Type_Community', 'Program Type_School', 'Program_Maturity_Years', 
        
        # Match demographics
        'Age_Difference', 'Gender_Match', 'Race_Ethnicity_Match',
        'Big_Age', 'Little_Age', 'Demographic_Similarity_Score',
        
        # Geographic factors
        'Geographic_Distance', 'Same_County', 'Same_Census_Block',
        
        # Temporal factors
        'Time_of_Year_Score', 'Activation_Day_of_Week'
    ]
    
    # Filter existing columns that are in the dataset
    clustering_features = [col for col in clustering_features if col in X.columns]
    
    # Standardize features for clustering
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(X[clustering_features])
    
    # Track results for each cluster count
    cluster_results = []
    
    # Test different numbers of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        print(f"\nTesting with {n_clusters} clusters...")
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster)
        
        # Calculate silhouette score for clustering quality
        silhouette = silhouette_score(X_cluster, cluster_labels)
        
        # Add cluster labels to dataset
        X_with_clusters = X.copy()
        X_with_clusters['Cluster'] = cluster_labels
        
        # Get cluster statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_proportion = cluster_size / len(X)
            avg_match_length = y[cluster_mask].mean()
            
            cluster_stats[cluster_id] = {
                'size': cluster_size,
                'proportion': cluster_proportion,
                'avg_match_length': avg_match_length
            }
        
        # Train ensemble model with these clusters
        with open(results_file, 'a') as f:
            f.write(f"\nEvaluating {n_clusters} clusters:\n")
            f.write(f"Silhouette Score: {silhouette:.4f}\n\n")
            
            # Write cluster statistics
            f.write("Cluster Statistics:\n")
            for cluster_id, stats in cluster_stats.items():
                f.write(f"Cluster {cluster_id}: {stats['size']} samples ({stats['proportion']:.1%}), ")
                f.write(f"Avg Match Length: {stats['avg_match_length']:.2f} months\n")
            f.write("\n")
        
        # Train cluster-specific models
        cluster_models, cluster_performances, small_clusters = train_cluster_models(
            X_with_clusters, y, cluster_labels, n_clusters, results_file
        )
        
        # Train baseline models
        baseline_models, baseline_performances = train_and_evaluate_baseline(
            X.copy(), y, results_file
        )
        
        # Evaluate ensemble predictions
        ensemble_predictions = evaluate_ensemble_predictions(
            X_with_clusters, y, cluster_labels, cluster_models, 
            baseline_models, small_clusters, results_file
        )
        
        # Store results
        cluster_results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'rf_rmse': ensemble_predictions['RandomForest']['rmse'],
            'rf_r2': ensemble_predictions['RandomForest']['r2'],
            'xgb_rmse': ensemble_predictions['XGBoost']['rmse'],
            'xgb_r2': ensemble_predictions['XGBoost']['r2'],
            'small_clusters': len(small_clusters),
            'cluster_stats': cluster_stats
        })
    
    # Find best cluster count by RandomForest RMSE
    best_cluster = min(cluster_results, key=lambda x: x['rf_rmse'])
    
    # Write summary of results
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write("Cluster Optimization Summary:\n")
        f.write(f"{'-'*80}\n\n")
        
        f.write("Results by cluster count:\n")
        f.write("-----------------------\n\n")
        
        # Create summary table
        summary_data = []
        for result in cluster_results:
            summary_data.append({
                'n_clusters': result['n_clusters'],
                'silhouette': f"{result['silhouette']:.4f}",
                'RF_RMSE': f"{result['rf_rmse']:.2f}",
                'RF_R2': f"{result['rf_r2']:.2f}",
                'XGB_RMSE': f"{result['xgb_rmse']:.2f}",
                'XGB_R2': f"{result['xgb_r2']:.2f}",
                'small_clusters': result['small_clusters']
            })
        
        summary_df = pd.DataFrame(summary_data)
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        f.write(f"Best cluster count: {best_cluster['n_clusters']} (RF RMSE: {best_cluster['rf_rmse']:.2f})\n\n")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    cluster_counts = [r['n_clusters'] for r in cluster_results]
    rf_rmse = [r['rf_rmse'] for r in cluster_results]
    xgb_rmse = [r['xgb_rmse'] for r in cluster_results]
    silhouette = [r['silhouette'] for r in cluster_results]
    
    plt.subplot(2, 1, 1)
    plt.plot(cluster_counts, rf_rmse, 'o-', label='RandomForest RMSE')
    plt.plot(cluster_counts, xgb_rmse, 's-', label='XGBoost RMSE')
    plt.ylabel('RMSE')
    plt.title('Model Performance by Cluster Count')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(cluster_counts, silhouette, 'o-', color='green', label='Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Clustering Quality by Cluster Count')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_optimization.png')
    
    print(f"Cluster optimization complete. Results saved to file and visualization saved as 'cluster_optimization.png'")
    return best_cluster['n_clusters']

# Create results file with timestamp
results_file = f'cluster_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

# Initialize results file with header
with open(results_file, 'w') as f:
    f.write("Match Length Prediction with Cluster-Based Models\n")
    f.write("=============================================\n")
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
    f.write(f"Processed feature count: {X.shape[1]}\n")
    f.write(f"Target variable: Match Length\n\n")

# Find optimal number of clusters
optimal_clusters = find_optimal_clusters(X, y, results_file, min_clusters=4, max_clusters=7)

# Run full model with optimal cluster count
print(f"\nRunning full model with optimal cluster count: {optimal_clusters}")

# Cluster the data with optimal cluster count
X, cluster_labels = cluster_data(X, results_file, n_clusters=optimal_clusters)

# Train baseline models for comparison
baseline_models, baseline_performances = train_and_evaluate_baseline(X.drop(columns=['Cluster']), y, results_file)

# Train cluster-specific models
cluster_models, cluster_performances, small_clusters = train_cluster_models(X, y, cluster_labels, optimal_clusters, results_file)

# Evaluate ensemble predictions
ensemble_predictions = evaluate_ensemble_predictions(X, y, cluster_labels, cluster_models, baseline_models, small_clusters, results_file)

print(f"\nCluster-based modeling complete! Results saved to: {results_file}") 
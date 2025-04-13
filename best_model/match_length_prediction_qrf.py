import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

def cluster_data(X, y, results_file, n_clusters=4):
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
        
        # Calculate average match length for this cluster
        cluster_indices = cluster_data.index
        cluster_profile['Average Match Length'] = y.loc[cluster_indices].mean()
        
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

def predict_quantiles(rf_model, X, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """
    Get quantile predictions from a random forest by using predictions from individual trees
    """
    # Get predictions from individual trees
    tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    
    # Calculate quantiles across tree predictions for each sample
    quantile_preds = {}
    for q in quantiles:
        quantile_preds[q] = np.quantile(tree_preds, q, axis=0)
    
    return quantile_preds

def train_and_evaluate_qrf(X, y, cluster_labels, results_file):
    """
    Train and evaluate ONLY Cluster-Specific Quantile Regression Forests
    """
    print("\nTraining and evaluating Cluster-Specific Quantile Regression Forests...")
    
    # Write to both console and file
    with open(results_file, 'a') as f:
        f.write(f"\n{'-'*80}\n")
        f.write(f"Evaluating Cluster-Specific Quantile Regression Forests\n")
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
        indices_map = {idx: i for i, idx in enumerate(X.index)}
        test_clusters = np.array([cluster_labels[indices_map[idx]] for idx in test_indices])
    else:
        test_clusters = cluster_labels.loc[test_indices].values
    
    # Define quantiles to predict
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Train cluster-specific models
    print("\nTraining cluster-specific random forests...")
    cluster_rf_models = {}
    
    # Get all unique cluster IDs
    unique_clusters = sorted(list(set(cluster_labels)))
    
    # Train a RF for each cluster
    for cluster_id in unique_clusters:
        # Get data for this cluster
        cluster_mask = X['Cluster'] == cluster_id
        X_cluster = X.loc[cluster_mask].drop(columns=['Cluster'])
        y_cluster = y.loc[cluster_mask]
        
        # Skip if cluster is too small
        if len(X_cluster) < 50:
            print(f"Cluster {cluster_id} has only {len(X_cluster)} samples, skipping...")
            continue
        
        print(f"Training random forest for Cluster {cluster_id}...")
        
        # Create and train a RF for this cluster
        cluster_rf = RandomForestRegressor(
            n_estimators=300,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        cluster_rf.fit(X_cluster, y_cluster)
        
        # Store the model
        cluster_rf_models[cluster_id] = cluster_rf
    
    # Make predictions for test set using cluster-specific models
    cluster_predictions = {}
    for q in quantiles:
        cluster_predictions[q] = np.zeros(len(y_test))
    
    # Also store mean predictions
    cluster_mean_pred = np.zeros(len(y_test))
    
    # Train a simple fallback model for any clusters without a specific model
    fallback_rf = RandomForestRegressor(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    fallback_rf.fit(X_train, y_train)
    
    # Track which predictions used which model
    prediction_sources = {'cluster_specific': 0, 'fallback': 0}
    
    for i, cluster_id in enumerate(test_clusters):
        # Skip if cluster has no model
        if cluster_id not in cluster_rf_models:
            # Use fallback model predictions
            X_sample = X_test.iloc[[i]]
            cluster_mean_pred[i] = fallback_rf.predict(X_sample)[0]
            
            # Get quantile predictions from fallback model
            tree_preds = np.array([tree.predict(X_sample) for tree in fallback_rf.estimators_])
            for q in quantiles:
                cluster_predictions[q][i] = np.quantile(tree_preds, q)
                
            prediction_sources['fallback'] += 1
            continue
        
        # Get the RF model for this cluster
        cluster_rf = cluster_rf_models[cluster_id]
        
        # Extract features for this sample
        X_sample = X_test.iloc[[i]]
        
        # Get mean prediction
        cluster_mean_pred[i] = cluster_rf.predict(X_sample)[0]
        
        # Get quantile predictions
        sample_quantiles = predict_quantiles(cluster_rf, X_sample, quantiles)
        for q in quantiles:
            cluster_predictions[q][i] = sample_quantiles[q][0]
            
        prediction_sources['cluster_specific'] += 1
    
    # Calculate metrics for cluster-specific predictions
    cluster_mse = mean_squared_error(y_test, cluster_mean_pred)
    cluster_rmse = np.sqrt(cluster_mse)
    cluster_r2 = r2_score(y_test, cluster_mean_pred)
    
    # Calculate interval coverage
    cluster_coverage = np.mean((y_test >= cluster_predictions[0.1]) & (y_test <= cluster_predictions[0.9]))
    cluster_coverage_75 = np.mean((y_test >= cluster_predictions[0.25]) & (y_test <= cluster_predictions[0.75]))
    
    # Calculate interval width
    cluster_interval_width = np.mean(cluster_predictions[0.9] - cluster_predictions[0.1])
    cluster_interval_width_75 = np.mean(cluster_predictions[0.75] - cluster_predictions[0.25])
    
    # Write cluster-specific model results
    cluster_results = f"""
Cluster-Specific Quantile Regression Forest Results:
--------------------------------------------------
Mean Squared Error: {cluster_mse:.2f}
Root Mean Squared Error: {cluster_rmse:.2f}
R2 Score: {cluster_r2:.2f}
80% Prediction Interval Coverage: {cluster_coverage:.3f} (ideal: 0.800)
50% Prediction Interval Coverage: {cluster_coverage_75:.3f} (ideal: 0.500)
80% Prediction Interval Width: {cluster_interval_width:.2f} months
50% Prediction Interval Width: {cluster_interval_width_75:.2f} months
Cluster-specific predictions: {prediction_sources['cluster_specific']}/{len(y_test)} ({prediction_sources['cluster_specific']/len(y_test)*100:.1f}%)
Fallback predictions: {prediction_sources['fallback']}/{len(y_test)} ({prediction_sources['fallback']/len(y_test)*100:.1f}%)
"""
    
    print(cluster_results)
    with open(results_file, 'a') as f:
        f.write(cluster_results)
    
    # Examine prediction errors by actual match length
    error_by_length = pd.DataFrame({
        'actual': y_test,
        'predicted': cluster_mean_pred,
        'error': np.abs(y_test - cluster_mean_pred),
        'lower_bound': cluster_predictions[0.1],
        'upper_bound': cluster_predictions[0.9],
        'interval_width': cluster_predictions[0.9] - cluster_predictions[0.1],
        'in_interval': (y_test >= cluster_predictions[0.1]) & (y_test <= cluster_predictions[0.9])
    })
    
    # Group by match length bins
    bins = [0, 6, 12, 24, 36, 100]
    labels = ['0-6 months', '6-12 months', '12-24 months', '24-36 months', '36+ months']
    error_by_length['length_bin'] = pd.cut(error_by_length['actual'], bins=bins, labels=labels)
    
    # Calculate metrics by length bin
    bin_metrics = error_by_length.groupby('length_bin').agg({
        'error': 'mean',
        'in_interval': 'mean',
        'interval_width': 'mean',
        'actual': 'count'
    }).rename(columns={
        'error': 'Mean Absolute Error',
        'in_interval': 'Interval Coverage',
        'interval_width': 'Interval Width',
        'actual': 'Count'
    })
    
    # Write bin metrics
    with open(results_file, 'a') as f:
        f.write("\nPerformance by Match Length Category:\n")
        f.write("---------------------------------\n\n")
        f.write(bin_metrics.to_string())
        f.write("\n\n")
    
    print("Performance by match length category:")
    print(bin_metrics)
    
    # Create feature importance visualization for the cluster models
    plt.figure(figsize=(12, 8))
    
    # Get feature importance from one of the cluster models (if available)
    if cluster_rf_models:
        # Get the first available cluster model
        first_cluster_id = list(cluster_rf_models.keys())[0]
        model = cluster_rf_models[first_cluster_id]
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X_cluster.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)
        
        # Plot feature importance
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top 20 Feature Importance for Cluster {first_cluster_id}')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        # Write feature importance to file
        with open(results_file, 'a') as f:
            f.write("\nTop 20 Feature Importance:\n")
            f.write("--------------------------\n\n")
            f.write(feature_importance.to_string())
            f.write("\n\n")
    
    # Return the results
    return {
        'cluster': {
            'rmse': cluster_rmse,
            'r2': cluster_r2,
            'coverage': cluster_coverage,
            'interval_width': cluster_interval_width
        },
        'bin_metrics': bin_metrics,
        'models': cluster_rf_models
    }

def save_model(models, filename='cluster_qrf_models.pkl'):
    """Save the trained models to a file"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to {filename}")

def main():
    # Create results file with timestamp
    results_file = f'qrf_model_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

    # Initialize results file with header
    with open(results_file, 'w') as f:
        f.write("Match Length Prediction with Cluster-Specific Quantile Regression Forests\n")
        f.write("==============================================================\n")
        f.write(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load the enhanced data
    print("Loading data...")
    try:
        # Try the absolute path first
        df = pd.read_excel('/Users/dameesh/Desktop/MinMUDAC2025/best_model/enhanced_data.xlsx')
    except FileNotFoundError:
        # Fall back to relative path
        try:
            df = pd.read_excel('enhanced_data.xlsx')
        except FileNotFoundError:
            # Try one directory up
            df = pd.read_excel('../enhanced_data.xlsx')

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

    # Cluster the data - pass y as well
    n_clusters = 4  # Optimal number determined by analysis
    X, cluster_labels = cluster_data(X, y, results_file, n_clusters=n_clusters)

    # Train and evaluate QRF models (only cluster-specific)
    qrf_results = train_and_evaluate_qrf(X, y, cluster_labels, results_file)
    
    # Save the trained models
    save_model(qrf_results['models'])

    print(f"\nCluster-Specific Quantile Regression Forest modeling complete! Results saved to: {results_file}")
    
    # Return a summary of the results
    return {
        'cluster_rmse': qrf_results['cluster']['rmse'],
        'r2_score': qrf_results['cluster']['r2'],
        'coverage': qrf_results['cluster']['coverage'],
        'results_file': results_file
    }

if __name__ == "__main__":
    main() 
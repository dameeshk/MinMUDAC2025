# Match Length Prediction Model

This tool evaluates several machine learning models to predict the match length in the BBBS dataset.

## Setup and Requirements

1. Make sure you have Python 3.8+ installed
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model Evaluation

Run the script from the command line:

```bash
python model_evaluation.py
```

The script:
1. Reads the Novice.xlsx dataset from the Data folder
2. Preprocesses the data (handles missing values, encodes categorical features, etc.)
3. Evaluates 5 regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Random Forest Regression
   - Gradient Boosting Regression
4. Performs hyperparameter tuning using grid search
5. Calculates RMSE (Root Mean Squared Error) and R² scores for each model
6. Ranks models based on performance
7. Generates visualizations in a 'visualizations' subfolder

## Output

The results are saved to:
- `model_results.txt`: Contains detailed metrics for all models and preprocessing steps
- `visualizations/`: Contains performance comparison charts and predicted vs actual plots

## Interpretation

RMSE (Root Mean Squared Error) is the primary metric for evaluation. Lower RMSE values indicate better model performance. 
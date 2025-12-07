import sys
import os
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)

from random_forest_model import load_and_clean_data, run_pipeline_pipeline5

print("=== TRAINING 0.65 SCORE MODEL ===\n")

# Load data
data_path = os.path.join(backend_dir, "data", "immovlan_cleaned_file_final.csv")
df = load_and_clean_data(data_path)

print(f"Data loaded: {len(df)} rows\n")

# Train model
model, X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline_pipeline5(
    df, 
    target_col='Price',
    feature_eng=True,
    rare_city=True,
    train_mode=True
)

# Save model
models_dir = os.path.join(backend_dir, "models")
joblib.dump(model, os.path.join(models_dir, "random_forest_065.pkl"))
joblib.dump(X_train.columns.tolist(), os.path.join(models_dir, "rf_train_columns_065.pkl"))

print("\n" + "="*70)
print("✅ NEW MODEL (0.65 SCORE) SAVED")
print("="*70)

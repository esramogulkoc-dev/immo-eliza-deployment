import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from predict import load_model_and_columns, get_prediction, PredictInput

# -------------------------------
# File paths
# -------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "../data")
TEST_FILE = os.path.join(DATA_DIR, "test_data.csv")

# -------------------------------
# Load model
# -------------------------------
if not load_model_and_columns():
    print("❌ Model or columns could not be loaded!")
    sys.exit(1)

print("\n--- Starting Model Test ---\n")

# -------------------------------
# Read test data
# -------------------------------
if not os.path.exists(TEST_FILE):
    print(f"❌ Test data not found: {TEST_FILE}")
    sys.exit(1)

df_test = pd.read_csv(TEST_FILE)

# Check if true values exist
if 'Price' not in df_test.columns:
    print("❌ 'Price' column not found in CSV!")
    sys.exit(1)

y_true = df_test['Price'].values
predictions = []

# -------------------------------
# Calculate predictions
# -------------------------------
for idx, row in df_test.iterrows():
    data_input = PredictInput(
        Livable_surface=row.get("Livable_surface", 0),
        Number_of_bedrooms=row.get("Number_of_bedrooms", 1),
        Total_land_surface=row.get("Total_land_surface", 0),
        postal_code=row.get("postal_code", 0),
        has_terrace=bool(row.get("has_terrace", 0)),
        has_garage=bool(row.get("has_garage", 0)),
        region=row.get("region", ""),
        type_of_heating=row.get("type_of_heating", ""),
        epc_rating=row.get("epc_rating", ""),
        type_of_property=row.get("type_of_property", "")
    )
    try:
        pred = get_prediction(data_input)
        predictions.append(pred)
        print(f"Test {idx+1}: Predicted price → € {pred:,.2f}")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        predictions.append(np.nan)

# -------------------------------
# Evaluate model performance
# -------------------------------
mask = ~np.isnan(predictions)
y_true_masked = y_true[mask]
y_pred_masked = np.array(predictions)[mask]

r2 = r2_score(y_true_masked, y_pred_masked)
rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
mae = mean_absolute_error(y_true_masked, y_pred_masked)

print("\n--- Model Performance Summary ---")
print(f"R² Score  : {r2:.4f}")
print(f"RMSE      : € {rmse:,.2f}")
print(f"MAE       : € {mae:,.2f}")
print("\n✅ Test completed.")

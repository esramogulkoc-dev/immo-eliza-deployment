import joblib
import os
import sys
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional
import traceback

# ============================================================
# FILE PATHS
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(CURRENT_DIR)

MODEL_FILE = os.path.join(BACKEND_DIR, "models", "random_forest_065.pkl")
COL_FILE = os.path.join(BACKEND_DIR, "models", "rf_train_columns_065.pkl")

print(f"API Import Started: {os.path.abspath(__file__)}", file=sys.stdout)

rf_model = None
train_columns = None

# ============================================================
# Pydantic Model - Receives Simple Input from Streamlit
# ============================================================
class PredictInput(BaseModel):
    # Basic features coming from Streamlit
    Livable_surface: Optional[float] = Field(None)
    Number_of_bedrooms: Optional[int] = Field(None)
    Total_land_surface: Optional[float] = Field(None)
    postal_code: Optional[int] = Field(None)
    
    has_terrace: Optional[bool] = False
    has_garage: Optional[bool] = False
    region: Optional[str] = Field(default="")
    type_of_heating: Optional[str] = Field(default="")
    epc_rating: Optional[str] = Field(default="")
    type_of_property: Optional[str] = Field(default="")

# ============================================================
# MODEL LOADING
# ============================================================
def load_model_and_columns():
    global rf_model
    global train_columns

    if rf_model is not None and train_columns is not None:
        print("Models already loaded.", file=sys.stdout)
        return True

    try:
        print(f"Model file path: {MODEL_FILE}")
        print(f"Column file path: {COL_FILE}")
        if os.path.exists(MODEL_FILE):
            print("✅ Model file found")
        else:
            print("❌ Model file not found!")

        if os.path.exists(COL_FILE):
            print("✅ Column file found")
        else:
            print("❌ Column file not found!")

        rf_model = joblib.load(MODEL_FILE)
        print("✅ Model loading completed")
        train_columns = joblib.load(COL_FILE)
        print(f"✅ Columns loading completed, total {len(train_columns)} columns")
        return True
    except Exception as e:
        print(f"!!! Model loading error: {e} !!!", file=sys.stderr)
        return False

# ============================================================
# PREDICTION - HYBRID (Simple input, Full features)
# ============================================================
def get_prediction(data: PredictInput) -> float:
    """
    Receives simple data from Streamlit, prepares full features for 0.65 model
    """
    global rf_model
    global train_columns
    
    if rf_model is None or train_columns is None:
        raise ValueError("Model could not be loaded!")

    input_dict = data.model_dump()
    
    # Empty DataFrame with all columns of the 0.65 model
    model_df = pd.DataFrame(0, index=[0], columns=train_columns)

    try:
        # ----------------------------------------------------------------
        # 1. BASIC NUMERIC FEATURES (From Streamlit)
        # ----------------------------------------------------------------
        livable = float(input_dict.get('Livable_surface') or 0)
        bedrooms = max(int(input_dict.get('Number_of_bedrooms') or 1), 1)
        land = float(input_dict.get('Total_land_surface') or 0)
        postal = int(input_dict.get('postal_code') or 0)
        
        if 'Livable surface' in model_df.columns:
            model_df['Livable surface'] = livable
        if 'Number of bedrooms' in model_df.columns:
            model_df['Number of bedrooms'] = bedrooms
        if 'Total land surface' in model_df.columns:
            model_df['Total land surface'] = land
        if 'postal_code' in model_df.columns:
            model_df['postal_code'] = postal
        
        # ----------------------------------------------------------------
        # 2. Boolean → Numeric Features
        # ----------------------------------------------------------------
        has_garage = input_dict.get('has_garage', False)
        has_terrace = input_dict.get('has_terrace', False)
        
        if 'Garage' in model_df.columns:
            model_df['Garage'] = 1 if has_garage else 0
        if 'Number of garages' in model_df.columns:
            model_df['Number of garages'] = 1 if has_garage else 0
        if 'Terrace' in model_df.columns:
            model_df['Terrace'] = 1 if has_terrace else 0
        
        # Default values (not in Streamlit but required by model)
        if 'Elevator' in model_df.columns:
            model_df['Elevator'] = 0
        if 'Garden' in model_df.columns:
            model_df['Garden'] = 0
        if 'Surface terrace' in model_df.columns:
            model_df['Surface terrace'] = 0
        if 'Swimming pool' in model_df.columns:
            model_df['Swimming pool'] = 0
        
        # ----------------------------------------------------------------
        # 3. FEATURE ENGINEERING (Used by 0.65 model)
        # ----------------------------------------------------------------
        if 'surface_ratio' in model_df.columns:
            model_df['surface_ratio'] = livable / land if land > 0 else 1.0
        
        if 'area_per_bedroom' in model_df.columns:
            model_df['area_per_bedroom'] = livable / bedrooms
        
        if 'has_swimming_pool' in model_df.columns:
            model_df['has_swimming_pool'] = 0
        
        if 'has_garden' in model_df.columns:
            model_df['has_garden'] = 0
        
        if 'has_terrace' in model_df.columns:
            model_df['has_terrace'] = 1 if has_terrace else 0
        
        # ----------------------------------------------------------------
        # 4. CATEGORICAL FEATURES (One-Hot Encoded)
        # ----------------------------------------------------------------
        
        # State of property → Default "Normal"
        state_col = 'State of the property_Normal'
        if state_col in model_df.columns:
            model_df[state_col] = 1
        
        # Type of heating (from Streamlit)
        heating = input_dict.get('type_of_heating', '').strip()
        if heating:
            # Capitalize first letter (gas → Gas)
            heating_formatted = heating.capitalize()
            heating_col = f'Type of heating_{heating_formatted}'
            if heating_col in model_df.columns:
                model_df[heating_col] = 1
        
        # Region (from Streamlit)
        region = input_dict.get('region', '').strip()
        if region:
            region_formatted = region.capitalize()  # flanders → Flanders
            region_col = f'Region_{region_formatted}'
            if region_col in model_df.columns:
                model_df[region_col] = 1
        
        # Type (apartment/house) (from Streamlit)
        property_type = input_dict.get('type_of_property', '').strip().lower()
        if property_type:
            type_col = f'type_{property_type}'
            if type_col in model_df.columns:
                model_df[type_col] = 1
        
        # Main type → derived from type_of_property
        if property_type:
            main_type_col = f'main_type_{property_type}'
            if main_type_col in model_df.columns:
                model_df[main_type_col] = 1
        
        # City → cannot derive from postal_code, use "other"
        if 'city_other' in model_df.columns:
            model_df['city_other'] = 1
        
        # Province → unknown, leave as 0
        
        # ----------------------------------------------------------------
        # 5. PREDICTION
        # ----------------------------------------------------------------
        model_df = model_df.reindex(columns=train_columns, fill_value=0).astype(float)
        
        prediction = rf_model.predict(model_df)[0]
        
        # Validity check
        if np.isinf(prediction) or np.isnan(prediction) or prediction < 0:
            return 250000.0
        
        return round(float(prediction), 2)

    except Exception as e:
        print(f"\n!!! PREDICTION ERROR: {e} !!!", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise ValueError("Prediction error.")

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, Any

# ✅ NEW: Load postal code list from CSV
@st.cache_data
def load_postal_codes():
    """Load unique postal codes and city information from CSV"""
    try:
        df = pd.read_csv("Backend/data/immovlan_cleaned_file_final.csv")
        
        # Postal code and city combination
        postal_city = df[['postal_code', 'city']].drop_duplicates().dropna()
        postal_city = postal_city.sort_values('postal_code')
        
        # Create dictionary: {postal_code: city_name}
        postal_dict = dict(zip(
            postal_city['postal_code'].astype(int), 
            postal_city['city'].str.title()
        ))
        
        return postal_dict
    except Exception as e:
        st.warning(f"CSV could not be loaded, using default values: {e}")
        # Fallback: popular cities
        return {
            1000: "Brussels", 1050: "Ixelles", 1070: "Anderlecht",
            2000: "Antwerp", 9000: "Gent", 3000: "Leuven",
            4000: "Liège", 8000: "Bruges", 5000: "Namur"
        }

# Load postal codes
postal_dict = load_postal_codes()
postal_codes_list = sorted(postal_dict.keys())

# ------------------------------------
# Interface Configuration
# ------------------------------------
st.set_page_config(
    page_title="Immo Eliza Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏡 Immo Eliza Real Estate Price Prediction")
st.markdown("Calculate the estimated property price based on Belgian real estate data.")

# API URL
API_BASE_URL = "https://immo-eliza-deployment-vgfy.onrender.com/predict"
API_PREDICT_URL = f"{API_BASE_URL}/predict"

# Backend health check
def check_backend_health():
    try:
        response = requests.get(API_BASE_URL, timeout=2)
        if response.status_code == 200:
            return True, "✅ Backend is active"
        return False, f"⚠️ Backend is not responding (Status: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return False, "❌ Cannot connect to backend!"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

# ------------------------------------
# Sidebar: Input Fields
# ------------------------------------
st.sidebar.header("🔍 Property Features")

# Backend status
is_connected, status_msg = check_backend_health()
if is_connected:
    st.sidebar.success(status_msg)
else:
    st.sidebar.error(status_msg)
    st.sidebar.info("""
    **To start the backend:**
    ```bash
    cd Backend/api
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    ```
    """)

st.sidebar.divider()

# Numeric Inputs
livable_surface = st.sidebar.number_input(
    "Livable Surface Area (m²)",
    min_value=10,
    max_value=1000,
    value=150,
    step=5,
    help="Total heated and livable area."
)

number_of_bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1,
    help="Total number of bedrooms in the property."
)

total_land_surface = st.sidebar.number_input(
    "Total Land Area (m²)",
    min_value=0,
    max_value=5000,
    value=500,
    step=50,
    help="Total land area including garden, terrace, and building. Usually 0 for apartments."
)

# ✅ CURRENT: Postal Code Dropdown
postal_code = st.sidebar.selectbox(
    "📍 Postal Code - City",
    options=postal_codes_list,
    index=0,
    format_func=lambda x: f"{x} - {postal_dict.get(x, 'Unknown')}",
    help=f"Select from {len(postal_codes_list)} different postal codes in Belgium"
)

# Categorical Selections
property_type = st.sidebar.selectbox(
    "Property Type",
    options=["house", "apartment", "villa", "land"],
    format_func=lambda x: x.title(),
    index=0
)

region = st.sidebar.selectbox(
    "Region",
    options=["flanders", "wallonia", "brussels"],
    format_func=lambda x: x.title(),
    index=0
)

type_of_heating = st.sidebar.selectbox(
    "Heating Type",
    options=["gas", "oil", "electricity", "solar", "fueloil"],
    format_func=lambda x: x.title(),
    index=0
)

epc_rating = st.sidebar.selectbox(
    "EPC Rating (Energy Performance Certificate)",
    options=["a", "b", "c", "d", "e", "f", "g"],
    format_func=lambda x: x.upper(),
    index=2,
    help="A is best, G is worst energy performance."
)

# Boolean Features
st.sidebar.subheader("Additional Features")
has_terrace = st.sidebar.checkbox("Has Terrace?", value=True)
has_garage = st.sidebar.checkbox("Has Garage?", value=True)

# ------------------------------------
# Prediction and Result Section
# ------------------------------------
st.header("📊 Prediction Result")

# Simple validation
all_required_inputs_are_valid = (
    livable_surface is not None and livable_surface > 0 and
    number_of_bedrooms is not None and number_of_bedrooms > 0
)

if st.button("🔮 Predict", type="primary", use_container_width=True):
    if not all_required_inputs_are_valid:
        st.error("Please fill in mandatory fields like Livable Surface Area and Number of Bedrooms with positive values.")
    elif not is_connected:
        st.error("❌ Cannot connect to backend! Ensure FastAPI is running.")
    else:
        # Prepare data to send to API
        data = {
            "Livable_surface": float(livable_surface),
            "Number_of_bedrooms": int(number_of_bedrooms),
            "Total_land_surface": float(total_land_surface),
            "postal_code": int(postal_code),
            "has_terrace": bool(has_terrace),
            "has_garage": bool(has_garage),
            "region": region,
            "type_of_heating": type_of_heating,
            "epc_rating": epc_rating,
            "type_of_property": property_type
        }

        # --- DEBUG SECTION ---
        with st.expander("🔍 Data Sent to API (Debug)"):
            st.json(data)
            st.code(f"URL: {API_PREDICT_URL}")
        # ----------------------

        with st.spinner("Predicting..."):
            try:
                # API request
                response = requests.post(
                    API_PREDICT_URL, 
                    json=data,
                    headers={"Content-Type": "application/json"},
                    timeout=20
                )

                if response.status_code == 200:
                    result = response.json()
                    price = result.get("predicted_price", 0)
                    
                    # Show price
                    st.success("✅ Prediction successful!")
                    
                    # Highlighted price display
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric(
                            label="💰 Predicted Price",
                            value=f"€ {price:,.2f}",
                            delta=None
                        )
                    
                    st.balloons()
                    
                    # Detailed summary
                    with st.expander("📋 Property Features Summary"):
                        city_name = postal_dict.get(postal_code, "Unknown")
                        st.markdown(f"""
                        - **Property Type:** {property_type.title()}
                        - **Region:** {region.title()}
                        - **Location:** {postal_code} - {city_name}
                        - **Livable Area:** {livable_surface} m²
                        - **Bedrooms:** {number_of_bedrooms}
                        - **Land Area:** {total_land_surface} m²
                        - **Heating:** {type_of_heating.title()}
                        - **EPC:** {epc_rating.upper()}
                        - **Terrace:** {'✅ Yes' if has_terrace else '❌ No'}
                        - **Garage:** {'✅ Yes' if has_garage else '❌ No'}
                        """)

                elif response.status_code == 422:
                    st.error("❌ Data format error! (Validation Error)")
                    with st.expander("Error Details"):
                        try:
                            error_json = response.json()
                            st.json(error_json)
                        except:
                            st.code(response.text)
                    st.warning("Data format does not match backend expectation.")
                    
                else:
                    # Other API errors
                    st.error(f"API Error (Code: {response.status_code})")
                    with st.expander("Error Details"):
                        try:
                            error_json = response.json()
                            st.json(error_json)
                        except json.JSONDecodeError:
                            st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Backend may be slow.")
                
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach API server!")
                st.info("""
                Ensure FastAPI is running:
                ```bash
                cd Backend/api
                uvicorn app:app --reload --host 0.0.0.0 --port 8000
                ```
                """)
                
            except Exception as e:
                st.error(f"❌ Unexpected error occurred: {e}")
                st.exception(e)

# ------------------------------------
# Additional Info
# ------------------------------------
st.markdown("---")
st.caption(f"This app uses a Machine Learning model trained on Immo Eliza data from {len(postal_codes_list)} different locations.")
st.caption("Powered by FastAPI + Streamlit")

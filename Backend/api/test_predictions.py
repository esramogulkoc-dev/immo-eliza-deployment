import requests

# FastAPI predict endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Sample data for testing
test_data = [
    {"Livable_surface": 100, "Number_of_bedrooms": 3, "Total_land_surface": 500, "postal_code": 1000,
     "has_terrace": True, "has_garage": False, "region": "flanders", "type_of_heating": "gas", "epc_rating": "c", "type_of_property": "house"},

    {"Livable_surface": 80, "Number_of_bedrooms": 2, "Total_land_surface": 200, "postal_code": 1050,
     "has_terrace": False, "has_garage": True, "region": "brussels", "type_of_heating": "electricity", "epc_rating": "b", "type_of_property": "apartment"},

    {"Livable_surface": 150, "Number_of_bedrooms": 4, "Total_land_surface": 700, "postal_code": 1070,
     "has_terrace": True, "has_garage": True, "region": "flanders", "type_of_heating": "gas", "epc_rating": "d", "type_of_property": "house"},

    {"Livable_surface": 60, "Number_of_bedrooms": 1, "Total_land_surface": 0, "postal_code": 2000,
     "has_terrace": False, "has_garage": False, "region": "flanders", "type_of_heating": "solar", "epc_rating": "a", "type_of_property": "apartment"},

    {"Livable_surface": 250, "Number_of_bedrooms": 5, "Total_land_surface": 1000, "postal_code": 3000,
     "has_terrace": True, "has_garage": True, "region": "wallonia", "type_of_heating": "gas", "epc_rating": "c", "type_of_property": "villa"},
]

print("\n--- Starting Model Test ---\n")

for i, sample in enumerate(test_data, 1):
    try:
        response = requests.post(API_URL, json=sample)
        if response.status_code == 200:
            predicted_price = response.json().get("predicted_price", None)
            print(f"Test {i}: Predicted price → € {predicted_price:,.2f}")
        else:
            print(f"Test {i}: API Error (Status: {response.status_code})")
    except Exception as e:
        print(f"Test {i}: Error → {e}")

print("\n--- Tests Completed ---\n")

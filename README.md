# 🏡 Immo Eliza - ML Model Deployment

This repository contains the end-to-end deployment solution for the Immo Eliza real estate price prediction model. The project is structured to offer both a **FastAPI** endpoint for developers (Backend) and a **Streamlit** web application for non-technical users (Frontend).

## 🚀 Project Architecture (Option 3: Full Stack)

This solution implements a complete separation of concerns between the Backend API and the consuming Frontend application, maximizing stability and scalability.

| Component | Technology | Deployment Environment | Purpose |
| :--- | :--- | :--- | :--- |
| **Backend (API)** | FastAPI + Python | **Render** | Serves the trained Machine Learning model, handles preprocessing, and returns predictions via a JSON endpoint. |
| **Frontend (App)** | Streamlit | **Streamlit Cloud** | Collects user inputs, sends prediction requests to the live API, and visualizes the results. |

## 📂 Repository Structure

The project maintains a clean, modular structure:

IMMO-ELIZA-DEPLOYMENT/
├── Backend/
│   ├── api/
│   │   ├── __init__.py            # Python package initializer (Crucial for module imports)
│   │   ├── app.py                 # Main FastAPI application and API Endpoints
│   │   ├── predict.py             # Model loading, preprocessing, and prediction logic
│   │   └── ...
│   ├── data/
│   │   └── immovlan_cleaned_file_final.csv # Data source for postal code mappings
│   └── models/
│       ├── random_forest_model.pkl    # Trained ML model artifact
│       └── rf_train_columns_065.pkl   # Artifact for required model columns
├── frontend/
│   ├── requirements.txt           # Frontend (Streamlit) dependencies
│   └── streamlit_app.py           # Streamlit Web Application code
├── requirements.txt               # Backend (FastAPI) dependencies
├── Dockerfile                     # Instructions for deploying the API on Render
└── README.me

## 🛠️ Deployment Status and Access

### 1. Backend API (FastAPI)

The API service is deployed and running live on **Render**.

* **Live API Base URL:** `https://immo-eliza-deployment-vgfy.onrender.com`
* **API Documentation (Swagger UI):** Access the auto-generated documentation for all routes and data schemas by appending `/docs` to the base URL.
    * Example: `https://immo-eliza-deployment-vgfy.onrender.com/docs`

### 2. Frontend Application (Streamlit)

The Streamlit application consumes the live API to provide a user interface.

* **Live Streamlit App:** (https://immo-eliza-deployment-cvn8wgrvb3nlp6gz8ub6xk.streamlit.app/)

## 💻 How to Run the App Locally

If a developer wants to inspect or run the application locally, they only need to run the Frontend, as the Backend API is already live on Render.

### Step 1: Clone and Prepare

1.  Clone the repository and navigate to the project root:
    ```bash
    git clone [https://github.com/esramogulkoc-dev/immo-eliza-deployment.git](https://github.com/esramogulkoc-dev/immo-eliza-deployment.git)
    cd immo-eliza-deployment
    ```
2.  Set up a virtual environment (highly recommended).

### Step 2: Run the Frontend (Streamlit App)

1.  Install the required dependencies for the frontend application:
    ```bash
    pip install -r frontend/requirements.txt
    ```
2.  Start the Streamlit application:
    ```bash
    streamlit run frontend/streamlit_app.py
    ```
3.  The app will open automatically in your browser. It is configured to send prediction requests directly to the live Render API.

4. Note on Health Check: The root path ( /) only accepts GETrequests for a basic health check. If you receive a 405 Method Not Allowed status when making a GETrequest, this still confirms the server is alive and running on Render.
---

## 💡 API Usage for Developers

Developers can make direct POST requests to the `/predict` endpoint to receive a price estimate.

### Endpoint: `/predict`

* **Method:** `POST`
* **Full URL:** `https://immo-eliza-deployment-vgfy.onrender.com/predict`
* **Input Body (JSON):** Must contain all features required by the prediction model:

```json
{
  "Livable_surface": 150.0,
  "Number_of_bedrooms": 3,
  "Total_land_surface": 500.0,
  "postal_code": 1000,
  "has_terrace": true,
  "has_garage": true,
  "region": "brussels",
  "type_of_heating": "gas",
  "epc_rating": "c",
  "type_of_property": "house"
}
Output Response (JSON): Returns the prediction and the HTTP status code:

JSON

{
  "predicted_price": 350000.00,
  "status_code": 200
}
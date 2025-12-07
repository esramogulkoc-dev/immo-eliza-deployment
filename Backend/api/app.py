from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import traceback
import sys
import os
from .predict import MODEL_FILE, COL_FILE


# --- Load Model and Functions ---
# Import functions from predict.py (model loading and prediction logic)
try:
    # Python path setup, depends on the directory uvicorn is started from.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from predict import load_model_and_columns, get_prediction, PredictInput
    from predict import load_model_and_columns, rf_model, train_columns

except ImportError as e:
    # If predict.py file is not found or imports inside it fail
    print(f"CRITICAL ERROR: Failed to import predict.py: {e}")
    print("Please ensure predict.py is located under Backend/api/ and all dependencies are installed correctly.")
    sys.exit(1)


app = FastAPI(
    title="Immo Eliza Price Prediction API",
    description="Receives real estate data from Streamlit app and returns price predictions.",
    version="1.0.0"
)

# --- CORS Middleware ---
# Middleware setup: allows requests coming from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Using * for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =======================================================
# CRITICAL GENERAL ERROR HANDLER (Logs all 500 errors to terminal)
# =======================================================
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Catches all unexpected server errors (500), logs detailed trace to terminal,
    and returns a general error message to the client.
    """
    error_trace = traceback.format_exc()
    # Print error trace to terminal (expected to debug server errors)
    print("\n--- SERVER CRITICAL ERROR TRACE (500) ---")
    print(f"Request URL: {request.url}")
    print(f"Error Type: {type(exc).__name__}")
    print(f"Error Message: {str(exc)}")
    print("\nFULL TRACEBACK:")
    print(error_trace)
    print("-------------------------------------------\n")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "message": "An unexpected server error occurred. Please check the server console for details."
        },
    )


# =======================================================
# Application Startup
# =======================================================
@app.on_event("startup")
async def startup_event():
    """Loads the model when the application starts."""
    if not load_model_and_columns():
        print("WARNING: The application may not function properly because the model could not be loaded.")
        
    print("FastAPI application is ready. /predict endpoint is active.")


# =======================================================
# Main Prediction Endpoint
# =======================================================
@app.post("/predict")
def predict(data: PredictInput):
    """
    Receives data from Streamlit and returns the predicted price.
    data: Input validated according to Pydantic model.
    """
    # Note: Pydantic validation errors (422) do not enter this block, handled directly by FastAPI.
    # However, 500 errors coming from get_prediction will trigger this block.
    try:
        predicted_price = get_prediction(data)
        return {"predicted_price": predicted_price}
    except Exception as e:
        # Catch errors from get_prediction and re-raise them
        # so they are handled by the general exception handler.
        raise e 


# =======================================================
# Health Check
# =======================================================
@app.get("/")
def read_root():
    return {"status": "ok", "service": "Immo Eliza API"}

@app.get("/health/model")
def model_health():
    # Attempts to load the model if not already loaded
    load_model_and_columns()

    return {
        "model_loaded": rf_model is not None,
        "columns_loaded": train_columns is not None,
        "num_columns": len(train_columns) if train_columns else 0,
        "model_file": MODEL_FILE,
        "column_file": COL_FILE
    }
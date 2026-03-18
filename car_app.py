from model import train_model
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# 1. Define the input schema for the API
class CarInput(BaseModel):
    Colour: str
    Odometer: int
    Doors: int
    Price: int

# 2. Setup model storage
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model when the app starts
    try:
        train_model()
        # This matches the filename saved in model.py
        ml_models["model"] = joblib.load("model.joblib")
        print("✅ Model loaded successfully from model.joblib")
    except FileNotFoundError:
        print("❌ Error: model.joblib not found. Run model.py first!")
    yield
    # Clean up on shutdown
    ml_models.clear()

# 3. Initialize FastAPI
app = FastAPI(
    title="Car Make Predictor",
    description="API to predict Car Make based on features.",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"status": "online", "model_loaded": "model" in ml_models}

@app.post("/predict")
async def predict(car: CarInput):
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded on the server.")
    
    try:
        # The model pipeline expects a DataFrame with specific column names
        # identical to those used during training in model.py
        input_data = pd.DataFrame([{
            'Colour': car.Colour,
            'Odometer (KM)': car.Odometer,
            'Doors': car.Doors,
            'Price': car.Price
        }])
        
        prediction = ml_models["model"].predict(input_data)
        
        return {
            "input": car.dict(),
            "predicted_make": str(prediction[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

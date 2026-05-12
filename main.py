from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model
model = joblib.load("disease_prediction_model.pkl")

# Load symptom columns
symptom_columns = joblib.load("symptom_columns.pkl")

# Create FastAPI app
app = FastAPI()

# Input format
class SymptomsInput(BaseModel):
    symptoms: list

# Home route
@app.get("/")
def home():
    return {"message": "Disease Prediction API Running"}

# Prediction route
@app.post("/predict")
def predict_disease(data: SymptomsInput):

    # Create empty symptom dictionary
    input_data = {symptom: 0 for symptom in symptom_columns}

    # Mark entered symptoms as 1
    for symptom in data.symptoms:
        if symptom in input_data:
            input_data[symptom] = 1

    # Convert to dataframe
    input_df = pd.DataFrame([input_data])

    # Predict disease
    prediction = model.predict(input_df)[0]

    return {
        "predicted_disease": prediction
    }
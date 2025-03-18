from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


model = joblib.load("regression_model.pkl")


class CementModel(BaseModel):
    Cement: float
    Blast_Furnace_Slag: float = Field(..., alias="Blast Furnace Slag")
    Fly_Ash: float = Field(..., alias="Fly Ash")
    Water: float
    Superplasticizer: float
    Coarse_Aggregate: float = Field(..., alias="Coarse Aggregate")
    Fine_Aggregate: float = Field(..., alias="Fine Aggregate")
    Age: int = Field(..., alias="Age (day)")

    class Config:
        populate_by_name = True

@app.get('/trial')
def trial():
    return {'message':'Wagwannn'}

@app.post('/predict')
def predict_response(data: CementModel):
    # Convert input data to match training feature names
    features = pd.DataFrame([{
        "Cement": data.Cement,
        "Blast Furnace Slag": data.Blast_Furnace_Slag,
        "Fly Ash": data.Fly_Ash,
        "Water": data.Water,
        "Superplasticizer": data.Superplasticizer,
        "Coarse Aggregate": data.Coarse_Aggregate,
        "Fine Aggregate": data.Fine_Aggregate,
        "Age (day)": data.Age
    }])

    # Predict using trained model
    prediction = model.predict(features)

    return {"predicted_strength": max(prediction[0], 0)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

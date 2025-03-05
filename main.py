from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel

app = FastAPI()

# Load your saved model and encoder
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

# Define input structure
class ProfitPredictionInput(BaseModel):
    rd_spend: float
    admin_spend: float
    marketing_spend: float
    state: str

@app.get("/")
def home():
    return {"message": "Profit Predictor API by Zara"}

@app.post("/predict")
def predict_profit(input: ProfitPredictionInput):
    state_encoded = encoder.transform([[input.state]])
    finput = np.concatenate(
        (state_encoded, [[input.rd_spend, input.admin_spend, input.marketing_spend]]), 
        axis=1
    )
    profit = model.predict(finput)
    return {"predicted_profit": float(profit[0])}

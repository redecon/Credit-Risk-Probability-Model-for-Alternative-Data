from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import joblib
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

app = FastAPI(title="Credit Risk API", version="1.0")


class PredictionRequest(BaseModel):
    TransactionId: str
    BatchId: str
    AccountId: str
    SubscriptionId: str
    CustomerId: str
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    Amount: float = Field(..., gt=0)
    Value: float = Field(..., gt=0)
    TransactionStartTime: str
    PricingStrategy: str
    FraudResult: int

class PredictionResponse(BaseModel):
    risk_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Load Production model from MLflow
        # New (alias-based, correct)
        model = mlflow.sklearn.load_model("models:/CreditRiskModel@Production")

        # Load preprocessor
        preprocessor = joblib.load("data/processed/preprocessor.pkl")
        # Convert request to DataFrame
        new_data = pd.DataFrame([request.dict()])
        # Transform features
        processed = preprocessor.transform(new_data)
        # Predict probability
        prob = model.predict_proba(processed)[:, 1][0]
        return PredictionResponse(risk_probability=float(prob))
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

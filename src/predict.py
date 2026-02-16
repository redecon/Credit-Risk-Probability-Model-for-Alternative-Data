import mlflow
import pandas as pd
import joblib

def predict(new_data: pd.DataFrame):
    model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    processed = preprocessor.transform(new_data)
    return model.predict_proba(processed)[:, 1]

import pandas as pd
import joblib
from sklearn.base import BaseEstimator

def predict(model: BaseEstimator, new_data: pd.DataFrame):
    """
    Predict risk probability for new customers using saved preprocessor.
    """
    preprocessor = joblib.load("data/processed/preprocessor.pkl")
    processed = preprocessor.transform(new_data)
    return model.predict_proba(processed)[:, 1]

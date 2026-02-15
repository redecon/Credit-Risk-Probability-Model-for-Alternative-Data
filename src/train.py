import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from src.data_processing import build_preprocessor
from src.config import TrainingConfig

def train_model(data_path: str):
    config = TrainingConfig()
    df = pd.read_csv(data_path)

    y = df["is_high_risk"]
    X = df.drop("is_high_risk", axis=1)

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=config.test_size, random_state=config.random_state
    )

    model = LogisticRegression(random_state=config.random_state)
    model.fit(X_train, y_train)

    print("✅ Model evaluation report:")
    print(classification_report(y_test, model.predict(X_test)))

    # Save artifacts
    joblib.dump(model, "data/processed/model.pkl")
    joblib.dump(preprocessor, "data/processed/preprocessor.pkl")

    return model

import pandas as pd
from src.train import train_model
from src.predict import predict
from src.target_variable import create_proxy_target
import joblib

def main():
    data_path = "data/raw/data.csv"
    df = pd.read_csv(data_path, parse_dates=["TransactionStartTime"])
    print("✅ Raw data loaded:", df.shape)

    snapshot_date = pd.Timestamp("2025-12-31")
    df_with_target = create_proxy_target(df, snapshot_date)
    print("✅ Proxy target created:", df_with_target.shape)

    processed_path = "data/processed/transactions_with_target.csv"
    df_with_target.to_csv(processed_path, index=False)
    print("✅ Processed dataset saved:", processed_path)

    model = train_model(processed_path)
    print("✅ Model trained")

    # Load model for prediction
    model = joblib.load("data/processed/model.pkl")

    # Sample new data with same columns as training
    new_data = df_with_target.drop("is_high_risk", axis=1).sample(2)
    preds = predict(model, new_data)
    print("✅ Predictions complete")
    print("Risk probabilities:", preds)

if __name__ == "__main__":
    main()

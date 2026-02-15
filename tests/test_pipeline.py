import pandas as pd
import numpy as np
from src.data_processing import build_preprocessor
from src.target_variable import create_proxy_target
from src.train import train_model
from src.predict import predict
import joblib


# 1. Feature engineering output shape
def test_feature_engineering_shape():
    df = pd.DataFrame({
        "Amount": [100, 200],
        "Value": [100, 200],
        "CurrencyCode": ["UGX", "UGX"],
        "ProductCategory": ["Fashion", "Electronics"],
        "ChannelId": ["web", "ios"]
    })
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 2  # two rows
    assert transformed.shape[1] > 0   # features created

# 2. RFM calculation correctness
def test_rfm_calculation():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionStartTime": pd.to_datetime(["2025-12-01", "2025-12-15", "2025-12-10"]),
        "Amount": [100, 200, 300]
    })
    snapshot_date = pd.Timestamp("2025-12-31")
    df_with_target = create_proxy_target(df, snapshot_date)
    assert "is_high_risk" in df_with_target.columns

# 3. Proxy label generation
def test_proxy_label_distribution():
    df = pd.DataFrame({
        "CustomerId": [1, 2, 3],
        "TransactionStartTime": pd.to_datetime(["2025-12-01", "2025-12-02", "2025-12-03"]),
        "Amount": [100, 200, 300]
    })
    snapshot_date = pd.Timestamp("2025-12-31")
    df_with_target = create_proxy_target(df, snapshot_date)
    assert set(df_with_target["is_high_risk"].unique()).issubset({0, 1})

# 4. Metric computation (train_model prints classification report)
def test_model_training_runs(tmp_path):
    data = tmp_path / "dummy.csv"
    data.write_text(
        "Amount,Value,CurrencyCode,ProductCategory,ChannelId,is_high_risk\n"
        "100,200,UGX,Fashion,web,0\n"
        "150,250,UGX,Electronics,ios,1\n"
        "200,300,UGX,Fashion,web,0\n"
        "250,350,UGX,Electronics,ios,1\n"
        "300,400,UGX,Fashion,web,0\n"
        "350,450,UGX,Electronics,ios,1"
    )
    model = train_model(str(data))
    assert model is not None

# 5. Model prediction output format
def test_prediction_output_format(tmp_path):
    data = tmp_path / "dummy.csv"
    data.write_text(
        "Amount,Value,CurrencyCode,ProductCategory,ChannelId,is_high_risk\n"
        "100,200,UGX,Fashion,web,0\n"
        "150,250,UGX,Electronics,ios,1\n"
        "200,300,UGX,Fashion,web,0\n"
        "250,350,UGX,Electronics,ios,1\n"
    )
    model = train_model(str(data))

    new_data = pd.DataFrame({
        "Amount": [120, 260],
        "Value": [220, 360],
        "CurrencyCode": ["UGX", "UGX"],
        "ProductCategory": ["Fashion", "Electronics"],
        "ChannelId": ["web", "ios"]
    })
    preds = predict(model, new_data)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 2


def test_rfm_calculation():
    df = pd.DataFrame({
        "CustomerId": [1, 1, 2, 3],
        "TransactionStartTime": pd.to_datetime(
            ["2025-12-01", "2025-12-15", "2025-12-10", "2025-12-20"]
        ),
        "Amount": [100, 200, 300, 400]
    })
    snapshot_date = pd.Timestamp("2025-12-31")
    df_with_target = create_proxy_target(df, snapshot_date)

    # Assertions
    assert "is_high_risk" in df_with_target.columns
    assert set(df_with_target["is_high_risk"].unique()).issubset({0, 1})

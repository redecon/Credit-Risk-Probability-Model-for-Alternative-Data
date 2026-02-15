import pandas as pd
from src.data_processing import build_preprocessor

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

    # Assertions
    assert transformed.shape[0] == 2   # two rows
    assert transformed.shape[1] > 0    # features created

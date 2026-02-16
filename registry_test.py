import joblib
preprocessor = joblib.load("data/processed/preprocessor.pkl")
print(preprocessor.feature_names_in_)


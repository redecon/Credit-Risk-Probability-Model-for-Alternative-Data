import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
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

    # MLflow experiment logging
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ✅ Log metrics
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, zero_division=0))

        # ✅ Log parameters
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("model_type", "LogisticRegression")

        # ✅ Save and log artifacts
        joblib.dump(model, "data/processed/model.pkl")
        joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
        mlflow.log_artifact("data/processed/model.pkl", artifact_path="models")
        mlflow.log_artifact("data/processed/preprocessor.pkl", artifact_path="preprocessor")

        # ✅ Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="CreditRiskModel"
        )

        # ✅ Log evaluation comparison table
        eval_df = evaluation_table(y_test, y_pred)
        eval_df.to_csv("data/processed/evaluation.csv", index=False)
        mlflow.log_artifact("data/processed/evaluation.csv", artifact_path="evaluation")

        # ✅ Log governance document
        mlflow.log_artifact("docs/model_selection.md", artifact_path="governance")

        return model


def evaluation_table(y_true, y_pred):
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    return pd.DataFrame([metrics])


# ✅ Point MLflow to the local tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# ✅ Set experiment name
mlflow.set_experiment("CreditRiskExperiment")

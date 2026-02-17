import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data_processing import build_preprocessor
from src.config import TrainingConfig


def train_model(data_path: str):
    config = TrainingConfig()
    print("📂 Loading dataset from:", data_path)
    df = pd.read_csv(data_path)

    y = df["is_high_risk"]
    X = df.drop("is_high_risk", axis=1)

    print("⚙️ Building preprocessor and transforming features...")
    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✂️ Splitting dataset into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=config.test_size, random_state=config.random_state
    )

    model = LogisticRegression(random_state=config.random_state)

    print("🚀 Starting MLflow run...")
    with mlflow.start_run():
        print("🤖 Training Logistic Regression model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ✅ Log metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"📊 Metrics logged: Precision={precision}, Recall={recall}, F1={f1}")

        # ✅ Log parameters
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("model_type", "LogisticRegression")
        print("⚙️ Parameters logged.")

        # ✅ Save and log artifacts
        joblib.dump(model, "data/processed/model.pkl")
        joblib.dump(preprocessor, "data/processed/preprocessor.pkl")
        mlflow.log_artifact("data/processed/model.pkl", artifact_path="models")
        mlflow.log_artifact("data/processed/preprocessor.pkl", artifact_path="preprocessor")
        print("📦 Artifacts saved and logged.")

        # ✅ Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="CreditRiskModel"
        )
        print("✅ Model registration attempted — check MLflow UI for CreditRiskModel.")

        # ✅ Log evaluation comparison table
        eval_df = evaluation_table(y_test, y_pred)
        eval_df.to_csv("data/processed/evaluation.csv", index=False)
        mlflow.log_artifact("data/processed/evaluation.csv", artifact_path="evaluation")
        print("📑 Evaluation table logged.")

        # ✅ Log governance document
        mlflow.log_artifact("docs/models.md", artifact_path="governance")
        print("📜 Governance document logged.")

        # ✅ SHAP Explainability
        print("🔍 Generating SHAP explainability plots...")
        explainer = shap.Explainer(model, X_train)

        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()

        # Global feature importance
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
        plt.savefig("data/processed/shap_global_importance.png")
        mlflow.log_artifact("data/processed/shap_global_importance.png", artifact_path="explainability")
        plt.close()
        print("📊 SHAP global importance logged.")

        # Single prediction explanation (first test sample)
        shap.force_plot(
            explainer.expected_value,
            shap_values[0].values,
            X_train[0].toarray()[0] if hasattr(X_train[0], "toarray") else X_train[0],
            matplotlib=True,
            show=False
        )
        plt.savefig("data/processed/shap_single_prediction.png")
        mlflow.log_artifact("data/processed/shap_single_prediction.png", artifact_path="explainability")
        plt.close()
        print("📈 SHAP single prediction explanation logged.")



        print("🎉 Training run completed successfully.")
        return model


def evaluation_table(y_true, y_pred):
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    return pd.DataFrame([metrics])


# ✅ Point MLflow to the local tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# ✅ Set experiment name
mlflow.set_experiment("CreditRiskExperiment")


if __name__ == "__main__":
    model = train_model("data/processed/transactions_with_target.csv")
    print("🎉 Training script finished, model trained and registered.")

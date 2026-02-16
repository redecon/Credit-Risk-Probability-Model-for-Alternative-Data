from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    name="CreditRiskModel",
    alias="Production",
    version=3
)
print("✅ Alias 'Production' set for CreditRiskModel version 3")

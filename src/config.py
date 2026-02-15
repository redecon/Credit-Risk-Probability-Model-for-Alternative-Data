from dataclasses import dataclass

@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    model_type: str = "logistic_regression"
    scaler_type: str = "standard"

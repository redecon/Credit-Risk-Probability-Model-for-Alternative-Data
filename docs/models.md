# Model Selection Rationale

## 1. Problem Context
The goal of this project is to build a **credit risk probability model** using alternative data sources.  
Key requirements:
- Interpretability for stakeholders
- Reliable performance across imbalanced datasets
- Reproducibility and governance through MLflow

---

## 2. Candidate Models Considered
- **Logistic Regression**  
  - Pros: Simple, interpretable coefficients, baseline for classification.  
  - Cons: Limited ability to capture complex nonlinear relationships.

- **Random Forest / Gradient Boosted Trees (future work)**  
  - Pros: Strong predictive power, handles feature interactions.  
  - Cons: Less interpretable, heavier compute cost.

---

## 3. Metrics Used
We evaluated models using:
- **Precision**: Ability to correctly identify high‑risk customers.  
- **Recall**: Coverage of actual high‑risk customers.  
- **F1 Score**: Balance between precision and recall.  
- **ROC‑AUC (future extension)**: Overall discrimination ability.

---

## 4. Trade‑offs
- **Interpretability vs. Accuracy**: Logistic Regression was chosen as the baseline because stakeholders need clear reasoning behind risk scores.  
- **Precision vs. Recall**: We prioritized **recall** to minimize false negatives (i.e., missing risky customers), while monitoring precision to avoid excessive false positives.  
- **Reproducibility**: MLflow logging ensures every run is tracked with parameters, metrics, and artifacts.

---

## 5. Model Governance
- **Experiment Tracking**: All runs logged in MLflow (`http://localhost:5000`).  
- **Model Registry**: Best model registered as `CreditRiskModel`.  
- **Versioning**: Models promoted to *Staging* or *Production* via MLflow UI.  
- **Reloadability**: Production model can be reloaded independently using:
  ```python
  model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
  ```

---

## 6. Current Best Model
- **Model Type:** Logistic Regression
- **Parameters:** random_state=42, default solver
 

---

## 7. Next Steps
- Experiment with tree‑based models (Random Forest, XGBoost).

- Compare ROC‑AUC and calibration curves.

- Document trade‑offs between interpretability and predictive lift.

- Update governance docs with rationale for any model promotion.


---

## Outcome
This file gives reviewers:
- Clear rationale for why Logistic Regression was chosen.  
- Metrics and trade‑offs explained.  
- Governance story tied to MLflow registry.  
- A roadmap for future model improvements.

---

# Model Selection Rationale

## 1. Problem Context
We aim to build a **credit risk probability model** using alternative data sources.  
Key requirements:
- Interpretability for stakeholders
- Reliable performance across imbalanced datasets
- Reproducibility and governance through MLflow

---

## 2. Candidate Models Considered
- **Logistic Regression (baseline)**  
  - Pros: Simple, interpretable coefficients, strong baseline for classification.  
  - Cons: Limited ability to capture complex nonlinear relationships.

- **Tree-based models (Random Forest, Gradient Boosted Trees)**  
  - Pros: Strong predictive power, handles feature interactions.  
  - Cons: Less interpretable, heavier compute cost.

---

## 3. Metrics Used
We evaluated models using:
- **Precision**: Correct identification of high‑risk customers.  
- **Recall**: Coverage of actual high‑risk customers.  
- **F1 Score**: Balance between precision and recall.  
- **ROC‑AUC**: Overall discrimination ability (future extension).

---

## 4. Trade‑offs
- **Interpretability vs. Accuracy**: Logistic Regression chosen as baseline for transparency.  
- **Precision vs. Recall**: Recall prioritized to minimize false negatives (missing risky customers).  
- **Reproducibility**: MLflow logging ensures every run is tracked with parameters, metrics, and artifacts.

---

## 5. Model Governance
- **Experiment Tracking**: All runs logged in MLflow (`http://localhost:5000`).  
- **Model Registry**: Best model registered as `CreditRiskModel`.  
- **Versioning**: Models promoted to *Staging* or *Production* via MLflow UI.  
- **Reloadability**: Production model can be reloaded independently:
  ```python
  model = mlflow.sklearn.load_model("models:/CreditRiskModel/Production")
  ```

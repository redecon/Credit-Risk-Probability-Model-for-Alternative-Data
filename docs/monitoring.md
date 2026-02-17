# Governance & Monitoring Plan

## 1. Model Registry Practices
- All models are registered in MLflow under the name **CreditRiskModel**.
- Versioning is enforced:
  - **Production** → actively serving predictions.
  - **Staging** → under evaluation/testing.
  - **Archived** → older versions retained for audit.
- Transition commands:
```bash
  mlflow models transition --model-name CreditRiskModel --version <N> --stage Production
  mlflow models transition --model-name CreditRiskModel --version <N> --stage Staging
```

## 2. Feature Importance Rationale

- **Transaction Amount & Value →** strongest drivers of risk, proxy for exposure.

- **Channel ID →** operational risk driver; mobile/alternative channels show higher risk.

- **Product Category →** airtime and financial services correlate with risk; utility bills and transport reduce risk.

- **Currency Code →** regional proxy; UGX transactions show distinct patterns.

These drivers are documented to ensure transparency and stakeholder alignment.

## 3. Monitoring Plan
**Prediction Drift**
- Weekly histograms of predicted risk scores compared to baseline.

- Alert if distribution shifts beyond threshold (e.g., PSI > 0.2).

**Feature Drift**
- Monitor input features (Amount, ChannelId, ProductCategory).

- Apply KS test / PSI to detect significant changes.

**Fairness Audits**
- Use SHAP subgroup analysis (e.g., by currency, product category).

- Document disparities and investigate if certain groups consistently show higher risk attribution.

**Feature Importance Stability**
- Compare current SHAP global importance to historical baseline.

- Trigger alerts if top drivers change significantly (e.g., ChannelId drops out, Currency spikes).

## 4. Governance Narrative
This governance plan ensures the Credit Risk Model remains:

**Reliable →** monitored for drift and fairness.

**Explainable →** SHAP outputs documented and reviewed.

**Transparent →** rationale for feature importance recorded.

**Auditable →** MLflow registry maintains version history and stage transitions.
## Credit Scoring Business Understanding

Bati Bank is developing a Credit Scoring Model to support its Buy-Now-Pay-Later (BNPL) partnership with a major eCommerce platform. Because BNPL loans are short-term, high-volume, and often issued to customers with limited credit histories, the bank requires a transparent, compliant, and data-driven approach to assessing credit risk.

### Basel II and the Need for Interpretable, Well‑Documented Models
Under the Basel II Capital Accord, financial institutions must demonstrate that their credit risk models are:
- **Interpretable**  
- **Well‑documented**  
- **Auditable**  
- **Grounded in sound risk measurement principles**

This regulatory environment strongly favors models whose decision logic can be explained to auditors, regulators, and internal risk committees. As a result, interpretable models such as **Logistic Regression with Weight of Evidence (WoE)** are often preferred as baseline or production models in regulated financial contexts.

### Why a Proxy Variable Is Necessary
The dataset provided by the eCommerce partner does not include direct repayment outcomes (e.g., late payment, default, delinquency). Without a target variable, supervised learning is not possible.

To address this, we construct a **proxy risk label** using behavioral patterns derived from RFM (Recency, Frequency, Monetary Value) metrics. This approach allows the bank to:
- Segment customers based on purchasing behavior  
- Identify high‑risk behavioral clusters  
- Train supervised models even in the absence of historical default data  

However, using a proxy introduces **model risk**, including:
- Potential misclassification of customers  
- Over‑ or under‑estimation of risk  
- Regulatory scrutiny if the proxy is not well‑justified  

Therefore, the proxy creation process must be transparent, documented, and validated.

### Trade-offs Between Interpretable and Complex Models
Two broad categories of models are considered:

#### 1. Interpretable Models (e.g., Logistic Regression with WoE)
**Advantages**
- High interpretability  
- Easy to document and justify to regulators  
- Stable and predictable behavior  
- Works well with WoE/IV feature engineering  

**Limitations**
- May underfit complex, nonlinear relationships  
- Lower predictive power compared to advanced models  

#### 2. Complex Models (e.g., Gradient Boosting Machines)
**Advantages**
- High predictive accuracy  
- Captures nonlinear interactions  
- Often outperforms simpler models  

**Limitations**
- Harder to interpret  
- Requires explainability tools (e.g., SHAP)  
- More challenging to justify in regulated environments  

In a Basel II context, the recommended approach is:
- Use **Logistic Regression + WoE** as the primary, regulator‑friendly model  
- Benchmark against **Gradient Boosting** to evaluate performance gains  
- Use SHAP values to bridge interpretability gaps if complex models are considered  

# Credit Risk Probability Model for Alternative Data

### Basel II, risk measurement, and the need for interpretable models


Under the Basel II Capital Accord, banks are required to quantify credit risk using sound, transparent, and consistently applied methodologies, and to hold capital commensurate with the underlying risk of their credit exposures. This pushes institutions toward models whose assumptions, inputs, and outputs can be clearly explained to internal risk committees, external auditors, and regulators, rather than purely “black-box” systems optimized only for accuracy. Basel II also emphasizes governance, validation, and backtesting: the bank must demonstrate how the model was developed, which risk drivers it uses, how performance is monitored, and how model limitations are addressed. For this project, that means our credit risk model must be interpretable, traceable, and well-documented end to end—from feature engineering and proxy default definition to model training, threshold selection, and deployment decisions.

Why a proxy default variable is necessary and its business risks In a typical credit scoring setup, models are trained on historical data where each loan carries an explicit “default” outcome (e.g., 90+ days past due, written off, or restructured). In this challenge, we do not have a direct default label; instead, we only see alternative behavioral data (eCommerce transactions, RFM patterns, fraud flags). To learn from this data, we must define a proxy target that approximates credit risk—for example, labeling customers as “high risk” or “low risk” based on their recency, frequency, monetary patterns, or other behavioral indicators. This step is necessary to translate raw behavior into a supervised learning problem, but it introduces model risk: if the proxy is poorly aligned with true default behavior, the model may systematically mis-rank customers.

The main business risks of using a proxy are:

Misclassification risk: Customers may be labeled “bad” based on behavior that is only weakly related to actual repayment capacity, leading to high rejection rates for creditworthy customers (lost revenue) or approval of truly risky ones (higher default and loss rates).

Bias and fairness risk: If the proxy variable reflects historical operational or behavioral patterns that correlate with demographic groups, the model may inadvertently encode unfair discrimination or unequal access to credit.

Model drift and instability: As customer behavior, platform usage, and economic conditions evolve, the mapping from behavior to true default risk may change. A static proxy definition can become stale, degrading model performance.

Regulatory and reputational risk: If regulators or customers challenge decisions, the bank must justify both the proxy definition and any downstream model. Weak justification or opaque logic can harm trust and may be viewed as non-compliant with sound risk management expectations.

Because of these risks, the proxy definition must be carefully reasoned, validated, documented, and regularly reviewed, with explicit acknowledgement of its limitations.

Trade-offs: simple interpretable models vs complex high-performance models In a regulated financial context, there is a fundamental trade-off between model interpretability and predictive power:

Simple, interpretable models (e.g., Logistic Regression with WoE):

Pros:

Transparency: Weight-of-Evidence (WoE) transformations and logistic regression coefficients provide clear, monotonic relationships between predictors and default odds, making it easier for risk managers, auditors, and regulators to understand and challenge the model.

Governance and documentation: These models align well with traditional scorecard practices, enabling straightforward documentation, reason codes for adverse actions, and policy overlays.

Stability and robustness: WoE binning and regularization can yield stable models that are less sensitive to noise and easier to monitor over time.

Cons:

Limited flexibility: They may underfit complex, nonlinear relationships and interactions present in alternative behavioral data, potentially leaving predictive power on the table.

Feature engineering burden: Achieving good performance may require extensive manual binning, transformation, and interaction design.

Complex, high-performance models (e.g., Gradient Boosting, XGBoost, LightGBM):

Pros:

Higher predictive accuracy: Tree-based ensembles capture nonlinearities and interactions automatically, often producing better risk rank-ordering and more granular probability estimates, especially on rich behavioral datasets.

Feature discovery: They can uncover patterns that traditional scorecards might miss, which is particularly valuable when working with alternative data sources like eCommerce transactions.

Cons:

Opacity and explainability challenges: Even with tools like SHAP or feature importance plots, explaining individual decisions and global model behavior is more complex, which can be problematic for regulatory reviews and customer-facing explanations.

Governance complexity: Model validation, challenger models, and monitoring frameworks must be more sophisticated, and regulators may expect stronger justification for using complex approaches.

Overfitting and operational risk: Without careful tuning, cross-validation, and continuous monitoring, these models can overfit historical patterns and degrade in new conditions, increasing model risk.

For this project, the optimal strategy is often hybrid: use interpretable, scorecard-style models as a baseline and governance-friendly option, while exploring gradient boosting as a performance benchmark or decision-support tool. Any complex model considered for production must be supported by robust documentation, explainability analysis, and clear evidence that its incremental performance gain justifies the added governance and operational burden.

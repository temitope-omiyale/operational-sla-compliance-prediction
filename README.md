# Operational SLA & Delivery Compliance Prediction

### Business Problem
Across **180,519 supply chain transactions**, **54.83%** experienced a late delivery SLA breach, creating operational bottlenecks and compliance risk. This project combines statistical inference with machine learning to quantify regional and segment drivers of SLA outcomes and flag high-risk orders before dispatch.

### Methodology
- **Dataset:** DataCo Global Supply Chain Dataset (180,519 records across 53 operational, customer, and order features).
- **Statistical Inference (`Statsmodels`):** Fitted a **Multinomial Logistic Regression (`MNLogit`)** across four operational states (*Late Delivery [Reference], Advance Shipping, Shipping on Time, Shipping Canceled*; LLR $p = 2.67 \times 10^{-23}$) to evaluate regional and segment odds ratios (`Exp(B)`).
- **Predictive Modeling (`scikit-learn`):** Trained a **Random Forest Classifier** (80/20 train-test split, 36,104 test records) incorporating shipping mode policy, scheduled lead time, region, segment, and order value.

### Key Findings
- **High Baseline SLA Breach Rate:** **54.83%** of all orders missed their scheduled delivery SLA, while **23.04%** shipped in advance, **17.84%** shipped on time, and **4.30%** were canceled.
- **Regional Fulfillment Bottlenecks:** Regional geography is a statistically significant driver of SLA outcomes ($p < 0.001$), with regions such as **Central Africa (`OR = 0.60`)** and **US Center (`OR = 0.63`)** showing ~37–40% lower odds of advance fulfillment relative to late delivery.
- **Predictive SLA Detection:** The Random Forest model achieved **75% recall** and **63% precision** (0.69 F1-score) in detecting **Late Delivery** breaches on unseen test data.

### Process Optimization & Governance Recommendations
1. **Regional SLA Buffer Calibration:** Adjust scheduled dispatch lead times for high-latency regions where baseline fulfillment odds lag significantly.
2. **Automated Pre-Dispatch Risk Flagging:** Integrate classification scoring at order entry to prioritize high-risk shipping tiers for proactive monitoring.

### Tools Used
Python (`pandas`, `NumPy`, `Statsmodels`, `scikit-learn`) • Jupyter / Kaggle Notebooks

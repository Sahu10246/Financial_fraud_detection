# Financial Fraud Detection Using Random Forest

## Overview

This project focuses on detecting fraudulent financial transactions using a machine learning approach. The dataset contains approximately 7 million transaction records, with fraud cases representing only about 0.07% of the total data. Due to this extreme class imbalance, the primary objective is to build a robust model that can effectively identify fraud while minimizing false positives.

The solution follows a complete end-to-end workflow, including data cleaning, feature engineering, model development, evaluation, and business interpretation.

---

## Objective

- Detect fraudulent transactions in a highly imbalanced dataset  
- Improve fraud recall while controlling false positives  
- Generate actionable business insights for proactive fraud prevention  

---

## Methodology

### 1. Data Preprocessing
- Removed duplicate records  
- Handled missing values  
- Treated outliers using the IQR method  
- Reduced multicollinearity using correlation analysis  

### 2. Feature Engineering
- Created balance difference features  
- Identified unusually large transactions  
- Derived behavioral indicators from transaction data  

### 3. Model Development
- Implemented Random Forest Classifier  
- Used `class_weight="balanced"` to address class imbalance  
- Applied stratified train-test split  

### 4. Evaluation Metrics
Due to severe imbalance, performance was evaluated using:
- ROC-AUC Score  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

Accuracy was not prioritized, as it is misleading in fraud detection problems.

---

## Key Fraud Indicators

The model identified the following as strong predictors of fraud:

- Large transaction amounts  
- Sudden discrepancies in account balances  
- Abnormal transaction behavior patterns  

These factors align with common real-world fraud tactics.

---

## Business Recommendations

Based on model insights:

- Implement real-time fraud risk scoring  
- Trigger additional verification (e.g., OTP) for high-risk transactions  
- Deploy monitoring dashboards for anomaly detection  
- Regularly retrain the model to handle concept drift  

---

## Measuring Success

Post-deployment effectiveness can be measured by:

- Reduction in fraud rate  
- Lower false positive rate  
- Improved fraud recall  
- Continuous performance monitoring  

---

## How to Run the Project

1. Install dependencies:


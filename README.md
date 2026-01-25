# Mortgage Loan Approval Risk Assessment System

```tree 
mortgage-loan-approval-risk-assessment-system/
├── data/
│   ├── Raw          # describes required CSVs (no raw data)
├── notebooks/
│   └── exploration.ipynb  # optional, cleaned
├── src/
│   ├── config.py
│   ├── metrics.py
│   ├── features.py
│   ├── train_churn.py
│   ├── train_price.py
│   ├── decision_logic.py
│   ├── evaluate.py
│   └── main.py
├── requirements.txt
├── README.md
└── .gitignore


## Overview
End-to-end machine learning system for mortgage loan approval that combines
default risk prediction, loan pricing, and profit optimization under capital
constraints.

## Problem Statement
- Predict default probability
- Predict loan price
- Maximize profit under risk constraints

## Models Used
- XGBoost + calibration (churn)
- CatBoost regression (price)

## Decision Logic
- Threshold-based issuance
- Capital cap
- Profit simulation

## Evaluation Metrics
- ROC AUC
- MSLE / MAPE
- Business profit

## Project Structure
## How to Run
## Results

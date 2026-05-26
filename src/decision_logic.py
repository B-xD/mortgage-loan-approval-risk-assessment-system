#src/decision_logic.py

from metrics import calc_all_metrics, alg1
import pandas as pd 


def apply_decision_rules(data: pd.DataFrame, churn_model, price_model, 
                         churn_features: list, price_features: list, max_account=50_000):
    """
    Apply ML predictions and decision rules to a dataframe.

    Args:
        data: pd.DataFrame (train, valid, test)
        churn_model: trained pipeline for churn prediction
        price_model: trained pipeline for price prediction
        churn_features: list of columns for churn prediction
        price_features: list of columns for price prediction
        max_account: maximum account limit for financial metric

    Returns:
        data_copy: pd.DataFrame with added predictions and priority
        metrics: pd.DataFrame with calculated metrics
        financial_outcome: pd.DataFrame with financial outcome
    """
    data_copy = data.copy()

    # Add churn predictions
    data_copy['__churn_prob'] = churn_model.predict_proba(data_copy[churn_features])[:, 1]

    # Add price predictions
    data_copy['__price_predict'] = price_model.predict(data_copy[price_features])

    # Apply loan issuance priority
    data_copy['__priority'] = data_copy.apply(alg1, axis=1)

    # Calculate metrics
    metrics = pd.DataFrame(data=[calc_all_metrics(data_copy)],
                           index=['dataset']).T

    financial_outcome = pd.DataFrame(
        data=[calc_all_metrics(data_copy, max_account=max_account)],
        index=['dataset_financial']
    ).T

    return data_copy, metrics, financial_outcome


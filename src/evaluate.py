#src/evaluate.py 

from metrics import metric_for_churn, metric_for_price
import pandas as pd 


def evaluate_system(train_all: pd.DataFrame, 
                    y_train_churn, y_valid_churn, y_train_price, y_valid_price, 
                    y_pred_churn_train, y_pred_churn_valid, 
                    y_pred_price_train, y_pred_price_valid) -> float:
    """
    Evaluates churn and price models using provided ground truths and predictions.
    """

    # --- Churn ---
    churn_score_train = metric_for_churn(y_train_churn, y_pred_churn_train)
    churn_score_valid = metric_for_churn(y_valid_churn, y_pred_churn_valid)

    print("\n--- Churn Metrics ---")
    print(f"roc_auc_train_churn: {churn_score_train}")
    print(f"roc_auc_valid_churn: {churn_score_valid}")

    # --- Price ---
    price_score_train = metric_for_price(y_train_price, y_pred_price_train)
    price_score_valid = metric_for_price(y_valid_price, y_pred_price_valid)

    print("\n--- Price Metrics ---")
    print(f"mslr_train_price: {price_score_train}")
    print(f"mslr_valid_price: {price_score_valid}")

    # --- Full data evaluation ---
    all_metrics_churn = metric_for_churn(train_all["__churn"], train_all["__churn_prob"])
    all_metrics_price = metric_for_price(train_all["__price_doc"], train_all["__price_predict"])

    print("\n--- Full Data Metrics ---")
    print(f"roc_auc_train_all_churn: {all_metrics_churn}")
    print(f"mslr_train_all_price: {all_metrics_price}")

    return {
        "churn_train": churn_score_train,
        "churn_valid": churn_score_valid,
        "price_train": price_score_train,
        "price_valid": price_score_valid,
        "all_churn": all_metrics_churn,
        "all_price": all_metrics_price
    }
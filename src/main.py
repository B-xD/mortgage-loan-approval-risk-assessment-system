#src/main.py 

#system orchestrator
import pandas as pd
from load import load_data
from train_churn import train_churn_model
from train_price import train_price_model
from features import select_variables
from decision_logic import apply_decision_rules
from evaluate import evaluate_system
from preprocessing import transformer
from config import INPUT_FEATURES_CHURN, INPUT_FEATURES_PRICE

def main():

    churn_target = '__churn'
    price_target = '__price_doc'
#----------------------------------------------------------------
# 1. LOAD THE DATA 
# ---------------------------------------------------------------
    train, valid, train_all,  test, y_train_price, y_valid_price, y_train_churn, y_valid_churn,= load_data()

#----------------------------------------------------------------
# 2. SELECT FEATURES 
# ---------------------------------------------------------------
    categorical_cols_churn, numerical_cols_churn = select_variables(train, INPUT_FEATURES_CHURN)
    categorical_cols_price, numerical_cols_price = select_variables(train, INPUT_FEATURES_PRICE)

#----------------------------------------------------------------
# 3. APPLY TRANSFORMATIONS TO THE DATA 
# ---------------------------------------------------------------
    transformed_churn = transformer(categorical_cols_churn, numerical_cols_churn)
    transformed_price = transformer(categorical_cols_price, numerical_cols_price)
    transformed_churn.fit(train[INPUT_FEATURES_CHURN])
    transformed_price.fit(train[INPUT_FEATURES_PRICE])

#----------------------------------------------------------------
# 4. TRAIN MODELS 
# ---------------------------------------------------------------
    churn_model = train_churn_model(transformed_churn, train[INPUT_FEATURES_CHURN], train[churn_target])
    price_model = train_price_model(transformed_price, train[INPUT_FEATURES_PRICE], train[price_target])

#----------------------------------------------------------------
# 5. MAKE PREDICTIONS 
# ---------------------------------------------------------------
    y_pred_churn_train = churn_model.predict_proba(train[INPUT_FEATURES_CHURN])[:, 1]
    y_pred_churn_valid = churn_model.predict_proba(valid[INPUT_FEATURES_CHURN])[:, 1]

    y_pred_price_train = price_model.predict(train[INPUT_FEATURES_PRICE])
    y_pred_price_valid = price_model.predict(valid[INPUT_FEATURES_PRICE])

    train_all["__churn_prob"] = churn_model.predict_proba(train_all[INPUT_FEATURES_CHURN])[:, 1]
    train_all["__price_predict"]= price_model.predict(train_all[INPUT_FEATURES_PRICE])

#----------------------------------------------------------------
# 6. EVALUATE MODEL'S PERFORMANCE 
# ---------------------------------------------------------------
    evaluation_results = evaluate_system(
        train_all=train_all,
        y_train_churn=y_train_churn,
        y_valid_churn=y_valid_churn,
        y_train_price=y_train_price,
        y_valid_price=y_valid_price,
        y_pred_churn_train=y_pred_churn_train,
        y_pred_churn_valid=y_pred_churn_valid,
        y_pred_price_train=y_pred_price_train,
        y_pred_price_valid=y_pred_price_valid
    )    

    print("\n--- Evaluation Results ---")
    print(evaluation_results)
#----------------------------------------------------------------
# 7. APPLY DECISION RULES 
# ---------------------------------------------------------------
    decisions, metrics_df, financial_outcome_df = apply_decision_rules(
        data=train_all,
        churn_model=churn_model,
        price_model=price_model,
        churn_features=INPUT_FEATURES_CHURN,
        price_features=INPUT_FEATURES_PRICE,
        max_account=50_000
    )

    print("\n--- Metrics DataFrame ---")
    print(metrics_df)

    print("\n--- Financial Outcome DataFrame ---")
    print(financial_outcome_df)

if __name__ == "__main__":
    main()

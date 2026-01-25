from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_percentage_error
from metrics import calc_all_metrics, alg1, metric_for_churn, metric_for_price
from decision_logic import train_raw
import pandas as pd 
from train_churn import pipe_clf, INPUT_FEATURES_CHURN
from train_price import pipe_reg, INPUT_FEATURES_PRICE

all_metrics_churn = metric_for_churn(train_raw['__churn'], train_raw['__churn_prob'])
all_metrics_price = metric_for_price(train_raw['__price_doc'], train_raw['__price_predict'])

financial_outcome = pd.DataFrame(data = [calc_all_metrics(train_raw, max_account=50_000)],
                                 index=['train_all']).T
print(financial_outcome)
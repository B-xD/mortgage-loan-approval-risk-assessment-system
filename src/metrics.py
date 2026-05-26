#src/metrics.py 

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_percentage_error
from typing import Any
import pandas as pd 


def metric_for_price(y_true:float, y_pred:float) -> float:
    """ mean_squared_log_error. bigger is better """
    return round(-mean_squared_log_error(y_true=y_true, y_pred=y_pred), 3)


def metric_for_churn(y_true:float, y_score:float) -> float:
    """ roc_auc_score. bigger is better """
    return round(roc_auc_score(y_true=y_true, y_score=y_score), 3)

def calc_all_metrics(data: pd.DataFrame, max_account=25e3) -> Any:

    def is_credit_issued(x):
        ratio = x['__price_predict'] / x['__price_doc']
        if x['__priority'] <= 0:
            value = 0.
        elif ratio > 0.9 and ratio < 1.:
            value = x['__price_predict']
        elif ratio >= 1. and ratio < 1.1:
            value = x['__price_doc']
        else:
            value = 0.

        return value

    def calc_profit(x):
        if x['is_credit'] == 0.:
            return 0.
        elif x['__churn'] == 1:
            return - x['debt'] * 2.
        elif x['debt'] < 5:
            return x['debt'] * 0.3
        elif x['debt'] < 9:
            return x['debt'] * 0.4
        elif x['debt'] >= 9:
            return x['debt'] * 0.5

    s = (
        data
        [['__priority', '__churn', '__churn_prob', '__price_doc', '__price_predict']]
        .sort_values('__priority', ascending=False)
        .copy(True)
    )

    s['debt'] = s.apply(is_credit_issued, axis=1)
    s['debt_cum'] = s['debt'].cumsum()
    s['is_credit'] = 0
    s.loc[(s['debt'] > 0) & (s['debt_cum'] <= max_account), 'is_credit'] = 1
    s['profit'] = s.apply(calc_profit, axis=1)

    total_profit = round(s['profit'].sum(), 2)
    good_credits_count = s['is_credit'].sum()
    good_credits_debt = round(s[s['is_credit'] == 1]['debt'].sum(), 2)
    bad_credits_count = s[s['is_credit'] == 1]['__churn'].sum()
    bad_credits_losses = s[(s['is_credit'] == 1) & (s['__churn'] == 1)]['debt'].sum()

    return {
        'total_profit': total_profit,
        '%profit_issued': round(total_profit / good_credits_debt * 100, 1),
        '%issued_loans': round(good_credits_debt / max_account * 100, 2),
        'issued_loans': good_credits_debt,
        'count_good': good_credits_count,
        'count_bad': bad_credits_count,
        '%bad': round(bad_credits_count / (good_credits_count + bad_credits_count) * 100., 1),
        'churn_auc': round(roc_auc_score(y_true=s['__churn'], y_score=s['__churn_prob']), 3),
        'price_nmsle': round(-mean_squared_log_error(y_true=s['__price_doc'], y_pred=s['__price_predict']), 3),
        'price_mape': round(-mean_absolute_percentage_error(y_true=s['__price_doc'], y_pred=s['__price_predict']), 3),
    }

def alg1(x:Any) -> Any:
    '''if the probability of defaulting on a loan is less than 20%
then we can issue the loan at a certain __price_predict '''
    if x['__churn_prob'] < 0.2:
        return x['__price_predict']
    return 0.0
    
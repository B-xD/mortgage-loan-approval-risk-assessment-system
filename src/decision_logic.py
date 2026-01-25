from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_percentage_error
from load import train, valid, train_raw, test_raw
from metrics import calc_all_metrics, alg1
import pandas as pd 
from train_churn import pipe_clf, INPUT_FEATURES_CHURN
from train_price import pipe_reg, INPUT_FEATURES_PRICE

#add CHURN predictions to the data
train_raw['__churn_prob'] = pipe_clf.predict_proba(train_raw[INPUT_FEATURES_CHURN])[:, 1]
train['__churn_prob'] = pipe_clf.predict_proba(train[INPUT_FEATURES_CHURN])[:, 1]
valid['__churn_prob'] = pipe_clf.predict_proba(valid[INPUT_FEATURES_CHURN])[:, 1]
test_raw['__churn_prob'] = pipe_clf.predict_proba(test_raw[INPUT_FEATURES_CHURN])[:, 1]

#add PRICE predictions to the data 
train_raw['__price_predict'] = pipe_reg.predict(train_raw[INPUT_FEATURES_PRICE])
train['__price_predict'] = pipe_reg.predict(train[INPUT_FEATURES_PRICE])
valid['__price_predict'] = pipe_reg.predict(valid[INPUT_FEATURES_PRICE])
test_raw['__price_predict'] = pipe_reg.predict(test_raw[INPUT_FEATURES_PRICE])

train_raw['__priority'] = train_raw.apply(alg1, axis=1)
train['__priority'] = train.apply(alg1, axis=1)
valid['__priority'] = valid.apply(alg1, axis=1)
test_raw['__priority'] = test_raw.apply(alg1, axis=1)

    
metrics = pd.DataFrame(data=[calc_all_metrics(train),
                             calc_all_metrics(valid)],
                       index=['train', 'valid']).T
print(metrics)

from config import INPUT_FEATURES_CHURN, INPUT_FEATURES_PRICE
from load import train, valid, y_train_churn, y_valid_churn, train_raw, test_raw  
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from features import col_transformer_churn
from metrics import metric_for_churn

"""
train a XGBoost model for churn and make predictions 
"""

XGB_clf= XGBClassifier(random_state= 47, max_depth=3, learning_rate = 0.1 )
pipe_clf = make_pipeline(col_transformer_churn, XGB_clf)
pipe_clf.fit(train[INPUT_FEATURES_CHURN], y_train_churn)

#make predictions 
y_pred_churn_train = pipe_clf.predict_proba(train[INPUT_FEATURES_CHURN])[:, 1]
y_pred_churn_valid = pipe_clf.predict_proba(valid[INPUT_FEATURES_CHURN])[:, 1]

#evaluate scores 
churn_score_train = metric_for_churn(y_train_churn, y_pred_churn_train)
churn_score_valid = metric_for_churn(y_valid_churn, y_pred_churn_valid)

print('train:', churn_score_train)
print('valid:', churn_score_valid)


#add predictions to the data
train_raw['__churn_prob'] = pipe_clf.predict_proba(train_raw[INPUT_FEATURES_CHURN])[:, 1]
train['__churn_prob'] = pipe_clf.predict_proba(train[INPUT_FEATURES_CHURN])[:, 1]
valid['__churn_prob'] = pipe_clf.predict_proba(valid[INPUT_FEATURES_CHURN])[:, 1]
test_raw['__churn_prob'] = pipe_clf.predict_proba(test_raw[INPUT_FEATURES_CHURN])[:, 1]

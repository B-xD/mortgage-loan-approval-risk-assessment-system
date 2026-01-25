from config import INPUT_FEATURES_CHURN, INPUT_FEATURES_PRICE
from load import train, valid, y_train_price, y_valid_price, test_raw, train_raw
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from features import col_transformer_price
from metrics import metric_for_price

"""
train a XGBoost model for price and make predictions 

"""

CB_reg = CatBoostRegressor(random_state=47, max_depth = 7, learning_rate = 0.05 )
pipe_reg = make_pipeline(col_transformer_price , CB_reg)
pipe_reg.fit(train[INPUT_FEATURES_PRICE], y_train_price)

# make predictions 
y_pred_price_train = pipe_reg.predict(train[INPUT_FEATURES_PRICE])
y_pred_price_valid = pipe_reg.predict(valid[INPUT_FEATURES_PRICE])

#evaluation
price_score_train = metric_for_price(y_train_price, y_pred_price_train)
price_score_valid  = metric_for_price(y_valid_price, y_pred_price_valid)

print('train:', price_score_train)
print('train:', price_score_valid)

#add predictions to the data 
train_raw['__price_predict'] = pipe_reg.predict(train_raw[INPUT_FEATURES_PRICE])
train['__price_predict'] = pipe_reg.predict(train[INPUT_FEATURES_PRICE])
valid['__price_predict'] = pipe_reg.predict(valid[INPUT_FEATURES_PRICE])
test_raw['__price_predict'] = pipe_reg.predict(test_raw[INPUT_FEATURES_PRICE])
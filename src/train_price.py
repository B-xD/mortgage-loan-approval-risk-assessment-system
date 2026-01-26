#src/train_price.py

from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline

def train_price_model(transformed_data, feature_col, target):
    """
instantiate a pipeline with CatBoost model for price
"""

    CB_reg = CatBoostRegressor(random_state=47, max_depth = 7, learning_rate = 0.05, verbose =0)
    pipe = make_pipeline(transformed_data, CB_reg)
    pipe.fit(feature_col, target)

    return pipe 





#src/train_churn.py 

from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline


def train_churn_model(transformed_data, feature_col, target):
    """
instantiate a pipeline with a XGBoost model for churn 
"""

    XGB_clf= XGBClassifier(random_state= 47, 
                           max_depth=3, 
                           learning_rate = 0.1 )
    
    pipe= make_pipeline(transformed_data, XGB_clf)

    pipe.fit(feature_col, target)

    return pipe





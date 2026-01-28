#src/train_churn.py 

from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, Pipeline
import pandas as pd 



def train_churn_model(transformed_data: pd.DataFrame, feature_col: list, target: pd.DataFrame) -> Pipeline:
    """
instantiate a pipeline with a XGBoost model for churn 
"""

    XGB_clf= XGBClassifier(random_state= 47, 
                           max_depth=3, 
                           learning_rate = 0.1 )
    
    pipe= make_pipeline(transformed_data, XGB_clf)

    pipe.fit(feature_col, target)

    return pipe





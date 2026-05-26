#src/features.py

#import important libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from typing import List, Tuple  
from config import INPUT_FEATURES_CHURN, INPUT_FEATURES_PRICE

#========================= selecting the variables ============
#select categorical and numerical variables 

def select_variables(data: pd.DataFrame, 
                     features: List[str]) -> Tuple[List[str], List[str]]:
  """
  This function takes the data and the selected features and returns a tuple of two lists,
  one for the categorical data and another one for numerical variables 
  """
  cat_cols = data[features].select_dtypes(include=['object', 'bool']).columns.tolist()
  num_cols = data[features].select_dtypes(include=['int64', 'float64']).columns.tolist()

  return cat_cols, num_cols

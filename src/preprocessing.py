#src/preprocessing.py

#import important libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import ColumnTransformer
from typing import List  


def transformer(categorical_cols: List[str], numerical_cols: List[float]) -> ColumnTransformer:
    """This function will contain a pipeline that takes raw data, 
     imputes empty values, encodes categorical values, and scales de data 
     """
  #create a pipeline for categorical data
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy= 'most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output = False)),
    ])

#create a pipeline for numerical data
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', QuantileTransformer(output_distribution='normal')),

    ])

    #create a numerical pipeline
    col_transformer= ColumnTransformer(transformers =[
    ('categorical', categorical_pipeline, categorical_cols),
    ('numerical', numerical_pipeline, numerical_cols)
    ])


    return col_transformer

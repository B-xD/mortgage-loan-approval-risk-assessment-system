#import important libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from config import INPUT_FEATURES_CHURN, INPUT_FEATURES_PRICE
from load import train, valid, y_train_price, y_valid_price, y_train_churn, y_valid_churn 


#======================== For Churn ==========================
#select categorical and numerical data
categorical_cols_churn = valid[INPUT_FEATURES_CHURN].select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols_churn = valid[INPUT_FEATURES_CHURN].select_dtypes(include=['int64', 'float64']).columns.tolist()

#create a pipeline for categorical data
categorical_pipeline_churn = Pipeline([
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output = False)),
])

#create a numerical pipeline
numerical_pipeline_churn = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', QuantileTransformer(output_distribution='normal')),

])

#set the transformer
col_transformer_churn= ColumnTransformer(transformers =[
    ('categorical', categorical_pipeline_churn, categorical_cols_churn),
    ('numerical', numerical_pipeline_churn, numerical_cols_churn)
])

# fit pipeline
col_transformer_churn.fit(train[INPUT_FEATURES_CHURN], y_train_churn)


#======================= For Price =====================
#select categorical and numerical data  
categorical_cols_price= train[INPUT_FEATURES_PRICE].select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols_price = train[INPUT_FEATURES_PRICE].select_dtypes(include=['int64', 'float64']).columns.tolist()

#create a pipeline for categorical data

categorical_pipeline_price = Pipeline([
    ('imputer', SimpleImputer(strategy= 'most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output = False)),
])

#create a numerical pipeline
numerical_pipeline_price= Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', QuantileTransformer(output_distribution='normal')),

])

#set up the transformer
col_transformer_price = ColumnTransformer(transformers =[
    ('categorical', categorical_pipeline_price, categorical_cols_price),
    ('numerical', numerical_pipeline_price, numerical_cols_price)
])

# fit the transformer pipeline
col_transformer_price.fit(train[INPUT_FEATURES_PRICE], y_train_price)
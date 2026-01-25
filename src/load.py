from sklearn.model_selection import train_test_split

"""
Here we are going to load and split  the data

"""
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
from pathlib import Path

DATA_DIR = Path('data/raw')

train_raw = pd.read_csv(Path('data/raw/train_data.csv'))
test_raw = pd.read_csv(Path('data/raw/test_data.csv'))

#split the data 
train, valid, y_train_price, y_valid_price, y_train_churn, y_valid_churn = train_test_split(
    train_raw, train_raw['__price_doc'], train_raw['__churn'],
    test_size=0.5, random_state=42,
)

print(train.shape, valid.shape,
      y_train_price.shape, y_valid_price.shape,
      y_train_churn.shape, y_valid_churn.shape)
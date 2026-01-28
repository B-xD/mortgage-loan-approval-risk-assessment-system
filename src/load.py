#src/load.py 

from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data(test_size=0.5, random_state=42) -> pd.DataFrame:
    """
    Load raw data and split into train / validation sets
    """

    train_all = pd.read_csv(DATA_DIR / "train_data.csv")
    test = pd.read_csv(DATA_DIR / "test_data.csv")

    train, valid, y_train_price, y_valid_price, y_train_churn, y_valid_churn = train_test_split(
        train_all,
        train_all["__price_doc"],
        train_all["__churn"],
        test_size=test_size,
        random_state=random_state,
    )

    return (
        train,
        valid,
        train_all, 
        test,
        y_train_price,
        y_valid_price,
        y_train_churn,
        y_valid_churn,
    )

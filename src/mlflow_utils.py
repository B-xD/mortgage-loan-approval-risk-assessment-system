#src/mlflow_utils

#---------- Importing Libraries ----------
from contextlib import nullcontext
import logging
from typing import Any
import os
from config import price_model_params, churn_model_params, tracking_url, tracking_user, tracking_password
logger = logging.getLogger(__name__)


#--------- helpers ---------- 

def _mlflow():
    "Import mlflow and return an error if it is not imported "
    try:
        import mlflow 
        return mlflow 
    except ImportError:
        return None 

def start_run(all_churn: float,
              all_price: float,
             total_profit: float,
              tracking_url: str = tracking_url,
             name_of_experiment: str = "mortgage-risk-assessment",
             run_name: str | None = None,
             enable: bool = True, 
             ):
    
    if not enable:
        return nullcontext()
    
    ml = _mlflow()
    if ml is None:
        logger.warning(
            "mlflow is not installed - experiment tracking disabled."
            "Install it with: pip install mlflow"
        )
        return nullcontext()

    ml.set_tracking_uri(tracking_url)
    ml.set_experiment(name_of_experiment)
    
    with ml.start_run(run_name=run_name):
        ml.log_params({
    **{f"churn_{k}": v for k, v in churn_model_params.items()},
    **{f"price_{k}": v for k, v in price_model_params.items()}
})
    
        ml.log_metric("all_churn", all_churn )
        ml.log_metric("all_price", all_price)
        ml.log_metric('total_profit', total_profit)

        ml.log_artifact("models/churn_model.joblib")
        ml.log_artifact("models/price_model.joblib")



    
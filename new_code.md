# mlflow suggested code 



#---------- Importing Libraries ----------
from contextlib import nullcontext
import logging
from typing import Any
import os
from config import price_model_params, churn_model_params
logger = logging.getLogger(__name__)

#--------- helpers ---------- 

def _mlflow():
    "Import mlflow and return an error if it is not imported "
    try:
        import mlflow 
        return mlflow 
    except ImportError:
        return None 
    
def _active_run():
    "return the mlflow module only if there is an active run"
    ml = _mlflow()
    if ml and ml.active_run():
        return ml 
    else:
        return None 
    
tracking_url = "https://dagshub.com/B-xD/mortgage-loan-approval-risk-assessment-system.mlflow"

#---- Context manager for mlflow runs ----
def start_run(
        experiment_name : str = "mortgage-risk-assessment",
        run_name : str | None = None,
        tracking_url: str = tracking_url,
        enable: bool = True, 

) -> Any:
    """
    Return a context manager that wraps an MLflow run.

    When *enabled* is False, or when mlflow is not installed, returns a
    plain nullcontext() so callers can always use ``with start_run(): …``
    without any conditional logic.

    Usage::

        with mlflow_utils.start_run("my-experiment", run_name="baseline"):
            mlflow_utils.log_config(config)
            trainer.fit(train_df)
            mlflow_utils.log_metrics(valid_metrics, prefix="valid__")
    
    """

    if enable:
        return nullcontext()
    
    ml = _mlflow()
    if ml is None:
        logger.warning(
            "mlflow is not installed - experiment tracking disabled."
            "Install it with: pip isntall mlflow"
        )
        return nullcontext()
    
    ml.set_tracking_uri(tracking_url)
    ml.set_experiment(experiment_name)
    return ml.start_run(run_name= run_name)

#----- logging helpers ------------------
def log_config(config) -> None: 
    """

    Log every Config field as MLflow parameters.

    Price model params are prefixed with ``price__``,
    churn model params with ``churn__``, all other fields are flat.

    Example logged params::

        name               = Belton_Manhica
        random_state       = 47
        price__n_estimators = 300
        churn__max_depth    = 4
    
    """
    ml = _active_run()
    if ml is None:
        return 
    
    base = {
        "name": config.name,
        "random_state": config.random_state,
        "valid_size": config.valid_size,
        "max_account": config.max_account,
        "max_price_clip": config.min_price_clip,

    }

    price = {f"price_{k}": v for k, v in config.price_model_params.items()}
    churn = {f"churn_{k}": v for k, v in config.churn_model_params.items()}

    ml.log_params({**base, **price, **churn})
    logger.info("MLflow: logged %d config params", len(base)+len(price)+len(churn))

def log_dataset_info(train_df, valid_df, trainer) -> None:
    """
    log row counts and enginerring feature counts after fitting 
    """

    ml = _active_run()
    if ml is None:
        return 
    
    params = {
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_features": len(trainer.feature_names),
        "n_numeric_features": len(trainer.numeric_features),
        "n_categorical_features": len(trainer.categorical_features),
    }

    ml.log_params(params)
    logger.info("MLflow: logged dataset info (%d train rows, %d valid rows )",
                len(train_df), len(valid_df))
    

def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    """
    Log a metrics dict, optionally adding a prefix to every key.

    Non-numeric values are cast to float. Keys that cannot be cast are
    skipped with a warning rather than raising an exception.

    Args:
        metrics: dict returned by ``calc_all_metrics()`` or similar.
        prefix:  string prepended to every key, e.g. ``"valid__"``.
    """

    ml = _active_run()
    if ml is None:
        return 
    
    safe: dict[str,float] = {}
    for k, v in metrics.items():
        try:
            #MLflow only allows alphanumerics, _ -. space : /
            #Replace % with 'pct" so %bad - pct_bad, %profit_issued - pct_profit_issued
            clean_key = f"{prefix}{k}".replace("%", "pct_")
            safe[clean_key] = float(v)
        except (TypeError, ValueError):
            logger.warning("MLflow: skipping non-numeric metric %s=%r", k,v)
    
    ml.log_metrics(safe)
    logger.info("MLflow: logged %d metrics (prefix=%r)", len(safe), prefix)



def log_artifacts(output_dir: str) -> None:
    """
    Log trainer.joblib and metrics_report.json from *output_dir*.
    """

    ml = _active_run()
    if ml is None:
        return 
    
    for filename in ("trainer.joblib", "metrics_report.json"):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            ml.log_artifact(path)
            logger.info("MLflow: logged artefact %s", path)
        else:
            logger.debug("MLflow: artefact not found, skipping - %s", path)

# end of mlflow code 

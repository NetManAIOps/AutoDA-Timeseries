from .classifier import AVAILABLE_CLASSIFIERS
from .predictor import AVAILABLE_PREDICTORS
from .regressor import AVAILABLE_REGRESSORS
from .forecaster import AVAILABLE_LONG_TERM_FORECASTER
from .anomalyor import AVAILABLE_ANOMALYORS
from utils import GlobalConfig
from typing import Dict, List, Type
from .downstream_base import DownstreamModelBase

AVAILABLE_DOWNSTREAM: Dict[str, Dict[str,Type[DownstreamModelBase]]]={
    "classification": AVAILABLE_CLASSIFIERS,
    "prediction": AVAILABLE_PREDICTORS,
    "regression": AVAILABLE_REGRESSORS,
    "long_term_forecasting": AVAILABLE_LONG_TERM_FORECASTER,
    "anomaly_detection":AVAILABLE_ANOMALYORS
}

def build_downstream_model(task:str, downstream:str, global_config:GlobalConfig) -> DownstreamModelBase:
    if task not in AVAILABLE_DOWNSTREAM:
        raise NotImplementedError(f"Unknown task:{task}(options:{list(AVAILABLE_DOWNSTREAM.keys())})")
    task_downstream: Dict[str,Type[DownstreamModelBase]] = AVAILABLE_DOWNSTREAM[task]
    if downstream not in task_downstream:
        raise NotImplementedError(f"Unknown downstream:{downstream}(options:{list(task_downstream.keys())})")
    downstream_class = task_downstream[downstream]
    return downstream_class(global_config)
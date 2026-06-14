from utils import GlobalConfig
from .exp_classification import ExpClassification
from .exp_regression import ExpRegression
from .exp_long_term_forecasting import ExpLongTermForecasting
from .exp_anomaly_detection import ExpAnomalyDetection
from .exp_basic import ExpBasic
from typing import Type


AVAILABLE_EXP = {
    "classification": ExpClassification,
    "regression": ExpRegression,
    "long_term_forecasting": ExpLongTermForecasting,
    "anomaly_detection": ExpAnomalyDetection
}

def build_exp(global_config:GlobalConfig) -> ExpBasic:
    if global_config.args.task not in AVAILABLE_EXP:
        raise NotImplementedError(f"Unknown experiment task: "
                                  f"{global_config.args.task} ([options] {'/'.join(AVAILABLE_EXP.keys())})")
    exp_class: Type[ExpBasic] = AVAILABLE_EXP[global_config.args.task]
    exp:ExpBasic = exp_class(global_config)
    return exp
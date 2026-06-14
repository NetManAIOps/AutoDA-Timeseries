from .dataset_classification import AVAILABLE_CLASSIFICATION_DATASETS
from .dataset_regression import AVAILABLE_REGRESSION_DATASETS
from .dataset_long_term_forecasting import AVAILABLE_LONG_TERM_FORECASTING_DATASETS
from .dataset_anomaly_detection import AVAILABLE_ANOMALY_DETECTION_DATASETS
from .dataset_basic import BasicDataset
from torch.utils.data import DataLoader
from utils import GlobalConfig


AVAILABLE_DATASETS={
    "classification":AVAILABLE_CLASSIFICATION_DATASETS,
    "regression":AVAILABLE_REGRESSION_DATASETS,
    "long_term_forecasting":AVAILABLE_LONG_TERM_FORECASTING_DATASETS,
    "anomaly_detection":AVAILABLE_ANOMALY_DETECTION_DATASETS
}

def get_dataset(config: GlobalConfig, flag:str) -> tuple[BasicDataset, DataLoader]:
    if config.args.task not in AVAILABLE_DATASETS:
        raise NotImplementedError(f"Unknown task [{config.args.task}] (options:{AVAILABLE_DATASETS.keys()})")

    default = AVAILABLE_DATASETS[config.args.task]["Default"]
    dataset:BasicDataset = AVAILABLE_DATASETS[config.args.task].get(config.args.dataset_type, default)(config, flag)

    batch_size = config.args.batch_size
    if batch_size<0:
        print(f"batch_size is set to the whole dataset {dataset.n_samples}")
        batch_size = dataset.n_samples
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if (flag == 'test' or flag == 'TEST') else True,
        num_workers=config.args.num_workers,
        drop_last=False)
    return dataset, data_loader

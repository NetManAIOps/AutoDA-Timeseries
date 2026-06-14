import numpy as np
import torch
from aeon.datasets import load_regression, load_covid_3month
import aeon
from utils import GlobalConfig
from utils.socks_proxy import ProxyContext
from .dataset_basic import BasicDataset
from .tsfeature_extractors import get_feature_extractor, FeatureExtractor


class DefaultRegressionDataset(BasicDataset):
    def __init__(self, config:GlobalConfig, flag):
        super().__init__(config, flag)

    def _load_data(self, root_path, dataset, flag):
        assert flag in {"TRAIN","TEST"}
        print(f"[{self.__class__.__name__}] loading data (flag={flag})")
        # (batch, n_channel, seq_len) and (batch, n_channel, 1)
        with ProxyContext(self.config):
            self.X, self.Y, metadata = load_regression(dataset, split=flag, extract_path=root_path, load_no_missing=True,
                                             return_metadata=True, load_equal_length=True)
            self.X = np.array(self.X)
            self.Y = np.array(self.Y)
            if len(self.Y.shape)==1:
                self.Y = self.Y.reshape(-1,1)

        print(f"[{self.__class__.__name__}] data loaded (flag={flag}) input size={self.X.shape} output size={self.Y.shape}")

        self.fe:FeatureExtractor = get_feature_extractor(self.config.args.feature_extractor)(self.config, None)
        # (batch, n_channel, cond_dim)
        self.F: np.ndarray = self.fe(self.X)

        self.X = torch.nan_to_num(torch.from_numpy(self.X), nan=0)
        self.Y = torch.nan_to_num(torch.from_numpy(self.Y), nan=0)
        self.F = torch.nan_to_num(torch.from_numpy(self.F), nan=0)
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)
        self._parse_dimensions()
        return self.X, self.Y, self.F, self.mask

AVAILABLE_REGRESSION_DATASETS={
    "Default":DefaultRegressionDataset
}
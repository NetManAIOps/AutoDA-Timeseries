from abc import ABC, abstractmethod
import pycatch22 as catch22
from joblib import Parallel, delayed
import numpy as np
import os

class FeatureExtractor(ABC):
    def __init__(self, config, cond_dim):
        self.cond_dim = cond_dim
        self.config = config

    def __call__(self, x):
        """
        :param x: input timeseries (batch, n_channels, seq_len)
        :return: feature vectors (batch, n_channels, cond_dim)
        """
        n_threads = self.config.args.num_workers if self.config.args.num_workers>0 else os.cpu_count()
        print(f"[{self.__class__.__name__}] Extracting features (thread={n_threads})")
        features = self._extract(x, n_threads=n_threads)
        features = self._adjust_feature_dim(features)
        features = np.nan_to_num(features, nan=0)
        print(f"[{self.__class__.__name__}] Features extracted (dim={features.shape})")
        return features

    def get_cond_dim(self):
        return self.cond_dim

    def _adjust_feature_dim(self, features):
        if self.cond_dim and features.shape[2] > self.cond_dim:
            features = features[:, :, :self.cond_dim]
        return features

    @abstractmethod
    def _extract(self, X, n_threads):
        """
        :param x: input timeseries (batch, n_channels, seq_len)
        :param n_threads: threads_to_use in feature extraction
        :return: feature vectors (batch, n_channels, cond_dim)
        """
        raise NotImplementedError(f"FeatureExtractor {self.__class__.__name__}.__call__ not implemented")

class Catch22FeatureExtractor(FeatureExtractor):
    def __init__(self, config, cond_dim):
        super().__init__(config, cond_dim if cond_dim else 24)

    def _extract(self, x, n_threads:int=1):
        """
        :param x: input timeseries (batch, n_channels, seq_len)
        :return: feature vectors (batch, n_channels, cond_dim)
        """
        batch, n_channels, seq_len = x.shape

        def compute_features(x):
            res = catch22.catch22_all(x, catch24=True)
            res = np.array(res['values'])
            res = np.nan_to_num(res, nan=0.0)
            return res  # just return the values

        ts_data = x.reshape((-1, seq_len))
        results_list = Parallel(n_jobs=n_threads)(
            delayed(compute_features)(ts_data[i]) for i in range(len(ts_data))
        )

        return np.vstack(results_list).reshape((batch, n_channels, 24))

class NoFeatureExtractor(FeatureExtractor):
    def __init__(self, config, cond_dim):
        super().__init__(config, cond_dim if cond_dim else 1)

    def _extract(self, x, n_threads:int=1):
        """
        :param x: input timeseries (batch, n_channels, seq_len)
        :return: feature vectors (batch, n_channels, cond_dim)
        """
        batch, n_channels, seq_len = x.shape
        return np.zeros((batch, n_channels, 1))

AVAILABLE_FEATURE_EXTRACTORS={
    "Catch22": Catch22FeatureExtractor,
    "No": NoFeatureExtractor,
    "Default": Catch22FeatureExtractor
}

def get_feature_extractor(name:str):
    return AVAILABLE_FEATURE_EXTRACTORS.get(name, AVAILABLE_FEATURE_EXTRACTORS["Default"])



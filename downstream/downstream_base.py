import torch.nn as nn
from abc import ABC, abstractmethod

from utils import GlobalConfig


class DownstreamModelBase(nn.Module, ABC):
    def __init__(self, global_config: GlobalConfig):
        super().__init__()
        self.global_config = global_config
        self.configs = self.global_config.args
        self.downstream_args:dict = self.global_config.downstream_args
        self.n_channels = self.configs.n_channels
        self.seq_len = self.configs.seq_len
        self.n_features = self.configs.n_features
        self.pred_len = self.configs.pred_len
        self._build_model()

    @abstractmethod
    def _build_model(self):
        return NotImplementedError()

    @abstractmethod
    def forward(self, batch_x, batch_f, batch_mask):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_f: (batch, n_channels, n_features)
        :param batch_mask: (batch, seq_len)
        :return: (batch, pred_len)
        """
        raise NotImplementedError()
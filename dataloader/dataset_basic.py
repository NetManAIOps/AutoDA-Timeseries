from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import GlobalConfig


class BasicDataset(Dataset, ABC):
    def __init__(self, config:GlobalConfig, flag):
        super().__init__()
        assert flag in {"TRAIN","TEST", "VALI"}
        self.config = config
        self.n_classes = -1 # only used in classification tasks
        self.X, self.Y, self.F, self.mask = self._load_data(self.config.args.dataset_root,
                                                            self.config.args.dataset,
                                                            flag)

        self._parse_dimensions()

    def _parse_dimensions(self):
        self.n_features = self.F.shape[-1]
        self.n_samples, self.n_channels, self.seq_len = self.X.shape
        self.pred_len = self.Y.shape[-1] if len(self.Y.shape)>1 else 1

    @abstractmethod
    def _load_data(self, root_path, dataset, flag):
        raise NotImplementedError()

    def extend_seq_len(self, assigned_len):
        batch, channel, seq_len = self.X.shape
        # 1. extend tensor
        if assigned_len > seq_len:
            # if assigned_len > seq_len
            padding = assigned_len - seq_len
            x_extended = F.pad(self.X, (0, padding))
        else:
            # if assigned_len <= seq_len, truncate the sequence.
            x_extended = self.X[:, :, :assigned_len]

        # 2. create mask
        mask = torch.ones((batch, assigned_len), dtype=torch.bool, device=self.X.device)
        if assigned_len > seq_len:
            mask[:, seq_len:] = False

        self.X, self.mask = x_extended, mask
        self._parse_dimensions()

    def __getitem__(self, ind):
        return self.X[ind], self.Y[ind], self.F[ind], self.mask[ind]

    def __len__(self):
        return self.n_samples

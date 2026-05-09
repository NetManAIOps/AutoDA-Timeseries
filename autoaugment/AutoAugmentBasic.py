from abc import ABC, abstractmethod
import torch.nn as nn

from downstream import DownstreamModelBase
from utils import GlobalConfig


class AutoAugmentBasic(nn.Module, ABC):
    def __init__(self, global_config: GlobalConfig):
        super().__init__()
        self.global_config = global_config
        self.configs = self.global_config.args
        self._initialize()
        print(f"[{self.__class__.__name__}] Downstream model is {self.configs.downstream}")


    def _initialize(self):
        return

    @abstractmethod
    def _perform_augment(self, batch_x, batch_y, batch_f, batch_masks):
        raise NotImplementedError()

    def forward(self, batch_x, batch_y, batch_f, batch_masks):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len)
        :param batch_f: (batch, n_channels, n_features)
        :param batch_masks: (batch, seq_len)
        :return: output_y, augmented_y, augmented_mask
        """
        if self.training:
            augmented_x, augmented_y, augmented_masks = self._perform_augment(batch_x, batch_y, batch_f, batch_masks)
        else:
            augmented_x, augmented_y, augmented_masks = batch_x, batch_y, batch_masks
            self._perform_augment(batch_x, batch_y, batch_f, batch_masks)
        task = getattr(self.configs, "task", "")
        is_forecasting = task in ("long_term_forecasting", "short_term_forecasting")

        if is_forecasting:
            return self.downstream_model(augmented_x, batch_y, batch_f, batch_masks), augmented_y, augmented_masks
        else:
            return self.downstream_model(augmented_x, batch_f, augmented_masks), augmented_y, augmented_masks
        # return self.downstream_model(augmented_x, batch_f, augmented_masks), augmented_y, augmented_masks

    def get_criterion(self, default_criterion):
        return default_criterion

    def summarize_state(self) -> str:
        """
        :return: msg to print
        """
        return ""
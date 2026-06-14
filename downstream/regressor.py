import torch.nn as nn
import torch.nn.functional as F
from sktime.regression.deep_learning import CNNRegressor

from .downstream_base import DownstreamModelBase


class MLPRegressor(DownstreamModelBase):
    def _build_model(self):
        # Flatten the input
        self.flatten = nn.Flatten()

        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.n_channels * self.seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_len)
        )

    def forward(self, batch_x, batch_f, batch_mask):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_f: (batch, n_channels, n_features)
        :param batch_mask: (batch, seq_len)
        :return: (batch, pred_len)
        """
        # Flatten the input
        x = self.flatten(batch_x)

        # Apply the MLP
        output = self.mlp(x)

        return output


class CNNRegressor(DownstreamModelBase):
    def _build_model(self):
        self.conv1 = nn.Conv1d(self.n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # 1x1 convolution for channel reduction
        self.conv1x1 = nn.Conv1d(256, 64, kernel_size=1)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(64, self.pred_len)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, batch_x, batch_f, batch_mask):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_f: (batch, n_channels, n_features)
        :param batch_mask: (batch, seq_len)
        :return: (batch, pred_len)
        """
        x = F.relu(self.conv1(batch_x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Channel reduction
        x = F.relu(self.conv1x1(x))

        # Global average pooling
        x = self.gap(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dropout
        x = self.dropout(x)

        # Final prediction
        x = self.fc(x)
        return x

AVAILABLE_REGRESSORS = {
    "MLP":MLPRegressor,
    "CNN":CNNRegressor,
}
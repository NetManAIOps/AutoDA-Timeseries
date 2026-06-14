import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

# from utils.timefeatures import time_features
from utils import GlobalConfig

from .tsfeature_extractors import get_feature_extractor, FeatureExtractor
from .dataset_basic import BasicDataset
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
import os

def _sliding_windows(arr_2d: np.ndarray, win_size: int, step: int=1):
    """
    arr_2d: [T, D] -> [N, win_size, D]
    """
    T, D = arr_2d.shape
    if T < win_size:
        pad = np.zeros((win_size - T, D), dtype=arr_2d.dtype)
        arr_2d = np.concatenate([arr_2d, pad], axis=0)
        T = win_size
    
    starts = np.arange(0, T - win_size + 1, step, dtype=np.int64)
    if len(starts) == 0:
        starts = np.array([0], dtype=np.int64)
    windows = np.stack([arr_2d[s:s+win_size] for s in starts], axis=0)
    return windows


# class Dataset_MSL(BasicDataset):
#     """
#     MSL_train.npy: [T_train, n_channels]
#     MSL_test.npy: [T_test, n_channels]
#     MSL_test_label.npy: [T_test, ]
#     """

#     def __init__(self, config: GlobalConfig, flag: str):
#         self.root_path = config.args.dataset_root
#         self.data_path = getattr(config.args, "dataset", "MSL")
#         self.flag = flag
#         self.scaler = StandardScaler()

        
#         self.win_size = getattr(config.args, "seq_len", 100)
#         self.step = getattr(config.args, "step", 1)




#         super().__init__(config, flag)
    
#     def _load_data(self, root_path, dataset, flag):
        
#         flag = flag.upper()
#         assert flag in {"TRAIN", "TEST", "VALI"}

#         print(f"[{self.__class__.__name__}] loading data (flag={flag})")

#         train_path = os.path.join(self.root_path, self.data_path, "MSL_train.npy")
#         test_path = os.path.join(self.root_path, self.data_path, "MSL_test.npy")
#         label_path = os.path.join(self.root_path, self.data_path, "MSL_test_label.npy")

#         for p in (train_path, test_path, label_path):
#             if not os.path.exists(p):
#                 raise FileNotFoundError(f"[MSLAnomalyDataset] Missing file: {p}")
        
#         train_raw = np.load(train_path) # [T_train, n_channels]
#         test_raw = np.load(test_path) # [T_test, n_channels]
#         test_lbl_raw = np.load(label_path) # [T_test, ]

#         train_std = self.scaler.fit_transform(train_raw)
#         test_std = self.scaler.transform(test_raw)

        

#         if flag == "TRAIN":
#             T = train_std.shape[0]
#             split = int(T*0.05)
#             src = train_std[:split] if split > 0 else train_std
#             X_win = _sliding_windows(src, self.win_size, self.step) # [N, seq_len, n_channels]
#             Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32) 
#             # print(f"X_win: {X_win.shape}")
#             # print(f"Y_win: {Y_win.shape}")
#         elif flag == "VALI":
#             T = train_std.shape[0]
#             split = int(T*0.005)
#             print(f"T - split = {T - split}")
#             print(f"win_size: {self.win_size}")
#             src = train_std[:split] if (T - split) >= self.win_size else train_std
#             X_win = _sliding_windows(src, self.win_size, self.step)
#             print(f"x_win_valid.size: {X_win.shape}")
#             Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)
#         # else: # TEST
#         #     X_win = _sliding_windows(test_std, self.win_size, self.step)
#         #     lbl = test_lbl_raw
#         #     if lbl.ndim == 2 and lbl.shape[1] == 1:
#         #         lbl = lbl[:, 0]
#         #     elif lbl.ndim == 2 and lbl.shape[1] > 1:
#         #         lbl = (lbl != 0).any(axis=1).astype(np.float32)
#         #     lbl = lbl.astype(np.float32)
#         #     starts = np.arange(0, len(lbl)-self.win_size+1, self.step, dtype=np.int64)
#         #     if len(starts) == 0:
#         #         lbl_aligned = np.zeros((1, self.win_size), dtype=np.float32)
#         #         lbl_aligned[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
#         #     else:
#         #         lbl_aligned = np.stack([lbl[s:s+self.win_size] for s in starts], axis=0).astype(np.float32)
            
#         #     Y_win = lbl_aligned # [N, L]

#         else:  # TEST
#             # ===== Configurable shortening parameters. Adjust one or more as needed. =====
#             TEST_RATIO      = 0.005   # Ratio crop. For example, 0.2 keeps the first 20%; None disables ratio cropping.
#             TEST_MAX_LEN    = 10000  # Fixed maximum length. For example, 10000; None disables the limit.
#             TEST_DOWNSAMPLE = None   # Downsampling step. For example, 5 keeps one point every 5 points; None disables downsampling.
#             # ===========================================

#             def _shrink_series_and_label(series, label, ratio=None, max_len=None, downsample=None):
#                 """
#                 series: [T, C] test sequence after standardization
#                 label:  [T] or [T,1] or [T,K] raw test labels
#                 return: shortened/downsampled (series2, label2)
#                 """
#                 T = series.shape[0]
#                 # 1) Ratio crop.
#                 if ratio is not None:
#                     keep = max(1, int(T * float(ratio)))
#                     series = series[:keep]
#                     label  = label[:keep]
#                     T = keep
#                 # 2) Fixed-length crop.
#                 if max_len is not None:
#                     keep = min(T, int(max_len))
#                     series = series[:keep]
#                     label  = label[:keep]
#                     T = keep
#                 # 3) Downsample.
#                 if downsample is not None and int(downsample) > 1:
#                     ds = int(downsample)
#                     series = series[::ds]
#                     label  = label[::ds]
#                 return series, label

#             # ---- Shorten test data and labels first. ----
#             test_std_eff, lbl_eff = _shrink_series_and_label(
#                 test_std, test_lbl_raw,
#                 ratio=TEST_RATIO,
#                 max_len=TEST_MAX_LEN,
#                 downsample=TEST_DOWNSAMPLE
#             )

#             # ---- Sliding-window samples. ----
#             X_win = _sliding_windows(test_std_eff, self.win_size, self.step)  # [N, L, C]
#             # print(f"X_win_test: {X_win.shape}")

#             # ---- Label handling, same as the original logic. ----
#             lbl = lbl_eff
#             if lbl.ndim == 2 and lbl.shape[1] == 1:
#                 lbl = lbl[:, 0]
#             elif lbl.ndim == 2 and lbl.shape[1] > 1:
#                 # Multi-channel or multi-label: any nonzero value is treated as anomaly.
#                 lbl = (lbl != 0).any(axis=1).astype(np.float32)
#             lbl = lbl.astype(np.float32)

#             # ---- Align with sliding windows. ----
#             starts = np.arange(0, len(lbl) - self.win_size + 1, self.step, dtype=np.int64)
#             if len(starts) == 0:
#                 lbl_aligned = np.zeros((1, self.win_size), dtype=np.float32)
#                 take = min(self.win_size, len(lbl))
#                 lbl_aligned[0, :take] = lbl[:take]
#             else:
#                 lbl_aligned = np.stack([lbl[s:s + self.win_size] for s in starts], axis=0).astype(np.float32)

#             Y_win = lbl_aligned  # [N, L]

#         X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32) # [N, D, L]
#         Y_np = Y_win[:, None, :].astype(np.float32) # [N, 1, L]

#         F_np = torch.randn(X_np.shape[0], X_np.shape[1], 24)

#         self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
#         self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
#         if isinstance(F_np, np.ndarray):
#             F = torch.from_numpy(F_np)
#         else:
#             F = F_np
        
#         self.F = torch.nan_to_num(F, nan=0.0)

#         # mask: [N, L]
#         self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

#         self.n_classes = 2
#         self._parse_dimensions()

#         print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")

#         return self.X, self.Y, self.F, self.mask


# class Dataset_MSL(BasicDataset):
#     """
#     MSL_train.npy: [T_train, n_channels]
#     MSL_test.npy: [T_test, n_channels]
#     MSL_test_label.npy: [T_test, ]
#     """

#     def __init__(self, config: GlobalConfig, flag: str):
#         self.root_path = config.args.dataset_root
#         self.data_path = getattr(config.args, "dataset", "MSL")
#         self.flag = flag
#         self.scaler = StandardScaler()

#         self.win_size = getattr(config.args, "seq_len", 100)
#         self.step = getattr(config.args, "step", 1)

#         super().__init__(config, flag)
    
#     def _load_data(self, root_path, dataset, flag):
#         assert dataset.upper() == "MSL", f"Dataset_MSL only support dataset_type = MSL, receive {dataset}"
#         flag = flag.upper()
#         assert flag in {"TRAIN", "TEST", "VALI"}

#         print(f"[{self.__class__.__name__}] loading data (flag={flag})")

#         train_path = os.path.join(self.root_path, self.data_path, "MSL_train.npy")
#         test_path = os.path.join(self.root_path, self.data_path, "MSL_test.npy")
#         label_path = os.path.join(self.root_path, self.data_path, "MSL_test_label.npy")

#         for p in (train_path, test_path, label_path):
#             if not os.path.exists(p):
#                 raise FileNotFoundError(f"[MSLAnomalyDataset] Missing file: {p}")
        
#         train_raw = np.load(train_path) # [T_train, n_channels]
#         # train_raw = train_raw[:, 10]

#         test_raw = np.load(test_path) # [T_test, n_channels]
#         # test_raw = test_raw[:, 10]

#         test_lbl_raw = np.load(label_path) # [T_test, ]

#         train_std = self.scaler.fit_transform(train_raw)
#         test_std = self.scaler.transform(test_raw)
        
#         # [N, C, L]
#         X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32) # [N, D, L]
#         Y_np = Y_win[:, None, :].astype(np.float32) # [N, 1, L]

#         F_np = torch.randn(X_np.shape[0], X_np.shape[1], 24)

#         # Remove NaN
#         self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
#         self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
#         if isinstance(F_np, np.ndarray):
#             F = torch.from_numpy(F_np)
#         else:
#             F = F_np
        
#         self.F = torch.nan_to_num(F, nan=0.0)

#         # mask: [N, L]
#         self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

#         self.n_classes = 2
#         self._parse_dimensions()

#         print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")

#         return self.X, self.Y, self.F, self.mask

class Dataset_MSL(BasicDataset):
    """
    MSL_train.npy: [T_train, n_channels]
    MSL_test.npy: [T_test, n_channels]
    MSL_test_label.npy: [T_test, ]
    """

    def __init__(self, config: GlobalConfig, flag: str):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "MSL")
        self.flag = flag
        self.scaler = StandardScaler()

        
        self.win_size = getattr(config.args, "seq_len", 100)
        self.step = getattr(config.args, "step", 1)




        super().__init__(config, flag)
    
    def _load_data(self, root_path, dataset, flag):
        
        flag = flag.upper()
        assert flag in {"TRAIN", "TEST", "VALI"}

        print(f"[{self.__class__.__name__}] loading data (flag={flag})")

        train_path = os.path.join(self.root_path, self.data_path, "MSL_train.npy")
        test_path = os.path.join(self.root_path, self.data_path, "MSL_test.npy")
        label_path = os.path.join(self.root_path, self.data_path, "MSL_test_label.npy")

        for p in (train_path, test_path, label_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[MSLAnomalyDataset] Missing file: {p}")
        
        train_raw = np.load(train_path) # [T_train, n_channels]
        test_raw = np.load(test_path) # [T_test, n_channels]
        test_lbl_raw = np.load(label_path) # [T_test, ]

        train_std = self.scaler.fit_transform(train_raw)
        test_std = self.scaler.transform(test_raw)

        

        if flag == "TRAIN":
            T = train_std.shape[0]
            split = int(T*0.8)
            src = train_std[:split] if split > 0 else train_std
            X_win = _sliding_windows(src, self.win_size, self.step) # [N, seq_len, n_channels]
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32) 
            # print(f"X_win: {X_win.shape}")
            # print(f"Y_win: {Y_win.shape}")
        elif flag == "VALI":
            T = train_std.shape[0]
            split = int(T*0.8)
            src = train_std[split:] if (T - split) >= self.win_size else train_std
            X_win = _sliding_windows(src, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)
        else: # TEST
            X_win = _sliding_windows(test_std, self.win_size, self.step)
            lbl = test_lbl_raw
            if lbl.ndim == 2 and lbl.shape[1] == 1:
                lbl = lbl[:, 0]
            elif lbl.ndim == 2 and lbl.shape[1] > 1:
                lbl = (lbl != 0).any(axis=1).astype(np.float32)
            lbl = lbl.astype(np.float32)
            starts = np.arange(0, len(lbl)-self.win_size+1, self.step, dtype=np.int64)
            if len(starts) == 0:
                lbl_aligned = np.zeros((1, self.win_size), dtype=np.float32)
                lbl_aligned[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
            else:
                lbl_aligned = np.stack([lbl[s:s+self.win_size] for s in starts], axis=0).astype(np.float32)
            
            Y_win = lbl_aligned # [N, L]

        X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32) # [N, D, L]
        Y_np = Y_win[:, None, :].astype(np.float32) # [N, 1, L]

        F_np = torch.randn(X_np.shape[0], X_np.shape[1], 24)

        self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
        self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
        if isinstance(F_np, np.ndarray):
            F = torch.from_numpy(F_np)
        else:
            F = F_np
        
        self.F = torch.nan_to_num(F, nan=0.0)

        # mask: [N, L]
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

        self.n_classes = 2
        self._parse_dimensions()

        print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")

        return self.X, self.Y, self.F, self.mask


class Dataset_Custom(BasicDataset):
    pass


class Dataset_SMAP(BasicDataset):
    """
    SMAP_train.npy: 
    """
    def __init__(self, config: GlobalConfig, flag: str):

        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "SMAP")
        self.flag = flag
        self.scaler = StandardScaler()

        self.win_size = getattr(config.args, "seq_len", 100)
        self.step = getattr(config.args, "step", 1)
        super().__init__(config, flag)
    

    def _load_data(self, root_path, dataset, flag):
        assert dataset.upper() == "SMAP", f"Dataset_SMAP only support dataset_type=SMAP, receive {dataset}"
        flag = flag.upper()
        assert flag in {"TRAIN", "TEST", "VALI"}

        print(f"[{self.__class__.__name__}] loading data (flag={flag})")

        train_path = os.path.join(self.root_path, self.data_path, "SMAP_train.npy")
        test_path  = os.path.join(self.root_path, self.data_path, "SMAP_test.npy")
        label_path = os.path.join(self.root_path, self.data_path, "SMAP_test_label.npy")

        for p in (train_path, test_path, label_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[Dataset_SMAP] Missing file: {p}")

        train_raw = np.load(train_path)  # [T_train, C]
        test_raw  = np.load(test_path)   # [T_test,  C]
        test_lbl_raw = np.load(label_path)  # [T_test,] or [T_test, K]

        train_std = self.scaler.fit_transform(train_raw)
        test_std  = self.scaler.transform(test_raw)


        if flag == "TRAIN":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        elif flag == "VALI":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        else:  # TEST
            X_win = _sliding_windows(test_std, self.win_size, self.step)

            lbl = test_lbl_raw
            if lbl.ndim == 2 and lbl.shape[1] == 1:
                lbl = lbl[:, 0]
            elif lbl.ndim == 2 and lbl.shape[1] > 1:
                lbl = (lbl != 0).any(axis=1).astype(np.float32)
            lbl = lbl.astype(np.float32)

            starts = np.arange(0, len(lbl) - self.win_size + 1, self.step)
            if len(starts) == 0:
                Y_win = np.zeros((1, self.win_size), dtype=np.float32)
                Y_win[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
            else:
                Y_win = np.stack(
                    [lbl[s:s + self.win_size] for s in starts],
                    axis=0
                ).astype(np.float32)


        # lbl = test_lbl_raw
        # if lbl.ndim == 2 and lbl.shape[1] == 1:
        #     lbl = lbl[:, 0]
        # elif lbl.ndim == 2 and lbl.shape[1] > 1:
        #     lbl = (lbl != 0).any(axis=1).astype(np.float32)
        # lbl = lbl.astype(np.float32)

        X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32)   # [N, C, L]
        Y_np = Y_win[:, None, :].astype(np.float32)                # [N, 1, L]

        F_np = np.random.randn(X_np.shape[0], X_np.shape[1], 24).astype(np.float32)  # [N, C, 24]

        # Remove NaN
        self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
        self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
        self.F = torch.nan_to_num(torch.from_numpy(F_np), nan=0.0)

        # mask: [N, L]
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

        self.n_classes = 2
        self._parse_dimensions()

        print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")
        return self.X, self.Y, self.F, self.mask


class Dataset_SMD(BasicDataset):
    def __init__(self, config: GlobalConfig, flag: str):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "SMD")
        self.flag = flag.upper()
        assert self.flag in {"TRAIN", "VALI", "TEST"}
        self.scaler = StandardScaler()

        self.win_size = getattr(config.args, "seq_len", 100)
        self.step = getattr(config.args, "step", 1)

        super().__init__(config, flag)

    def _load_data(self, root_path, dataset, flag):
        assert dataset.upper() == "SMD", f"Dataset_SMD only support dataset=SMD, receive {dataset}"
        flag = flag.upper()

        print(f"[{self.__class__.__name__}] loading data (flag={flag})")

        train_path = os.path.join(self.root_path, self.data_path, "SMD_train.npy")
        test_path  = os.path.join(self.root_path, self.data_path, "SMD_test.npy")
        label_path = os.path.join(self.root_path, self.data_path, "SMD_test_label.npy")

        for p in (train_path, test_path, label_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[Dataset_SMD] Missing file: {p}")

        train_raw = np.load(train_path)  # [T_train, C]
        test_raw  = np.load(test_path)   # [T_test,  C]
        test_lbl_raw = np.load(label_path)  # [T_test,] or [T_test, K]

        train_std = self.scaler.fit_transform(train_raw)
        test_std  = self.scaler.transform(test_raw)


        if flag == "TRAIN":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        elif flag == "VALI":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        else:  # TEST
            X_win = _sliding_windows(test_std, self.win_size, self.step)

            lbl = test_lbl_raw
            if lbl.ndim == 2 and lbl.shape[1] == 1:
                lbl = lbl[:, 0]
            elif lbl.ndim == 2 and lbl.shape[1] > 1:
                lbl = (lbl != 0).any(axis=1).astype(np.float32)
            lbl = lbl.astype(np.float32)

            starts = np.arange(0, len(lbl) - self.win_size + 1, self.step)
            if len(starts) == 0:
                Y_win = np.zeros((1, self.win_size), dtype=np.float32)
                Y_win[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
            else:
                Y_win = np.stack(
                    [lbl[s:s + self.win_size] for s in starts],
                    axis=0
                ).astype(np.float32)

        # lbl = test_lbl_raw
        # if lbl.ndim == 2 and lbl.shape[1] == 1:
        #     lbl = lbl[:, 0]
        # elif lbl.ndim == 2 and lbl.shape[1] > 1:
        #     lbl = (lbl != 0).any(axis=1).astype(np.float32)
        # lbl = lbl.astype(np.float32)

        X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32)  # [N, C, L]
        Y_np = Y_win[:, None, :].astype(np.float32)               # [N, 1, L]

        F_np = np.random.randn(X_np.shape[0], X_np.shape[1], 24).astype(np.float32)  # [N, C, 24]

        self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
        self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
        self.F = torch.nan_to_num(torch.from_numpy(F_np), nan=0.0)

        # mask: [N, L]
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

        self.n_classes = 2
        self._parse_dimensions()

        print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")
        return self.X, self.Y, self.F, self.mask


class Dataset_PSM(BasicDataset):
    def __init__(self, config: GlobalConfig, flag: str):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "PSM")
        self.flag = flag.upper()
        assert self.flag in {"TRAIN", "VALI", "TEST"}
        self.scaler = StandardScaler()

        self.win_size = getattr(config.args, "seq_len", 100)
        self.step = getattr(config.args, "step", 1)
        super().__init__(config, flag)
    
    def _load_data(self, root_path, dataset, flag):
        assert dataset.upper() == "PSM", f"Dataset_PSM only support dataset=PSM, receive {dataset}"
        flag = flag.upper()

        print(f"[{self.__class__.__name__}] loading data (flag={flag})")

        train_csv = os.path.join(self.root_path, self.data_path, "PSM_train.csv")
        test_csv  = os.path.join(self.root_path, self.data_path, "PSM_test.csv")
        label_csv = os.path.join(self.root_path, self.data_path, "PSM_test_label.csv")

        for p in (train_csv, test_csv, label_csv):
            if not os.path.exists(p):
                raise FileNotFoundError(f"[Dataset_PSM] Missing file: {p}")

        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)
        train_raw = train_df.values[:, 1:].astype(np.float32)  # [T_train, n_channels]
        test_raw  = test_df.values[:, 1:].astype(np.float32)   # [T_test,  n_channels]

        train_raw = np.nan_to_num(train_raw)
        test_raw  = np.nan_to_num(test_raw)

        lbl_df = pd.read_csv(label_csv)
        lbl_raw = lbl_df.values[:, 1:]
        lbl_raw = np.nan_to_num(lbl_raw)

        train_std = self.scaler.fit_transform(train_raw)
        test_std  = self.scaler.transform(test_raw)

        if flag == "TRAIN":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        elif flag == "VALI":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        else:  # TEST
            X_win = _sliding_windows(test_std, self.win_size, self.step)

            lbl = lbl_raw
            if lbl.ndim == 2 and lbl.shape[1] == 1:
                lbl = lbl[:, 0]
            elif lbl.ndim == 2 and lbl.shape[1] > 1:
                lbl = (lbl != 0).any(axis=1).astype(np.float32)
            lbl = lbl.astype(np.float32)

            starts = np.arange(0, len(lbl) - self.win_size + 1, self.step)
            if len(starts) == 0:
                Y_win = np.zeros((1, self.win_size), dtype=np.float32)
                Y_win[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
            else:
                Y_win = np.stack(
                    [lbl[s:s + self.win_size] for s in starts],
                    axis=0
                ).astype(np.float32)

        # if lbl_raw.ndim == 2 and lbl_raw.shape[1] == 1:
        #     lbl = lbl_raw[:, 0]
        # elif lbl_raw.ndim == 2 and lbl_raw.shape[1] > 1:
        #     lbl = (lbl_raw != 0).any(axis=1).astype(np.float32)
        # else:
        #     lbl = lbl_raw.astype(np.float32)
        # lbl = lbl.astype(np.float32)

        
        X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32)  # [N, C, L]
        Y_np = Y_win[:, None, :].astype(np.float32)               # [N, 1, L]

        F_np = np.random.randn(X_np.shape[0], X_np.shape[1], 24).astype(np.float32)  # [N, C, 24]

        self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
        self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
        self.F = torch.nan_to_num(torch.from_numpy(F_np), nan=0.0)

        # mask: [N, L]
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

        self.n_classes = 2
        self._parse_dimensions()

        print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")
        return self.X, self.Y, self.F, self.mask


class Dataset_SWAT(BasicDataset):
    def __init__(self, config: GlobalConfig, flag: str):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "SWAT")
        self.flag = flag.upper()
        assert self.flag in {"TRAIN", "VALI", "TEST"}
        self.scaler = StandardScaler()

        self.win_size = getattr(config.args, "seq_len", 100)
        self.step = getattr(config.args, "step", 1)

        super().__init__(config, flag)

    def _load_data(self, root_path, dataset, flag):
        assert dataset.upper() == "SWAT", f"Dataset_SWAT only support dataset=SWAT，receive {dataset}"
        flag = flag.upper()
        print(f"[{self.__class__.__name__}] loading data (flag={flag})")

        import pandas as pd
        import numpy as np
        import os
        import torch

        train_csv = os.path.join(self.root_path, self.data_path, "swat_train2.csv")
        test_csv  = os.path.join(self.root_path, self.data_path, "swat2.csv")

        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"[Dataset_SWAT] Missing file: {train_csv}")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"[Dataset_SWAT] Missing file: {test_csv}")

        train_df = pd.read_csv(train_csv)
        test_df  = pd.read_csv(test_csv)

        train_raw = train_df.values[:, :-1].astype(np.float32)   # [T_train, C]
        test_feat = test_df.values[:, :-1].astype(np.float32)    # [T_test,  C]
        test_lbl_raw = test_df.values[:, -1:].astype(np.float32) # [T_test, 1]

        train_raw = np.nan_to_num(train_raw)
        test_feat = np.nan_to_num(test_feat)
        test_lbl_raw = np.nan_to_num(test_lbl_raw)

        train_std = self.scaler.fit_transform(train_raw)
        test_std  = self.scaler.transform(test_feat)

        if flag == "TRAIN":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        elif flag == "VALI":
            X_win = _sliding_windows(train_std, self.win_size, self.step)
            Y_win = np.zeros((X_win.shape[0], self.win_size), dtype=np.float32)

        else:  # TEST
            X_win = _sliding_windows(test_std, self.win_size, self.step)

            lbl = test_lbl_raw
            if lbl.ndim == 2 and lbl.shape[1] == 1:
                lbl = lbl[:, 0]
            elif lbl.ndim == 2 and lbl.shape[1] > 1:
                lbl = (lbl != 0).any(axis=1).astype(np.float32)
            lbl = lbl.astype(np.float32)

            starts = np.arange(0, len(lbl) - self.win_size + 1, self.step)
            if len(starts) == 0:
                Y_win = np.zeros((1, self.win_size), dtype=np.float32)
                Y_win[0, :min(self.win_size, len(lbl))] = lbl[:min(self.win_size, len(lbl))]
            else:
                Y_win = np.stack(
                    [lbl[s:s + self.win_size] for s in starts],
                    axis=0
                ).astype(np.float32)

        # if test_lbl_raw.ndim == 2 and test_lbl_raw.shape[1] == 1:
        #     lbl = test_lbl_raw[:, 0]
        # elif test_lbl_raw.ndim == 2 and test_lbl_raw.shape[1] > 1:
        #     lbl = (test_lbl_raw != 0).any(axis=1).astype(np.float32)
        # else:
        #     lbl = test_lbl_raw.astype(np.float32)
        # lbl = lbl.astype(np.float32)

        
        X_np = np.transpose(X_win, (0, 2, 1)).astype(np.float32)   # [N, C, L]
        Y_np = Y_win[:, None, :].astype(np.float32)                # [N, 1, L]

        F_np = np.random.randn(X_np.shape[0], X_np.shape[1], 24).astype(np.float32)  # [N, C, 24]

        self.X = torch.nan_to_num(torch.from_numpy(X_np), nan=0.0)
        self.Y = torch.nan_to_num(torch.from_numpy(Y_np), nan=0.0)
        self.F = torch.nan_to_num(torch.from_numpy(F_np), nan=0.0)

        # mask: [N, L]
        self.mask = torch.ones((self.X.shape[0], self.X.shape[-1]), dtype=torch.bool, device=self.X.device)

        self.n_classes = 2
        self._parse_dimensions()

        print(f"[{self.__class__.__name__}] X={tuple(self.X.shape)} Y={tuple(self.Y.shape)} F={tuple(self.F.shape)}")
        return self.X, self.Y, self.F, self.mask


AVAILABLE_ANOMALY_DETECTION_DATASETS = {
    "MSL": Dataset_MSL,
    "Default": Dataset_Custom,
    "SMAP": Dataset_SMAP,
    "SMD": Dataset_SMD,
    "PSM": Dataset_PSM,
    "SWaT": Dataset_SWAT,
}

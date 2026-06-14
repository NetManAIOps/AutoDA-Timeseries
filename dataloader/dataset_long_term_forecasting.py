import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from utils.timefeatures import time_features
from utils import GlobalConfig

from .tsfeature_extractors import get_feature_extractor, FeatureExtractor
from .dataset_basic import BasicDataset
import torch
import torch.nn as nn

class Dataset_ETT_hour(BasicDataset):
    def __init__(self, config:GlobalConfig, flag='TRAIN'):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "ETTh1")
        self.features = getattr(config.args, "features", "S") # "S" | "M" | "MS"
        self.target = getattr(config.args, "target", "OT")
        self.scale = getattr(config.args, "scale", True)
        # embed='timeF' -> timeenc=1
        self.timeenc = 1 if getattr(config.args, "embed", "timeF") == "timeF" else 0
        self.freq = getattr(config.args, "freq", "h")

        self.seq_len = getattr(config.args, "seq_len", 24*4*4)
        self.label_len = getattr(config.args, "label_len", 24*4)
        self.pred_len = getattr(config.args, "pred_len", 24*4)

        super().__init__(config, flag)
        
    
    def _load_data(self, root_path, dataset_name_unused, flag):
        assert flag in ["TRAIN", "VALI", "TEST"]
        set_type = {"TRAIN": 0, "VALI": 1, "TEST": 2}[flag]
        
        year = 12*30*24
        four_moths = 4*30*24

        # train -> [0, year]; val -> [year, year+four months]; test -> [year+four months, year+2*four months]
        # start index of train, val, test
        border1s = [0, year - self.seq_len, year + four_moths - self.seq_len]
        # end index of train, val, test
        border2s = [year, year + four_moths, year + 2*four_moths]
        b1, b2 = border1s[set_type], border2s[set_type]

        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, self.data_path + ".csv"))
        # df_raw = df_raw.head(200)
        
        # MTS forecasting
        if self.features in ("M", "MS"):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data_all = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data_all = df_data.values
            self.scaler = None

        df_stamp = df_raw[['date']].iloc[b1:b2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            print(f"============No Time Series Features============")
            # feats = pd.DataFrame({
            #     'month': df_stamp.data.dt.month,
            #     'day': df_stamp.data.dt.day,
            #     'weekday': df_stamp.data.dt.weekday,
            #     'hour': df_stamp.data.dt.hour
            # })

            feats = pd.DataFrame({
                'month': df_stamp['date'].dt.month,
                'day': df_stamp['date'].dt.day,
                'weekday': df_stamp['date'].dt.weekday,
                'hour': df_stamp['date'].dt.hour
            })
            
            data_stamp = feats.values # [T, Denc] -> [T, 4]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 
        
        # Current split segment.
        seg = data_all[b1:b2] # [T, C]
        T, C = seg.shape
        stamps = data_stamp 


        Lx, Llabel, Ly = self.seq_len, self.label_len, self.pred_len
        N = T - Lx - Ly + 1
        Xs, Ys, Fenc, Fdec = [], [], [], []
        for i in range(N):
            s0, s1 = i, i+Lx
            r0, r1 = s1 - Llabel, s1 - Llabel + Llabel + Ly

            seq_x = seg[s0:s1, :] # [Lx, C]
            seq_y = seg[r0:r1, :] # [Llabel+Ly, C]
            x_mark = stamps[s0:s1, :] # [Lx, Denc]
            y_mark = stamps[r0:r1, :] # [Llabel+Ly, Ddec]
            
            Xs.append(seq_x)
            Ys.append(seq_y[-Ly:, :]) 
            Fenc.append(x_mark)
            Fdec.append(y_mark)

        X = torch.from_numpy(np.stack(Xs)).float().permute(0, 2, 1).contiguous() # [B, C, Lx]
        Y = torch.from_numpy(np.stack(Ys)).float().permute(0, 2, 1).contiguous() # [B, C, Ly]
        F = torch.from_numpy(np.stack(Fenc)).float() # [B, Lx, Denc]
        mask = torch.from_numpy(np.stack(Fdec)).float() # [B, Llabel+Ly, Ddec]


        # print(f"X.shape: {X.shape}")
        # print(f"F.shape: {F.shape}")

        
        # exit(0)

        self.n_channels = C
        self.seq_len = Lx
        # self.label_len = label_len
        self.pred_len = Ly


        # self.fe:FeatureExtractor = get_feature_extractor(self.config.args.feature_extractor)(self.config, None)
        # F = self.fe(X)

        # proj = nn.Linear(4, 24)
        # F_proj = proj(F)

        # F_proj = F_proj.permute(0, 2, 1) # [B, 22, 96]
        # F_new = nn.functional.interpolate(F_proj, size=C, mode="linear", align_corners=False)
        # F_new = F_new.permute(0, 2, 1) # [B, 7, 22]

        # F = F_new

        X = X.detach()
        Y = Y.detach()
        F = F.detach()
        mask = mask.detach()


        # print(f"X.shape: {X.shape}") # [8449, 7, 96]
        # print(f"F.shape: {F.shape}") # [8449, 96, 4]

        return X, Y, F, mask

    
    def inverse_transform(self, data_np):
        return data_np if self.scaler is None else self.scaler.inverse_transform(data_np)



    
        

    

class Dataset_ETT_minute(BasicDataset):
    def __init__(self, config:GlobalConfig, flag='TRAIN'):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "ETTm1")
        self.features = getattr(config.args, "features", "S") # "S" | "M" | "MS"
        self.target = getattr(config.args, "target", "OT")
        self.scale = getattr(config.args, "scale", True)
        # embed='timeF' -> timeenc=1
        self.timeenc = 1 if getattr(config.args, "embed", "timeF") == "timeF" else 0
        self.freq = getattr(config.args, "freq", "t")

        self.seq_len = getattr(config.args, "seq_len", 24*4*4)
        self.label_len = getattr(config.args, "label_len", 24*4)
        self.pred_len = getattr(config.args, "pred_len", 24*4)

        super().__init__(config, flag)
        
    
    def _load_data(self, root_path, dataset_name_unused, flag):
        assert flag in ["TRAIN", "VALI", "TEST"]
        set_type = {"TRAIN": 0, "VALI": 1, "TEST": 2}[flag]
        
        year = 12*30*24*4
        four_moths = 4*30*24*4

        # train -> [0, year]; val -> [year, year+four months]; test -> [year+four months, year+2*four months]
        # start index of train, val, test
        border1s = [0, year - self.seq_len, year + four_moths - self.seq_len]
        # end index of train, val, test
        border2s = [year, year + four_moths, year + 2*four_moths]
        b1, b2 = border1s[set_type], border2s[set_type]

        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, self.data_path + ".csv"))
        # df_raw = df_raw.head(200)

        # MTS forecasting
        if self.features in ("M", "MS"):
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data_all = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data_all = df_data.values
            self.scaler = None

        df_stamp = df_raw[['date']].iloc[b1:b2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            # feats = pd.DataFrame({
            #     'month': df_stamp.data.dt.month,
            #     'day': df_stamp.data.dt.day,
            #     'weekday': df_stamp.data.dt.weekday,
            #     'hour': df_stamp.data.dt.hour,
            #     'minute': df_stamp.data.dt.minute
            # })

            feats = pd.DataFrame({
                'month': df_stamp['date'].dt.month,
                'day': df_stamp['date'].dt.day,
                'weekday': df_stamp['date'].dt.weekday,
                'minute': df_stamp['date'].dt.minute
            })

            feats['minute'] = feats['minute'] // 15 
            
            data_stamp = feats.values # [T, Denc] -> [T, 4]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 
        
        seg = data_all[b1:b2] # [T, C]
        T, C = seg.shape
        stamps = data_stamp 


        Lx, Llabel, Ly = self.seq_len, self.label_len, self.pred_len
        N = T - Lx - Ly + 1
        Xs, Ys, Fenc, Fdec = [], [], [], []
        for i in range(N):
            s0, s1 = i, i+Lx
            r0, r1 = s1 - Llabel, s1 - Llabel + Llabel + Ly

            seq_x = seg[s0:s1, :] # [Lx, C]
            seq_y = seg[r0:r1, :] # [Llabel+Ly, C]
            x_mark = stamps[s0:s1, :] # [Lx, Denc]
            y_mark = stamps[r0:r1, :] # [Llabel+Ly, Ddec]
            
            Xs.append(seq_x)
            Ys.append(seq_y[-Ly:, :])
            Fenc.append(x_mark)
            Fdec.append(y_mark)

        X = torch.from_numpy(np.stack(Xs)).float().permute(0, 2, 1).contiguous() # [B, C, Lx]
        Y = torch.from_numpy(np.stack(Ys)).float().permute(0, 2, 1).contiguous() # [B, C, Ly]
        F = torch.from_numpy(np.stack(Fenc)).float() # [B, Lx, Denc]
        mask = torch.from_numpy(np.stack(Fdec)).float() # [B, Llabel+Ly, Ddec]


        self.n_channels = C
        self.seq_len = Lx
        # self.label_len = label_len
        self.pred_len = Ly


        X = X.detach()
        Y = Y.detach()
        F = F.detach()
        mask = mask.detach()


        # print(f"X.shape: {X.shape}") # [8449, 7, 96]
        # print(f"F.shape: {F.shape}") # [8449, 96, 4]

        return X, Y, F, mask

    
    def inverse_transform(self, data_np):
        return data_np if self.scaler is None else self.scaler.inverse_transform(data_np)
    




class Dataset_Custom(BasicDataset):
    def __init__(self, config:GlobalConfig, flag='TRAIN'):
        self.root_path = config.args.dataset_root
        self.data_path = getattr(config.args, "dataset", "ETTh1")
        self.features = getattr(config.args, "features", "M") # "S" | "M" | "MS"
        self.target = getattr(config.args, "target", "OT")
        self.scale = getattr(config.args, "scale", True)
        # embed='timeF' -> timeenc=1
        self.timeenc = 1 if getattr(config.args, "embed", "timeF") == "timeF" else 0
        self.freq = getattr(config.args, "freq", "h")

        self.seq_len = getattr(config.args, "seq_len", 24*4*4)
        self.label_len = getattr(config.args, "label_len", 24*4)
        self.pred_len = getattr(config.args, "pred_len", 24*4)

        super().__init__(config, flag)
        
    
    def _load_data(self, root_path, dataset_name_unused, flag):
        assert flag in ["TRAIN", "VALI", "TEST"]
        set_type = {"TRAIN": 0, "VALI": 1, "TEST": 2}[flag]
        

        # df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path, self.data_path + ".csv"))
        # df_raw = df_raw.head(200)

        n = len(df_raw)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        
        # train -> [0, year]; val -> [year, year+four months]; test -> [year+four months, year+2*four months]
        # start index of train, val, test
        border1s = [0, max(0, num_train - self.seq_len), max(0, n - num_test - self.seq_len)]
        # end index of train, val, test
        border2s = [num_train, num_train + num_vali, n]
        b1, b2 = border1s[set_type], border2s[set_type]
        
        # MTS forecasting
        if self.features in ("M", "MS"):
            cols_data = df_raw.columns[1:] 
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            scaler = StandardScaler()
            scaler.fit(df_data.iloc[border1s[0]:border2s[0]].values)
            data_all = scaler.transform(df_data.values)
            self.scaler = scaler
        else:
            data_all = df_data.values
            self.scaler = None

        df_stamp = df_raw[['date']].iloc[b1:b2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            # feats = pd.DataFrame({
            #     'month': df_stamp.data.dt.month,
            #     'day': df_stamp.data.dt.day,
            #     'weekday': df_stamp.data.dt.weekday,
            #     'hour': df_stamp.data.dt.hour,
            #     # 'minute': df_stamp.data.dt.minute
            # })
            feats = pd.DataFrame({
                'month': df_stamp['date'].dt.month,
                'day': df_stamp['date'].dt.day,
                'weekday': df_stamp['date'].dt.weekday,
                'hour': df_stamp['date'].dt.hour
            })
            
            data_stamp = feats.values # [T, Denc] -> [T, 4]
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 
        
        seg = data_all[b1:b2] # [T, C]
        T, C = seg.shape
        stamps = data_stamp


        Lx, Llabel, Ly = self.seq_len, self.label_len, self.pred_len
        N = T - Lx - Ly + 1
        Xs, Ys, Fenc, Fdec = [], [], [], []
        for i in range(N):
            s0, s1 = i, i+Lx
            r0, r1 = s1 - Llabel, s1 - Llabel + Llabel + Ly

            seq_x = seg[s0:s1, :] # [Lx, C]
            seq_y = seg[r0:r1, :] # [Llabel+Ly, C]
            x_mark = stamps[s0:s1, :] # [Lx, Denc]
            y_mark = stamps[r0:r1, :] # [Llabel+Ly, Ddec]
            
            Xs.append(seq_x)
            Ys.append(seq_y[-Ly:, :])
            Fenc.append(x_mark)
            Fdec.append(y_mark)

        X = torch.from_numpy(np.stack(Xs)).float().permute(0, 2, 1).contiguous() # [B, C, Lx]
        Y = torch.from_numpy(np.stack(Ys)).float().permute(0, 2, 1).contiguous() # [B, C, Ly]
        F = torch.from_numpy(np.stack(Fenc)).float() # [B, Lx, Denc]
        mask = torch.from_numpy(np.stack(Fdec)).float() # [B, Llabel+Ly, Ddec]


        self.n_channels = C
        self.seq_len = Lx
        # self.label_len = label_len
        self.pred_len = Ly

        # F = F_new

        X = X.detach()
        Y = Y.detach()
        F = F.detach()
        mask = mask.detach()


        # print(f"X.shape: {X.shape}") # [8449, 7, 96]
        # print(f"F.shape: {F.shape}") # [8449, 96, 4]

        return X, Y, F, mask

    
    def inverse_transform(self, data_np):
        return data_np if self.scaler is None else self.scaler.inverse_transform(data_np)





AVAILABLE_LONG_TERM_FORECASTING_DATASETS = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "Default": Dataset_Custom
}

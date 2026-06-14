from dataloader import get_dataset
from utils import GlobalConfig
from .exp_basic import ExpBasic
import torch.nn as nn
import os
from utils.tools import EarlyStopping, cal_accuracy
import time
import numpy as np
from torch import optim
import torch
from autoaugment import AutoAugmentBasic
import torch
from typing import Optional, Tuple
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import os
import matplotlib.pyplot as plt



def masked_mse(x_hat: torch.Tensor, x: torch.Tensor, 
               mask: Optional[torch.Tensor] = None,
               reduction: str = "mean") -> torch.Tensor:
    """
    x_hat: [B, C, L]
    """
    diff2 = (x_hat - x) ** 2
    if mask is not None:
        m = mask.unsqueeze(1).to(diff2.dtype) # [B, 1, L]
        diff2 = diff2 * m
        denom = m.sum() if reduction == "mean" else None
    
    if reduction == "mean":
        val = diff2.sum()
        if mask is not None:
            val = val / denom.clamp_min(1.0)
        else:
            val = val / diff2.numel()
        return val
    elif reduction == "none":
        return diff2
    else:
        return diff2.mean()

def pointwise_error(x_hat: torch.Tensor,
                    x: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> np.ndarray:
    """
    x_hat, x: [B, C, L], mask: [B, L]
    """
    device = x_hat.device
    x = x.to(device)
    err = (x_hat - x) ** 2 # [B, C, L]
    if mask is not None:
        mask = mask.to(device)
        err = err * mask.unsqueeze(1).to(err.dtype)
    err = err.mean(dim=1) # [B, L]
    return err.detach().cpu().numpy()


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred

class ExpAnomalyDetection(ExpBasic):


    def _build_model(self):
        train_data, _ = self._get_data(flag='TRAIN', load_as='TRAIN')
        test_data, _ = self._get_data(flag='TEST', load_as='TEST')

        self.config.dimensions.update(
            n_channels=train_data.n_channels,
            seq_len=max(train_data.win_size, test_data.win_size),
            n_features=train_data.n_features,
            pred_len=max(train_data.win_size, test_data.win_size)
        )

        self.config.args.n_channels = train_data.n_channels
        self.config.args.seq_len = max(train_data.win_size, test_data.win_size)
        self.config.args.n_features = train_data.n_features
        self.config.args.pred_len = max(train_data.win_size, test_data.win_size)

        model: AutoAugmentBasic = self.model_class(self.config).float().to(self.device)
        return model.to(self.device)
    
    def _get_data(self, flag: str, load_as: str):
        if load_as not in self.loaded_data:
            dataset, data_loader = get_dataset(self.config, flag)
            self.loaded_data[load_as] = (dataset, data_loader)
        else:
            dataset, data_loader = self.loaded_data[load_as]
        return dataset, data_loader
    
    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.config.args.learning_rate)


    def _select_criterion(self):
        return self.model.get_criterion(default_criterion=nn.MSELoss(reduction="mean"))
    

    def vali(self, vali_data, vali_loader, criterion=None):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_f, batch_masks in vali_loader:
                batch_x = batch_x.float().to(self.device) # [B, C, L]
                batch_f = batch_f.float().to(self.device) # [B, C, Df]
                batch_y = batch_y.float().to(self.device)
                batch_masks = batch_masks.to(self.device) # [B, L]

                output_y, _, _ = self.model(batch_x, batch_y, batch_f, batch_masks)
                loss = masked_mse(output_y, batch_x, batch_masks, reduction="mean")
                total_loss.append(loss.item())
        
        self.model.train()
        return float(np.mean(total_loss)) if total_loss else 0.0
    
    

    def train(self):
        train_data, train_loader = self._get_data(flag='TRAIN', load_as='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')
        vali_data, vali_loader = self._get_data(flag='VALI', load_as='VALI')

        checkpoint_path = self.config.get_checkpoint_path()
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.config.args.train_epochs):
            print("[Model State]", end="", flush=True)
            print(self.model.summarize_state(), flush=True)

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_f, batch_masks) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_f = batch_f.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_masks = batch_masks.to(self.device)

                output_y, aug_y, aug_mask = self.model(batch_x, batch_y, batch_f, batch_masks)
                loss = masked_mse(output_y, batch_x, batch_masks, reduction="mean")
                train_loss.append(loss.item())

                if (i + 1) % 5 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()), flush=True)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time), flush=True)
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {:.1f}s".format(epoch + 1, time.time() - epoch_time), flush=True)
            train_loss = float(np.mean(train_loss)) if train_loss else 0.0
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} "
                  f"Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.5f} "
                  f"Vali Loss: {vali_loss:.5f} "
                  f"Test Loss: {test_loss:.5f}", flush=True)
            
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early Stopping", flush=True)
                break


        # self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=self.device))
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=self.device))

        return self.model

    def test(self, load_checkpoint: bool = False):
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')
        train_data, train_loader = self._get_data(flag='TRAIN', load_as='TRAIN')

        if load_checkpoint:
            checkpoint_path = self.config.get_checkpoint_path()

            print(f"try load model checkpoint from {checkpoint_path}", flush=True)
            self.model.load_state_dict(torch.load(checkpoint_path,
                                                  weights_only=True,
                                                  map_location=self.device))
            
        folder_path = os.path.join("./test_results", self.config.get_keyword())
        os.makedirs(folder_path, exist_ok=True)
        
        self.model.eval()

        train_err = []
        with torch.no_grad():
            for batch_x, batch_y, batch_f, batch_masks in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_f = batch_f.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_masks = batch_masks.to(self.device)

                output_y, _, _ = self.model(batch_x, batch_y, batch_f, batch_masks)
                err = pointwise_error(output_y, batch_x, batch_masks) # [B, L]
                train_err.append(err)
        train_err = np.concatenate(train_err, axis=0).reshape(-1)


        test_err = []
        test_labels = []

        all_batch_y = []
        all_output_y = []

        with torch.no_grad():
            for batch_x, batch_y, batch_f, batch_masks in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_f = batch_f.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_masks = batch_masks.to(self.device)

                output_y, _, _ = self.model(batch_x, batch_y, batch_f, batch_masks)
                err = pointwise_error(output_y, batch_y, batch_masks) # [B, L]
                test_err.append(err)

                all_batch_y.append(batch_y.cpu().numpy())     
                all_output_y.append(output_y.cpu().numpy())   

                # batch_y: [B, 1, L] -> [B*L, ]
                test_labels.append(batch_y.view(batch_y.shape[0], -1).cpu().numpy())
        
        test_err = np.concatenate(test_err, axis=0).reshape(-1)
        gt = np.concatenate(test_labels, axis=0).reshape(-1).astype(int)
    
        anomaly_ratio = float(getattr(self.config.args, "anomaly_ratio", 5.0))
        combined = np.concatenate([train_err, test_err], axis=0)
        threshold = np.percentile(combined, 100-anomaly_ratio)
        print("Threshold: ", threshold, flush=True)

        pred = (test_err > threshold).astype(int)

        print("pred: ", pred.shape, flush=True)
        print("gt: ", gt.shape, flush=True)

        gt_adj, pred_adj = adjustment(gt, pred)



        all_batch_y = np.concatenate(all_batch_y, axis=0)       # [N, 1, L]
        all_output_y = np.concatenate(all_output_y, axis=0)     # [N, C, L]
        save_path = os.path.join(folder_path, "test_outputs.npz")
        # np.savez_compressed(
        #     save_path,
        #     batch_y=all_batch_y,      
        #     output_y=all_output_y,    # model output
        #     gt_adj=gt_adj,            
        #     pred_adj=pred_adj,        
        #     threshold=threshold       
        # )
        # print(f"Saved test results to {save_path}")




        accuracy = accuracy_score(gt_adj, pred_adj)
        precision, recall, f1, support = precision_recall_fscore_support(gt_adj, pred_adj, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f1), flush=True)
        
        # test_result_path = os.path.join(folder_path, "test_result.txt")
        test_result_path = self.config.get_test_result_path()
        if not os.path.exists(os.path.dirname(test_result_path)):
            os.makedirs(os.path.dirname(test_result_path))
        
        with open(test_result_path, 'a') as f:
            f.write(self.config.get_keyword() + "   \n")
            f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} \n\n".format(
                accuracy, precision, recall, f1))
        
        return



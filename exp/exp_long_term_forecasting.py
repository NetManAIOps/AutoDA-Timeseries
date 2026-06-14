from dataloader import get_dataset
from utils import GlobalConfig
from .exp_basic import ExpBasic
import torch.nn as nn
import os
from utils.tools import EarlyStopping
import time
import numpy as np
from torch import optim
import torch
from autoaugment import AutoAugmentBasic

# from dataloader.dataset import get_dataset_forecasting
from utils.metrics import metric

from torch.utils.data import DataLoader
from torch.utils.data import Subset

import os, matplotlib.pyplot as plt
from matplotlib import font_manager




class ExpLongTermForecasting(ExpBasic):

    def _build_model(self) -> AutoAugmentBasic:
        train_data, _ = self._get_data(flag='TRAIN', load_as='TRAIN')
        test_data, _ = self._get_data(flag='TEST', load_as='TEST')
        # vali_data, _ = self._get_data(flag='VALI', load_as='VALI')

        self.config.dimensions.update(
            n_channels=train_data.n_channels,
            seq_len=max(train_data.seq_len, test_data.seq_len),
            n_features=train_data.n_features, # = D_enc
            pred_len=train_data.pred_len
        )
        self.config.args.n_channels =train_data.n_channels
        self.config.args.seq_len = max(train_data.seq_len, test_data.seq_len)
        self.config.args.n_features = train_data.n_features
        self.config.args.pred_len = train_data.pred_len

        # model init
        model:AutoAugmentBasic = self.model_class(self.config).float().to(self.device)
        return model

    def _get_data(self, flag: str, load_as: str):
        if load_as not in self.loaded_data:
            dataset, data_loader = get_dataset(self.config, flag)
            self.loaded_data[load_as] = (dataset, data_loader)
        else:
            dataset, data_loader = self.loaded_data[load_as]        
        
        return dataset, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.config.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.config.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        # criterion = nn.MSELoss()
        return self.model.get_criterion(default_criterion=nn.MSELoss())

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, :, -self.config.args.pred_len:], device=self.device).float()
                dec_inp = torch.cat([batch_x[:, :, -self.config.args.label_len:], dec_inp], dim=2) # [B, C, LLabel_len + Ly]

                # outputs, aug_y, aug_mask = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, aug_y, aug_mask = self.model(batch_x, dec_inp, batch_x_mark, batch_y_mark)
                f_dim = -1 if self.config.args.features == 'MS' else 0
                # outputs = outputs[:, -self.config.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.config.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs[:, f_dim:, -self.config.args.pred_len:]
                batch_y = batch_y[:, f_dim:, -self.config.args.pred_len:].to(self.device)

                # print(f"vali_outputs.shape: {outputs.shape}")
                # print(f"vali_batch_y.shape: {batch_y.shape}")
                # exit(0)
                
                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self):
        train_data, train_loader = self._get_data('TRAIN', 'TRAIN')
        vali_data, vali_loader = self._get_data('VALI', 'VALI')
        test_data, test_loader = self._get_data('TEST', 'TEST')

        checkpoint_path = self.config.get_checkpoint_path()
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # if self.config.args.use_amp:
        #     scaler = torch.cuda.amp.grad_scaler()

        for epoch in range(self.config.args.train_epochs):
            print("[Model State]", end="")
            print(self.model.summarize_state())
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                
                # print(f"batch_x.shape: {batch_x.shape}")
                # print(f"batch_y.shape: {batch_y.shape}")
                # print(f"batch_x_mark: {batch_x_mark.shape}")
                # print(f"batch_y_mark: {batch_y_mark.shape}")

                # print("=============================")
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device) # [B, C, Lx]
                batch_y = batch_y.float().to(self.device) # [B, C, Ly]
                batch_x_mark = batch_x_mark.float().to(self.device) # [B, Lx, Denc]
                batch_y_mark = batch_y_mark.float().to(self.device) # [B, Llabel+Ly, Ddec]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, :, -self.config.args.pred_len:], device=self.device).float()
                dec_inp = torch.cat([batch_x[:, :, -self.config.args.label_len:], dec_inp], dim=2) # [B, C, LLabel_len + Ly] L


           
                # batch_y_mark: [Batch_size, Llabel+Ly, Ddec]
                outputs, _, _ = self.model(batch_x, dec_inp, batch_x_mark, batch_y_mark)
                f_dim = -1 if self.config.args.features == 'MS' else 0

                outputs = outputs[:, f_dim:, -self.config.args.pred_len:]
                batch_y = batch_y[:, f_dim:, -self.config.args.pred_len:].to(self.device)

                # print(f"outputs.shape: {outputs.shape}")
                # print(f"batch_y.shape: {batch_y.shape}")

                # print(f"train_outputs.shape: {outputs.shape}")
                # print(f"train_batch_y.shape: {batch_y.shape}")
                # exit(0)


                if self.config.args.tsa == "AutoTSA_bypass_attention_1":
                    # Bypass Augmentation
                    loss = self.model.compute_total_loss(
                        outputs,
                        batch_y.float().squeeze(-1)
                    )
                else:
                    # Other AutoDA
                    loss = criterion(outputs, batch_y)



                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location=self.device))
        return self.model

    



    def test(self, load_checkpoint:bool=False):

        

        test_data, test_loader = self._get_data('TEST', 'TEST')
        if load_checkpoint:
            checkpoint_path = self.config.get_checkpoint_path()
            
            print(f"try load model checkpoint from {checkpoint_path}")
            self.model.load_state_dict(torch.load(checkpoint_path,
                                                  weights_only=False,
                                                  map_location=self.device))
            # msg = load_ckpt_with_tau(self.model, checkpoint_path, self.device,
            #              cfg_tau=float(self.config.args.tau),
            #              learnable=bool(getattr(self.config.args, "l_tau", 0)))
            # print(msg)

        preds = []
        trues = []

        


        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, :, -self.config.args.pred_len:], device=self.device).float()
                dec_inp = torch.cat([batch_x[:, :, -self.config.args.label_len:], dec_inp], dim=2) # [B, C, LLabel_len + Ly] Ly

                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, aug_y, aug_mask = self.model(batch_x, dec_inp, batch_x_mark, batch_y_mark)

                f_dim = -1 if self.config.args.features == 'MS' else 0
                # outputs = outputs[:, -self.config.args.pred_len:, :]
                # batch_y = batch_y[:, -self.config.args.pred_len:, :].to(self.device)

                outputs = outputs[:, :, -self.config.args.pred_len:]
                batch_y = batch_y[:, :, -self.config.args.pred_len:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()


                # print(f"outputs.shape: {outputs.shape}")
                # print(f"batch_y.shape: {batch_y.shape}")
                # exit(0)

                if test_data.scale and self.config.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], - 1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                
                # outputs = outputs[:, :, f_dim:]
                # batch_y = batch_y[:, :, f_dim:]

                outputs = outputs[:, f_dim:, :]
                batch_y = batch_y[:, f_dim:, :]

                pred = outputs # [B, n_channels, pred_len]
                true = batch_y # [B, n_channels, pred_len]


                preds.append(pred)
                trues.append(true)
        
        preds = np.concatenate(preds, axis=0) # [B, C, L]
        trues = np.concatenate(trues, axis=0) # [B, C, L]
        print("test shape:", preds.shape, trues.shape)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        preds_eval = np.transpose(preds, (0, 2, 1)) # [B, L, C]
        trues_eval = np.transpose(trues, (0, 2, 1)) # [B, L, C]

        print("test shape", preds.shape, trues.shape)

        test_result_path = self.config.get_test_result_path()
        if not os.path.exists(os.path.dirname(test_result_path)):
            os.makedirs(os.path.dirname(test_result_path))
        

        mae, mse, rmse, mape, mspe = metric(preds_eval, trues_eval)
        print('mse: {}, mae: {}'.format(mse, mae))
        with open(test_result_path, 'a') as f:
            f.write(f"{self.config.get_keyword()} \n MSE: {mse}, MAE: {mae}\n\n")

        return
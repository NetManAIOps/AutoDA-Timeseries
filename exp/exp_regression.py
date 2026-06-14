from torchmetrics.functional import accuracy

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
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ExpRegression(ExpBasic):

    def _build_model(self) -> AutoAugmentBasic:
        train_data, train_loader = self._get_data(flag='TRAIN', load_as='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')
        self.config.dimensions.update(
            n_channels=train_data.n_channels,
            seq_len=max(train_data.seq_len, test_data.seq_len),
            n_features=train_data.n_features,
            pred_len= train_data.pred_len
        )
        self.config.args.n_channels =train_data.n_channels
        self.config.args.seq_len = max(train_data.seq_len, test_data.seq_len)
        self.config.args.n_features = train_data.n_features
        self.config.args.pred_len = train_data.pred_len
        # model init
        model:AutoAugmentBasic  = self.model_class(self.config).float().to(self.device)
        return model.to(self.device)

    def _get_data(self, flag: str, load_as:str):
        if load_as not in self.loaded_data:
            dataset, data_loader = get_dataset(self.config, flag)
            self.loaded_data[load_as]=(dataset, data_loader)
        else:
            dataset, data_loader = self.loaded_data[load_as]
        return dataset, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.config.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        return self.model.get_criterion(default_criterion=nn.MSELoss())

    def train(self):
        train_data, train_loader = self._get_data(flag='TRAIN', load_as='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')
        vali_data, vali_loader = self._get_data(flag='TEST', load_as='TEST')

        checkpoint_path = self.config.get_checkpoint_path()
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        for epoch in range(self.config.args.train_epochs):
            print("[Model State]", end="")
            print(self.model.summarize_state())
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_f, batch_masks) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)
                batch_f = batch_f.float().to(self.device)
                batch_masks = batch_masks.to(self.device)
                output_y, aug_y, aug_mask = self.model(batch_x, batch_y, batch_f, batch_masks)
                loss = criterion(output_y.squeeze(-1), batch_y.squeeze(-1).float())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.config.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} "
                  f"Steps: {train_steps} | "
                  f"Train Loss: {train_loss:.3f} "
                  f"Vali Loss: {vali_loss:.3f} "
                  f"Vali Acc: {val_accuracy:.3f} "
                  f"Test Loss: {test_loss:.3f} "
                  f"Test Acc: {test_accuracy:.3f}")
            early_stopping(-val_accuracy, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break


        self.model.load_state_dict(torch.load(checkpoint_path,
                                              weights_only=True,
                                              map_location=self.device))
        return self.model

    def vali(self, vali_data, vali_loader, criterion=None):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_f, batch_masks) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)
                batch_f = batch_f.to(self.device)
                batch_masks = batch_masks.to(self.device)

                output_y, aug_y, aug_mask = self.model(batch_x, batch_y, batch_f, batch_masks)

                pred = output_y.detach()
                if criterion:
                    loss = criterion(pred.squeeze(-1), batch_y.squeeze(-1)).detach().cpu()
                    total_loss.append(loss)

                preds.append(output_y.detach())
                trues.append(batch_y)
        if criterion:
            total_loss = torch.mean(torch.tensor(total_loss))

        preds = torch.cat(preds, 0).view(-1).cpu().numpy()
        trues = torch.cat(trues, 0).view(-1).cpu().numpy()

        accuracy = -mean_squared_error(trues, preds)

        self.model.train()
        if criterion:
            return total_loss, accuracy
        else:
            return accuracy

    def test(self, load_checkpoint:bool=False):
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')
        if load_checkpoint:
            checkpoint_path = self.config.get_checkpoint_path()
            print(f'try load model checkpoint from {checkpoint_path}')

            self.model.load_state_dict(torch.load(checkpoint_path,
                                                  weights_only=True,
                                                  map_location=self.device))


        accuracy = self.vali(test_data, test_loader, None)

        test_result_path = self.config.get_test_result_path()
        if not os.path.exists(os.path.dirname(test_result_path)):
            os.makedirs(os.path.dirname(test_result_path))

        print('accuracy:{}'.format(accuracy))
        with open(test_result_path, 'a') as f:
            f.write(f"{self.config.get_keyword()}  \naccuracy:{accuracy}\n\n")
        return

    def plot(self, load_checkpoint:bool=True):
        test_data, test_loader = self._get_data(flag='TEST', load_as='TEST')

        if load_checkpoint:
            checkpoint_path = self.config.get_checkpoint_path()
            print(f'try load model checkpoint from {checkpoint_path}')
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            self.model = self.model.float().to(self.device)

        from torchviz import make_dot
        crt = self.model.get_criterion(nn.CrossEntropyLoss())



        for i, (batch_x, batch_y, batch_f, batch_masks) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.to(self.device)
            batch_f = batch_f.to(self.device)
            batch_masks = batch_masks.to(self.device)

            output_y, aug_y, aug_mask = self.model(batch_x, batch_y, batch_f, batch_masks)
            #dot = make_dot(output_y, params=dict(self.model.named_parameters()))
            #dot.render(f"model_graph", format="png")
            print(crt.prev_probs)
            return
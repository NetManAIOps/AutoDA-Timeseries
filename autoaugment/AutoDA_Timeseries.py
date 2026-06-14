import torch.nn as nn

from .augments import AVAILABLE_TRANSFORMS, AugmentTransform
from typing import List, Type
import torch
import torch.nn.functional as F
from .AutoAugmentBasic import AutoAugmentBasic
from utils import GlobalConfig
from downstream import build_downstream_model
from typing import List


def _param_value(v):
    if isinstance(v, nn.Parameter):
        return v.item()
    elif isinstance(v, torch.Tensor):
        return v.item()
    else:
        return v

def _parse_config(keywords: List[str], default):
    global_config = GlobalConfig.get_config()
    arg_dict = global_config.tsa_args
    for k in keywords:
        if k in arg_dict:
            return arg_dict[k]
    return default

class CompositeLoss(nn.Module):

    def __init__(self, task_criterion, layer_cnt, learnable_weight=False,
                 task_weight=1.0, entropy_weight=0.1, diversity_weight=0.1):
        super().__init__()
        self.task_criterion = task_criterion
        self.learnable_weight = learnable_weight

        global_config = GlobalConfig.get_config()
        device = global_config.device

        self.task_weight = torch.ones(1, device=device) * task_weight
        self.entropy_weight = torch.ones(1, device=device) * entropy_weight
        self.diversity_weight = torch.ones(1, device=device) * diversity_weight


        if self.learnable_weight:
            print(f"[Composite Loss] Dynamic weights")
            self.task_weight = nn.Parameter(self.task_weight)
            self.entropy_weight = nn.Parameter(self.entropy_weight)
            self.diversity_weight = nn.Parameter(self.diversity_weight)
        else:
            print(f"[Composite Loss] Fixed weights")
        self.layer_cnt = layer_cnt
        # element: (batch_size, n_transform)
        self.prev_probs: [None|torch.Tensor] = [None] * layer_cnt
        # element: (batch_size, n_transform)
        self.current_probs: [None|torch.Tensor] = [None] * layer_cnt
        self._last_loss = None


    def _compute_augment_loss(self, task_loss):
        entropy_loss = torch.zeros_like(task_loss, device=task_loss.device)
        diversity_loss = torch.zeros_like(task_loss, device=task_loss.device)
        for layer_id in range(self.layer_cnt):

            # self.current_probs[layer_id]: (batch_size, n_transform)
            current_probs: torch.Tensor = self.current_probs[layer_id]
            if current_probs is None:
                continue
            # entropy_loss: scalar
            entropy_loss += -torch.sum(current_probs * torch.log(current_probs + 1e-10), dim=1).mean()

            # self.prev_probs[layer_id]: (batch_size, n_transform)
            prev_probs: torch.Tensor = self.prev_probs[layer_id]
            if prev_probs is None:
                continue

            min_batch_size = min(prev_probs.shape[0], current_probs.shape[0])
            prev_probs = prev_probs[:min_batch_size]
            current_probs = current_probs[:min_batch_size]

            # diversity_loss: scalar
            diversity_loss += F.kl_div(
                current_probs.log(),
                prev_probs,
                reduction='batchmean'
            )
        # Update previous probabilities
        for layer_id in range(self.layer_cnt):
            if self.current_probs[layer_id] is not None:
                # self.prev_probs[layer_id]: (batch_size, n_transform)
                self.prev_probs[layer_id] = self.current_probs[layer_id].detach()
            self.current_probs[layer_id] = None

        return entropy_loss, diversity_loss

    def _combine_all_loss(self, task_loss, entropy_loss, diversity_loss):

        def get_loss(loss, weight):
            if not self.learnable_weight:
                return loss * weight
            else:
                if weight==0:
                    return 0
                c_tau = weight * weight

                return loss / c_tau + torch.log(c_tau+1.)

        final_loss = 0
        final_loss = final_loss + get_loss(task_loss, self.task_weight)
        assert not torch.isnan(final_loss).any()
        final_loss = final_loss + get_loss(entropy_loss, self.entropy_weight)
        assert not torch.isnan(final_loss).any()
        final_loss = final_loss + get_loss(diversity_loss, self.diversity_weight)
        assert not torch.isnan(final_loss).any()
        return final_loss

    def forward(self, output, label):
        """
        :param output: (batch_size, pred_len)
        :param label: (batch_size,)
        :return: loss: scalar
        """
        # task_loss: scalar
        task_loss = self.task_criterion(output, label)

        # entropy_loss, diversity_loss: scalar
        entropy_loss, diversity_loss = self._compute_augment_loss(task_loss)

        # total_loss: scalar
        total_loss = self._combine_all_loss(task_loss, entropy_loss, diversity_loss)

        return total_loss

    def record_prob(self, prob, layer_id):
        """
        :param prob: (batch_size, n_transform)
        :param layer_id: int
        :return: None
        """
        # self.prev_probs[layer_id]: (batch_size, n_transform)
        self.current_probs[layer_id] = prob

    def get_state(self) -> str:
        msg = ""
        msg += "\nCompositeLoss ({}) weight:{}".format(
            "dynamic" if self.learnable_weight else "fixed",
            [
            _param_value(self.task_weight),
            _param_value(self.entropy_weight),
            _param_value(self.diversity_weight)
        ])
        if self._last_loss:
            msg+= f"\nLast Loss={[_param_value(v) for v in self._last_loss]}"
        return msg

class MLPModule(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x.float())

class AugProbMLPModule(nn.Module):
    def __init__(self, global_config:GlobalConfig, n_features, n_channels, n_transforms, hidden_dim=64):
        super().__init__()
        self.mlp = MLPModule(n_features*n_channels+n_transforms, n_transforms, hidden_dim)
        self.global_config = global_config
        self.device=global_config.device
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)

        self.min_prob = float(_parse_config(["min_prob","minp"], 0))
        self.raw_op_bonus = torch.zeros((1,n_transforms), device=self.device)
        self.raw_op_bonus[:,0] += float(_parse_config(["raw_op_bonus","rob"], 1))

    def forward(self, batch_f, prev_prob):
        """
        :param batch_f: (batch, n_channels, n_features)
        :param prev_prob: (batch, n_transforms)
        :return: (batch, n_transforms)
        """
        batch, n_channels, n_features = batch_f.shape
        batch_f = batch_f.view(batch, -1)
        inputs = torch.cat([batch_f, prev_prob], dim=1)

        outputs = self.softmax1(self.mlp(inputs))
        # outputs = F.gumbel_softmax(logits=self.mlp(inputs), tau=1.0, hard=True)
        outputs = F.relu(outputs)
        outputs = torch.clamp(outputs, min=self.min_prob)
        outputs = self.softmax2(outputs + self.raw_op_bonus)
        # outputs = F.gumbel_softmax(logits=self.mlp(inputs), tau=1.0, hard=True)
        # print(f"outputs: {outputs.shape}")
        # print(f"prev_prob: {prev_prob.shape}")
        return outputs

class AugStrengthMLPModule(nn.Module):
    """
    Strength Module that allows information fusion in each channel
    """
    def __init__(self, n_features, n_channels, n_transforms, hidden_dim=64):
        super().__init__()
        self.n_channels = n_channels
        self.mlp = MLPModule(n_features*n_channels + n_transforms, n_transforms*n_channels, hidden_dim*n_channels)
        self.act = nn.Sigmoid()

    def forward(self, cond, prev_prob):
        """
        :param cond: (batch, n_channels, n_features)
        :param prev_prob: (batch, n_transforms)
        :return: (batch, n_channels, n_transforms)
        """
           
        batch_size, n_channels, n_features = cond.shape
        assert n_channels == self.n_channels

        # Reshape cond to (batch, n_channels * n_features)
        cond_flattened = cond.view(batch_size, -1)

        # Concatenate flattened cond and prev_prob
        # input: (batch, n_channels * n_features+n_transforms)
        inputs = torch.cat([cond_flattened, prev_prob], dim=1)

        # Apply MLP and activation
        # input: (batch, n_transforms*n_channels)
        output = self.act(self.mlp(inputs))

        # Reshape output to (batch, n_channels, n_transforms)
        return output.view(batch_size, self.n_channels, -1)

class AugmentLayer(nn.Module):
    def __init__(self, global_config: GlobalConfig, seq_len, n_channels, n_features,
                 transforms:List[Type[AugmentTransform]]=None):
        super().__init__()
        self.global_config = global_config
        self.configs = self.global_config.args
        self.device = self.global_config.device
        if transforms is None:
            transforms:List[Type[AugmentTransform]] = AVAILABLE_TRANSFORMS
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.cond_dim = n_features
        self.transforms:List[AugmentTransform] = [trs(n_features, n_channels, seq_len, device=self.global_config.device)
                                                  for trs in transforms]
        self.n_transforms = len(self.transforms)
        self.tau = float(_parse_config(["tau"], 10))
        if _parse_config(["learnable_tau","l_tau"], 0)==1:
            self.tau = nn.Parameter(torch.ones(1) * self.tau)  # learnable

        self.P = AugProbMLPModule(n_features=n_features, n_channels=n_channels, n_transforms=self.n_transforms,
                                  global_config=self.global_config)
        self.S = AugStrengthMLPModule(n_features=n_features, n_channels=n_channels, n_transforms=self.n_transforms)

    def forward(self, batch_x, batch_y, batch_f, batch_mask, prev_prob):
        """
        :param batch_x: input x (batch, n_channels, seq_len)
        :param batch_y: label y (batch, label_len) or (batch, n_channels, label_len)
        :param batch_f: features (batch, n_channels, n_features)
        :param batch_mask: (batch, seq_len)
        :param prev_prob: (batch, n_transforms)
        :return: augmented batch_x, batch_y, batch_mask and augmentation probability
        """
        batch, n_channels, seq_len = batch_x.shape
        if prev_prob is None:
            prev_prob = torch.ones(size=(batch, self.n_transforms), device=self.device) / self.n_transforms

        # p shape: (batch, n_transforms)
        p = self.P(batch_f, prev_prob)

        # strength shape: (batch, n_channels, n_transforms)
        strength = self.S(batch_f, prev_prob)
        # choice shape: (batch, n_transforms) one_hot_vectors
        choice = F.gumbel_softmax(p, tau=self.tau, hard=True).view(batch, self.n_transforms)

        # current_strength (batch, n_channels, 1)
        current_strength = torch.sum(strength * choice.view(batch, 1, self.n_transforms), dim=2, keepdim=True)

        # Apply transformation based on choice
        aug_x, aug_y, aug_mask = self._apply_transform4(batch_x, batch_y, batch_f, batch_mask, choice, current_strength)
        return aug_x, aug_y, aug_mask, choice, p

    def _apply_transform4(self, batch_x, batch_y, batch_f, batch_mask, choice, current_strength):
        """
        :param batch_x: (batch, n_channels, seq_len)
        :param batch_y: (batch, label_len) or (batch, n_channels, label_len)
        :param batch_f: (batch, n_channels, n_features)
        :param batch_mask: (batch, seq_len)
        :param choice:  (batch, n_transforms)
        :param current_strength: (batch, n_channels, 1)
        :return: augmented x, y, mask
        """
        augmented_x = torch.zeros_like(batch_x, device=self.device)
        augmented_y = torch.zeros_like(batch_y, device=self.device)
        augmented_mask = torch.zeros_like(batch_mask, device=self.device)


        for i, transform in enumerate(self.transforms):
            # (batch,)
            selected = choice[:, i].bool()
            if selected.any():
                # Reshape inputs to match the expected dimensions for the transform
                aug_x, aug_y, aug_mask = transform(batch_x[selected],
                                                   batch_y[selected],
                                                   batch_f[selected],
                                                   batch_mask[selected],
                                                   current_strength[selected])

                # Reshape the transformed output back to the original dimensions
                augmented_x[selected] = aug_x
                augmented_y[selected] = aug_y
                augmented_mask[selected] = aug_mask
        return augmented_x, augmented_y, augmented_mask

class AugmentModel(nn.Module):
    def __init__(self, global_config:GlobalConfig, seq_len, n_channels, n_features, n_layers=1, transforms=AVAILABLE_TRANSFORMS):
        super().__init__()
        self.global_config = global_config
        self.configs = self.global_config.args
        self.n_channels = n_channels
        self.layers = nn.ModuleList([AugmentLayer(global_config, seq_len, n_channels, n_features, transforms)
                                     for _ in range(n_layers)])
        self.composite_loss:[None|CompositeLoss] = None

    def set_criterion(self, criterion: CompositeLoss):
        self.composite_loss = criterion

    def forward(self, batch_x, batch_y, batch_f, batch_mask):
        prev_prob = None
        for i, layer in enumerate(self.layers):
            batch_x, batch_y, batch_mask, prev_prob, p = layer(batch_x, batch_y, batch_f, batch_mask, prev_prob)
            if isinstance(self.composite_loss, CompositeLoss):
                self.composite_loss.record_prob(p, i)

        return batch_x, batch_y, batch_mask

class Model(AutoAugmentBasic):
    def __init__(self, global_config: GlobalConfig):
        super().__init__(global_config)
        self.criterion: CompositeLoss = None

    def _initialize(self):
        self.configs = self.global_config.args
        self.n_channels = self.configs.n_channels
        self.seq_len = self.configs.seq_len
        self.n_features = self.configs.n_features
        self.n_layers = 3  # follow previous works

        self.augment_model = AugmentModel(self.global_config, self.seq_len, self.n_channels, self.n_features,
                                          self.n_layers, transforms=AVAILABLE_TRANSFORMS)
        self.downstream_model = build_downstream_model(self.configs.task, self.configs.downstream, self.global_config)
        self.criterion: CompositeLoss = None

    def _perform_augment(self, batch_x, batch_y, batch_f, batch_masks):
        augmented_x, augmented_y, augmented_masks = self.augment_model(batch_x, batch_y, batch_f, batch_masks)
        return augmented_x, augmented_y, augmented_masks

    def get_criterion(self, default_criterion):
        ew = float(_parse_config(["entropy_weight", "ew"], 0))
        dw = float(_parse_config(["diversity_weight", "dw"], 0))
        dy_loss = bool(_parse_config(["dyLoss", "dyl"], 0))
        if ew>0 or dw>0:
            print(f"[{self.__class__.__name__}] Use composite loss")
            criterion = CompositeLoss(default_criterion, self.n_layers, learnable_weight=dy_loss,
                                      task_weight=1.0, entropy_weight=ew, diversity_weight=dw)
            self.augment_model.set_criterion(criterion)
            self.criterion = criterion
            return criterion
        else:
            print(f"[{self.__class__.__name__}] Use default loss")
            return default_criterion

    def summarize_state(self)->str:
        msg = ""
        msg += f"\nTau: {[_param_value(lyr.tau) for lyr in self.augment_model.layers]}"
        #if self.criterion:
        #    msg += self.criterion.get_state()
        return msg

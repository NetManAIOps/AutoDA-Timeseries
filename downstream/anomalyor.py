import torch.nn as nn
import torch.nn.functional as F
from .downstream_base import DownstreamModelBase
import math

import torch
import os

from types import SimpleNamespace


# TimesNet
from .Embed import DataEmbedding, DataEmbedding_wo_pos
from .Conv_Blocks import Inception_Block_V1
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp

# KAN-AD
import numpy as np

class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.order = int(order)
        self.window = int(window)
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine(self.window, self.order).unsqueeze(0),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Linear(self.window, self.window)

    def forward(self, x: torch.Tensor, return_last: bool = False, *args, **kwargs):
        res = []
        res.append(x.unsqueeze(1))  # [B,1,L]
        ff = torch.cat(
            [self.orders.repeat(x.size(0), 1, 1)]  # [B, order, L]
            + [torch.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]  # [(B,1,L)] * order
            + [x.unsqueeze(1)],  # [B,1,L]
            dim=1,
        )  # [B, channels, L]
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()  # [B,1,L]
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        if return_last:
            return ff.squeeze(1), ff
        return ff.squeeze(1)                # [B,L]

    def _create_custom_periodic_cosine(self, window: int, period) -> torch.Tensor:
        d = len(period) if isinstance(period, list) else period
        pl = period if isinstance(period, list) else [i for i in range(1, period + 1)]
        result = torch.empty(d, window, dtype=torch.float32)
        for i, p in enumerate(pl):
            t = torch.arange(0, 1, 1 / window, dtype=torch.float32) / p * 2 * np.pi
            result[i, :] = torch.cos(t)
        return result


class KANADAnomaly(DownstreamModelBase):

    def _build_model(self):
        args = self.downstream_args
        self.order = int(args.get("order", args.get("d_model", 8)))
        self.normalize = bool(args.get("normalize", True))
        # self._window = None
        # self.enc = None
        self.eps = 1e-5

        L0 = int(args.get("seq_len", 100))
        self.enc = KANADModel(window=L0, order=self.order)
        self._window = L0

    def _ensure_model(self, L: int):
        if (self.enc is None) or (self._window != L):
            self._window = int(L)
            self.enc = KANADModel(window=self._window, order=self.order).to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _norm(self, x):
        if not self.normalize:
            B, C, _ = x.shape
            mean = torch.zeros((B, C, 1), device=x.device, dtype=x.dtype)
            std  = torch.ones((B, C, 1),  device=x.device, dtype=x.dtype)
            return x, mean, std
        mean = x.mean(dim=-1, keepdim=True).detach()
        var  = x.var (dim=-1, keepdim=True, unbiased=False).detach()
        std  = torch.sqrt(var + self.eps)
        return (x - mean) / std, mean, std

    def forward(self, batch_x, batch_f=None, batch_mask=None):
        """
        batch_x : [B, C, L]
        return  : [B, C, L]
        """
        x = batch_x
        B, C, L = x.shape

        self._ensure_model(L)

        x_n, mean, std = self._norm(x)                 # [B,C,L]

        xt = x_n.permute(0, 2, 1).contiguous()
        xbc = xt.reshape(B * C, L)

        ybc = self.enc(xbc)                            # [B*C, L]

        yt = ybc.view(B, L, C)
        y  = yt.permute(0, 2, 1).contiguous()          # [B,C,L]

        x_hat = y * std + mean

        if batch_mask is not None:
            m = batch_mask.to(x_hat.dtype).unsqueeze(1)  # [B,1,L]
            x_hat = x_hat * m + x * (1 - m)

        return x_hat




def _fft_for_period(x, k=2):
    """
    x: [B, T, C]
    return:
        period_list: np.ndarray shape [k]
        period_weight: [B, k]
    """
    xf = torch.fft.rfft(x, dim=1)                    # [B, T//2+1, C]
    amp = xf.abs()                                   # [B, F, C]
    freq_score = amp.mean(0).mean(-1)                # [F]
    freq_score[0] = 0
    _, top_idx = torch.topk(freq_score, k)           # [k]
    period = x.shape[1] // top_idx.detach().cpu().numpy()
    weight = amp.mean(-1)[:, top_idx]                # [B, k]
    return period, weight


class _TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.k = int(configs.top_k)

        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        """
        x: [B, T, C]
        return: [B, T, C]
        """
        B, T, N = x.size()
        period_list, period_weight = _fft_for_period(x, self.k)

        outs = []
        target_len = self.seq_len + self.pred_len
        for i in range(self.k):
            p = int(period_list[i])
            p = max(p, 1) # Avoid p = 0.

            if target_len % p != 0:
                length = ((target_len // p) + 1) * p
                pad = torch.zeros(B, length - target_len, N, device=x.device, dtype=x.dtype)
                out = torch.cat([x, pad], dim=1)
            else:
                length = target_len
                out = x

            # [B, length//p, p, N] -> [B, N, length//p, p]
            out = out.reshape(B, length // p, p, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)                    # 2D conv
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)[:, :target_len, :]
            outs.append(out)

        outs = torch.stack(outs, dim=-1)            # [B, T, C, k]
        w = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)  # [B,1,1,k]
        outs = (outs * w).sum(dim=-1)               # [B, T, C]
        return outs + x                          


class TimesNetAnomaly(DownstreamModelBase):
    def _build_model(self):
        args = self.downstream_args

        self.d_model     = int(args.get("d_model", 64))
        self.d_ff        = int(args.get("d_ff", 128))
        self.e_layers    = int(args.get("e_layers", 2))
        self.num_kernels = int(args.get("num_kernels", 6))
        self.top_k       = int(args.get("top_k", 3))
        self.embed       = args.get("embed", "timeF")
        self.freq        = args.get("freq", "h")
        self.dropout     = float(args.get("dropout", 0.1))

        self.pred_len = 0
        self.seq_len  = int(args.get("seq_len", 0))

        self.proj = nn.Linear(self.d_model, self.n_channels, bias=True)

        cfg = SimpleNamespace(
            seq_len=max(self.seq_len, 1),
            pred_len=self.pred_len,
            top_k=self.top_k,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
        )

        self.enc_embedding = DataEmbedding(self.n_channels, self.d_model, self.embed, self.freq, self.dropout)
        self.blocks = nn.ModuleList([_TimesBlock(cfg) for _ in range(self.e_layers)])
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.eps = 1e-5
        self.normalize = bool(args.get("normalize", True))

    def _ensure_seq_len(self, L: int):
        if self.seq_len != L:
            self.seq_len = L
            for m in self.blocks:
                m.seq_len = L

    def _norm(self, x):
        # x: [B, C, L]
        if not self.normalize:
            B, C, _ = x.shape
            device, dtype = x.device, x.dtype
            mean = torch.zeros((B, C, 1), device=device, dtype=dtype)
            std  = torch.ones((B, C, 1),  device=device, dtype=dtype)
            return x, mean, std
        mean = x.mean(dim=-1, keepdim=True).detach()
        var  = x.var (dim=-1, keepdim=True, unbiased=False).detach()
        std  = torch.sqrt(var + self.eps)
        return (x - mean) / std, mean, std

    def forward(self, batch_x, batch_f=None, batch_mask=None):
        """
        batch_x : [B, C, L]
        return  : [B, C, L]
        """
        x = batch_x
        B, C, L = x.shape
        self._ensure_seq_len(L)

        x_n, mean, std = self._norm(x)              # [B,C,L]

        xt = x_n.permute(0, 2, 1).contiguous()      # [B,L,C]

        h = self.enc_embedding(xt, None)            # [B,L,d_model]

        for blk in self.blocks:
            h = self.layer_norm(blk(h))             # [B,L,d_model]

        y = self.proj(h)                            # [B,L,C]
        y = y.permute(0, 2, 1).contiguous()         # [B,C,L]

        x_hat = y * std + mean

        if batch_mask is not None:
            m = batch_mask.to(x_hat.dtype).unsqueeze(1)  # [B,1,L]
            x_hat = x_hat * m + x * (1 - m)

        return x_hat






class AutoformerAnomaly(DownstreamModelBase):

    def _build_model(self):
        args = self.downstream_args

        self.d_model   = int(args.get("d_model", 64))
        self.d_ff      = int(args.get("d_ff", 128))
        self.n_heads   = int(args.get("n_heads", 4))
        self.e_layers  = int(args.get("e_layers", 2))
        self.factor    = int(args.get("factor", 3))
        self.moving_avg = int(args.get("moving_avg", 25))
        self.activation = args.get("activation", "gelu")
        self.embed     = args.get("embed", "timeF")
        self.freq      = args.get("freq", "h")
        self.dropout   = float(args.get("dropout", 0.1))
        self.normalize = bool(args.get("normalize", True))

        self.pred_len = 0
        self.seq_len  = int(args.get("seq_len", 0))

        self.enc_embedding = DataEmbedding_wo_pos(self.n_channels, self.d_model, self.embed, self.freq, self.dropout)

        # Encoder stacked
        enc_layers = [
            EncoderLayer(
                AutoCorrelationLayer(
                    AutoCorrelation(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                    self.d_model, self.n_heads
                ),
                d_model=self.d_model,
                d_ff=self.d_ff,
                moving_avg=self.moving_avg,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(self.e_layers)
        ]
        self.encoder = Encoder(enc_layers, norm_layer=my_Layernorm(self.d_model))

        self.proj = nn.Linear(self.d_model, self.n_channels, bias=True)

        self.eps = 1e-5

    def _ensure_seq_len(self, L: int):
        if self.seq_len != L:
            self.seq_len = L

    def _norm(self, x: torch.Tensor):
        if not self.normalize:
            B, C, _ = x.shape
            device, dtype = x.device, x.dtype
            mean = torch.zeros((B, C, 1), device=device, dtype=dtype)
            std  = torch.ones((B, C, 1),  device=device, dtype=dtype)
            return x, mean, std
        mean = x.mean(dim=-1, keepdim=True).detach()
        var  = x.var (dim=-1, keepdim=True, unbiased=False).detach()
        std  = torch.sqrt(var + self.eps)
        return (x - mean) / std, mean, std

    def forward(self, batch_x, batch_f=None, batch_mask=None):
        """
        batch_x : [B, C, L]
        return  : x_hat [B, C, L]
        """
        x = batch_x
        B, C, L = x.shape
        self._ensure_seq_len(L)

        x_n, mean, std = self._norm(x)                  # [B,C,L]

        xt = x_n.permute(0, 2, 1).contiguous()          # [B,L,C]

        h = self.enc_embedding(xt, None)                # [B,L,d_model]
        h, _ = self.encoder(h, attn_mask=None)          # [B,L,d_model]


        print(f"h.shape: {h.shape}")

        y = self.proj(h)                                # [B,L,C]
        y = y.permute(0, 2, 1).contiguous()             # [B,C,L]

        x_hat = y * std + mean

        if batch_mask is not None:
            m = batch_mask.to(x_hat.dtype).unsqueeze(1)  # [B,1,L]
            x_hat = x_hat * m + x * (1 - m)

        return x_hat


AVAILABLE_ANOMALYORS = {
    # "LSTM":LSTMAEAnomaly,
    # "UNet":UNetAnomalyDetector,
    # "VAE":VAEAnomalyDetector,
    "TimesNet":TimesNetAnomaly,
    "KAN":KANADAnomaly,
    "Autoformer":AutoformerAnomaly
}

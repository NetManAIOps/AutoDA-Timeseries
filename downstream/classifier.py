import torch.nn as nn
import torch.nn.functional as F
from .downstream_base import DownstreamModelBase
import math
import torch


class TCNClassifier(DownstreamModelBase):
    
    class _SamePadConv1d(nn.Module):
        def __init__(self, in_ch, out_ch, k, dilation=1):
            super().__init__()
            rf = (k - 1) * dilation + 1
            pad = rf // 2
            self.remove = 1 if (rf % 2 == 0) else 0
            self.conv = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=dilation)

        def forward(self, x):
            y = self.conv(x)  # [B, C, T or T+1]
            if self.remove > 0:
                y = y[:, :, :-self.remove]
            return y

    class _TemporalBlock(nn.Module):
        def __init__(self, in_ch, out_ch, k=3, dilation=1, dropout=0.1, use_gn=False, groups=8):
            super().__init__()
            Norm = (lambda c: nn.GroupNorm(num_groups=min(groups, c), num_channels=c)) if use_gn else nn.BatchNorm1d

            self.conv1 = TCNClassifier._SamePadConv1d(in_ch, out_ch, k, dilation=dilation)
            self.bn1   = Norm(out_ch)
            self.conv2 = TCNClassifier._SamePadConv1d(out_ch, out_ch, k, dilation=dilation)
            self.bn2   = Norm(out_ch)
            self.act   = nn.GELU()
            self.drop  = nn.Dropout(dropout)
            self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

        def forward(self, x):  # x:[B,C,T]
            y = self.act(self.bn1(self.conv1(x)))
            y = self.drop(y)
            y = self.act(self.bn2(self.conv2(y)))
            y = self.drop(y)
            res = x if self.down is None else self.down(x)
            return self.act(y + res)

    def _build_model(self):
        args = self.downstream_args
        base_ch   = int(args.get("channels", 64))
        num_layers = int(args.get("num_layers", 4))
        kernel     = int(args.get("kernel_size", 3))
        dropout    = float(args.get("dropout", 0.1))
        doubling   = bool(args.get("doubling", True))
        use_gn     = bool(args.get("use_gn", False))
        groups     = int(args.get("groups", 8))
        self.pool_mode = str(args.get("pool", "gap"))  # "gap" or "last"

        chs = []
        cur = base_ch
        for i in range(num_layers):
            chs.append(cur)
            if doubling:
                cur *= 2
        self._out_ch = chs[-1]

        blocks = []
        in_ch = self.n_channels
        for i, out_ch in enumerate(chs):
            dilation = 2 ** i
            blocks.append(self._TemporalBlock(
                in_ch, out_ch, k=kernel, dilation=dilation,
                dropout=dropout, use_gn=use_gn, groups=groups
            ))
            in_ch = out_ch
        self.tcn = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self._out_ch, self.pred_len)

    def forward(self, batch_x, batch_f, batch_mask):
        
        x = batch_x  # [B,C,T]
        x = self.tcn(x)  # [B,C_out,T]
        if self.pool_mode == "last":
            feat = x[:, :, -1]                     # [B,C_out]
        else:
            feat = x.mean(dim=-1)                  # GAP: [B,C_out]
        feat = self.dropout(feat)
        out = self.fc(feat)                        # [B,pred_len]
        return out


class ROCKETClassifier(DownstreamModelBase):
    
    def _build_model(self):
        args = self.downstream_args
        self.num_kernels  = int(args.get("num_kernels", 2000))
        # self.kernel_sizes = list(args.get("kernel_sizes", [7, 9, 11]))
        self.kernel_sizes = list(args.get("kernel_sizes", [3, 5, 7]))
        self.normalise    = bool(args.get("normalise", True))
        self.use_channel_wise = bool(args.get("use_channel_wise", False))
        self.dropout = nn.Dropout(float(args.get("dropout", 0.0)))


        T = self.seq_len
        raw_dil = args.get("dilations", "auto")
        if isinstance(raw_dil, str) and raw_dil == "auto":

            max_d = max(1, T // 2)
            get_dil = lambda ks: int(torch.randint(1, max(2, min(max_d, max(2, T // ks))), (1,)).item())
        elif isinstance(raw_dil, int):
            get_dil = lambda ks: int(torch.randint(1, max(2, raw_dil+1), (1,)).item())
        else:
            dil_list = list(raw_dil)
            get_dil = lambda ks: int(dil_list[torch.randint(0, len(dil_list), (1,)).item()])


        ks_list, dil_list, ch_pick = [], [], []
        for _ in range(self.num_kernels):
            ks = int(self.kernel_sizes[torch.randint(0, len(self.kernel_sizes), (1,)).item()])
            d  = get_dil(ks)
            ks_list.append(ks)
            dil_list.append(d)
            if self.use_channel_wise:
                ch_pick.append(int(torch.randint(0, self.n_channels, (1,)).item()))
            else:
                ch_pick.append(-1)  # -1 

        self.register_buffer("_ks", torch.tensor(ks_list, dtype=torch.long))
        self.register_buffer("_dil", torch.tensor(dil_list, dtype=torch.long))
        self.register_buffer("_chpick", torch.tensor(ch_pick, dtype=torch.long))


        from collections import defaultdict
        group_idx = defaultdict(list)
        for i, (ks, d) in enumerate(zip(ks_list, dil_list)):
            group_idx[(ks, d)].append(i)
        self._groups_meta = []  # [(ks, d, start, end)]
        self._weight = nn.ParameterList()  
        self._bias   = nn.ParameterList()
        self._chmask = [] 

        for (ks, d), idxs in group_idx.items():
            ng = len(idxs)
            if self.use_channel_wise:
                w = torch.zeros(ng, 1, ks)
            else:
                w = torch.randn(ng, self.n_channels, ks)
            if self.normalise:
                w = w - w.mean(dim=(1,2), keepdim=True)
                w = w / (w.norm(p=2, dim=(1,2), keepdim=True) + 1e-8)

            b = torch.empty(ng).uniform_(-1.0, 1.0)

            p_w = nn.Parameter(w, requires_grad=False)
            p_b = nn.Parameter(b, requires_grad=False)
            self._weight.append(p_w)
            self._bias.append(p_b)
            self._groups_meta.append((ks, d, torch.tensor(idxs, dtype=torch.long)))
            self._chmask.append(self._chpick[idxs])

        self.head = nn.Linear(2 * self.num_kernels, self.pred_len)

    def _conv_group(self, x, w, b, ks, dil, chpick):

        B, C, T = x.shape
        ng = w.size(0)

        if self.use_channel_wise:

            from collections import defaultdict
            per_ch = defaultdict(list)
            for i in range(ng):
                per_ch[int(chpick[i].item())].append(i)
            feats_max, feats_ppv = [], []
            for ch, idxs in per_ch.items():
                # [B,1,T] × [nch,1,ks] -> [B,nch,T']
                xch = x[:, ch:ch+1, :]
                wsub = w[idxs, :, :]
                bsub = b[idxs]
                y = F.conv1d(xch, wsub, bias=bsub, stride=1, padding=0, dilation=dil)  # valid
                fmax = y.amax(dim=-1)                     # [B,nch]
                fppv = (y > 0).float().mean(dim=-1)       # [B,nch]
                feats_max.append(fmax)
                feats_ppv.append(fppv)
            feat_max = torch.cat(feats_max, dim=1)
            feat_ppv = torch.cat(feats_ppv, dim=1)
        else:
            y = F.conv1d(x, w, bias=b, stride=1, padding=0, dilation=dil)
            feat_max = y.amax(dim=-1)                   # [B,ng]
            feat_ppv = (y > 0).float().mean(dim=-1)     # [B,ng]

        return feat_max, feat_ppv

    def forward(self, batch_x, batch_f, batch_mask):
        
        x = batch_x
        feats = []

        for (ks, dil, idxs), w, b, chpick in zip(self._groups_meta, self._weight, self._bias, self._chmask):
            fmax, fppv = self._conv_group(x, w, b, ks=int(ks), dil=int(dil), chpick=chpick)
            feats.append(fmax)
            feats.append(fppv)

        feats = torch.cat(feats, dim=1)           # [B, 2*num_kernels]
        feats = self.dropout(feats)
        out = self.head(feats)                    # [B, pred_len]
        return out



AVAILABLE_CLASSIFIERS = {
    "TCN":TCNClassifier,
    "ROCKET":ROCKETClassifier,
}
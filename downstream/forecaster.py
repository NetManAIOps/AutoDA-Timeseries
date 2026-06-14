import torch.nn as nn
import torch.nn.functional as F
from .downstream_base import DownstreamModelBase
import math


import torch

# PatchTST
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import PatchEmbedding, DataEmbedding_inverted, DataEmbedding
from .Conv_Blocks import Inception_Block_V1


class RNNForecaster(DownstreamModelBase):
    """
    Simple two-tower RNN forecaster:
      - Encoder: processes x_enc (+ x_mark_enc) and returns hidden state h.
      - Decoder: uses h as the initial state, processes x_dec (+ x_mark_dec),
        projects to output channels, and keeps the last pred_len steps.
    Shapes:
      x_enc:      [B, C, Lx]
      x_mark_enc: [B, Lx, Denc]
      x_dec:      [B, C, Ld]              (Ld = label_len + pred_len)
      x_mark_dec: [B, Ld, Ddec]
      return:     [B, C_out, pred_len]
    """

    def _build_model(self):
        # Task and data dimensions.
        self.pred_len  = self.configs.pred_len
        self.label_len = self.configs.label_len
        self.enc_in    = self.configs.n_channels
        # self.c_out     = self.downstream_args.get("c_out", self.enc_in)
        self.c_out = self.configs.n_channels

        # RNN and projection hyperparameters.
        self.hidden_size   = self.downstream_args.get("d_model", 256)
        self.num_layers    = self.downstream_args.get("num_layers", 2)
        self.dropout       = self.downstream_args.get("dropout", 0.1)
        self.rnn_type      = self.downstream_args.get("rnn_type", "GRU")  # "GRU" or "LSTM"

        # Time-feature switch and projection dimensions.
        self.use_time_features = self.downstream_args.get("use_time_features", True)
        self.denc_proj = self.downstream_args.get("denc_proj", 16)  # encoder time-feature projection dim
        self.ddec_proj = self.downstream_args.get("ddec_proj", 16)  # decoder time-feature projection dim

        # Normalization switch, per sample and per channel along the time axis.
        self.normalize = self.downstream_args.get("normalize", True)
        self.eps = 1e-5

        # Input dimensions. [B,C,L] is converted to [B,L,C] before the RNN,
        # then concatenated with time features.
        enc_input_size = self.enc_in
        dec_input_size = self.enc_in

        if self.use_time_features:
            # Time-feature dimensions are known only in forward.
            # LazyLinear infers in_features on the first forward pass.
            self.enc_time_proj = nn.LazyLinear(self.denc_proj)
            self.dec_time_proj = nn.LazyLinear(self.ddec_proj)
            enc_input_size += self.denc_proj
            dec_input_size += self.ddec_proj
        else:
            self.enc_time_proj = None
            self.dec_time_proj = None

        # Build the RNN encoder-decoder.
        RNN = nn.LSTM if self.rnn_type.upper() == "LSTM" else nn.GRU

        self.encoder = RNN(
            input_size=enc_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.decoder = RNN(
            input_size=dec_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Project outputs to channel count.
        self.proj_out = nn.Linear(self.hidden_size, self.c_out, bias=False)

    def _norm(self, x_btC):
        """
        x_btC: [B, T, C]
        Return normalized x and the mean/std used for denormalization.
        """
        if not self.normalize:
            B, T, C = x_btC.shape
            device = x_btC.device
            mean = torch.zeros((B, 1, C), device=device, dtype=x_btC.dtype)
            std  = torch.ones((B, 1, C), device=device, dtype=x_btC.dtype)
            return x_btC, mean, std
        mean = x_btC.mean(dim=1, keepdim=True).detach()
        var  = x_btC.var(dim=1, keepdim=True, unbiased=False).detach()
        std  = torch.sqrt(var + self.eps)
        return (x_btC - mean) / std, mean, std

    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        """
        x_enc: [B, C, Lx]
        x_dec: [B, C, Ld]  (Ld = label_len + pred_len)
        x_mark_enc: [B, Lx, Denc] or None
        x_mark_dec: [B, Ld, Ddec] or None
        return: [B, C_out, pred_len]
        """
        B, C, Lx = x_enc.shape
        _, _, Ld = x_dec.shape

        # print(f"x_enc.shape: {x_enc.shape}") # [16, 1, 18]
        # print(f"x_dec.shape: {x_dec.shape}") # [16, 1, 36]
        # exit(0)

        # 1) Convert to [B, L, C].
        enc_val = x_enc.permute(0, 2, 1).contiguous()  # [B, Lx, C]
        dec_val = x_dec.permute(0, 2, 1).contiguous()  # [B, Ld, C]

        # 2) Optionally normalize per sample and per channel along time.
        enc_norm, mean, std = self._norm(enc_val)

        # 3) Concatenate time features if enabled.
        if self.use_time_features and x_mark_enc is not None:
            # x_mark_enc: [B, Lx, Denc] -> project to denc_proj.
            enc_tf = self.enc_time_proj(x_mark_enc)  # [B, Lx, denc_proj]
            enc_in = torch.cat([enc_norm, enc_tf], dim=-1)  # [B, Lx, C + denc_proj]
        else:
            enc_in = enc_norm  # [B, Lx, C]

        if self.use_time_features and x_mark_dec is not None:
            dec_tf = self.dec_time_proj(x_mark_dec)  # [B, Ld, ddec_proj]
            dec_in = torch.cat([dec_val, dec_tf], dim=-1)   # [B, Ld, C + ddec_proj]
        else:
            dec_in = dec_val  # [B, Ld, C]

        # 4) Encode and use the final hidden state as the decoder initial state.
        enc_out, enc_state = self.encoder(enc_in)  # enc_out: [B, Lx, H], enc_state: (h,c) or h
        # 5) Decode with teacher forcing over label_len + pred_len.
        dec_out, _ = self.decoder(dec_in, enc_state)  # [B, Ld, H]

        # 6) Project outputs and keep the last pred_len steps.
        dec_y = self.proj_out(dec_out)          # [B, Ld, C_out]
        dec_y = dec_y[:, -self.pred_len:, :]    # [B, pred_len, C_out]

        # print(f"dec_y.shape: {dec_y.shape}")
        # exit(0)

        # 7) Denormalize to the original scale. mean/std come from the encoder input.
        dec_y = dec_y * std + mean              # broadcast [B, pred_len, C_out]

        # 8) Convert back to [B, C_out, pred_len].
        dec_y = dec_y.permute(0, 2, 1).contiguous()

        # print(f"dec_y.shape: {dec_y.shape}") # [16, 7, 100]
        # exit(0)
        return dec_y



class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)       # flatten d_model × patch_num
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x: [B, n_vars, d_model, patch_num]
        x = self.flatten(x)         # → [B, n_vars, nf]
        x = self.linear(x)          # → [B, n_vars, target_window]
        x = self.dropout(x)
        return x


# ============================================================
#                  PatchTST Forecaster
# ============================================================

class PatchTSTForecaster(DownstreamModelBase):

    def _build_model(self):
        """
        Build PatchTST components according to your framework style.
        """
        cfg = self.configs
        args = self.downstream_args

        # Basic config
        self.pred_len  = cfg.pred_len
        self.label_len = cfg.label_len
        self.enc_in    = cfg.n_channels        # number of variables
        self.c_out     = cfg.n_channels        # multi-variate forecasting

        # PatchTST hyperparams (from downstream_args or configs)
        self.d_model   = args.get("d_model", getattr(cfg, "d_model", 512))
        self.d_ff      = args.get("d_ff", getattr(cfg, "d_ff", 2048))
        self.n_heads   = args.get("n_heads", getattr(cfg, "n_heads", 8))
        self.e_layers  = args.get("e_layers", getattr(cfg, "e_layers", 2))
        self.dropout   = args.get("dropout", getattr(cfg, "dropout", 0.1))
        self.factor    = args.get("factor", getattr(cfg, "factor", 1))
        self.activation = args.get("activation", getattr(cfg, "activation", "gelu"))

        # patch params
        self.patch_len = args.get("patch_len", 8)
        self.stride    = args.get("stride", 4)
        self.padding   = self.stride           # consistent with original

        # --------------------------------------------------------
        # Build patch embedding: output is [B*C, patch_num, d_model]
        # --------------------------------------------------------
        self.patch_embedding = PatchEmbedding(
            self.d_model,
            self.patch_len,
            self.stride,
            self.padding,
            self.dropout
        )

        # --------------------------------------------------------
        # Encoder: same as PatchTST
        # --------------------------------------------------------
        encoder_layers = [
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False),
                    self.d_model,
                    self.n_heads
                ),
                self.d_model,
                self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            )
            for _ in range(self.e_layers)
        ]

        # BN along the embedding dimension
        self.encoder = Encoder(
            encoder_layers,
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(self.d_model),
                Transpose(1, 2)
            )
        )

        # --------------------------------------------------------
        # Prediction head
        # head_nf = d_model × patch_num
        # patch_num = int((seq_len - patch_len) / stride + 2)
        # --------------------------------------------------------
        patch_num = int((cfg.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_model * patch_num

        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.pred_len,
            head_dropout=self.dropout
        )

        # Normalize (your framework style)
        self.normalize = args.get("normalize", True)
        self.eps = 1e-5

    # ------------------------------------------------------------
    # Your framework normalization: per-sample, per-channel
    # ------------------------------------------------------------
    def _norm(self, x_btC):
        """
        x_btC: [B, T, C]
        return normalized_x, mean, std
        """
        if not self.normalize:
            B, T, C = x_btC.shape
            mean = torch.zeros((B, 1, C), device=x_btC.device)
            std  = torch.ones((B, 1, C), device=x_btC.device)
            return x_btC, mean, std

        mean = x_btC.mean(dim=1, keepdim=True).detach()
        var  = x_btC.var(dim=1, keepdim=True, unbiased=False).detach()
        std  = torch.sqrt(var + self.eps)
        return (x_btC - mean) / std, mean, std

    # ------------------------------------------------------------
    #                       FORWARD
    # ------------------------------------------------------------
    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        """
        x_enc: [B, C, Lx]
        x_dec: [B, C, Ld]
        Return: [B, C, pred_len]
        (PatchTST uses only x_enc)
        """
        B, C, L = x_enc.shape
        # print(f"x_enc.shape: {x_enc.shape}")

        # 1) Move to [B, L, C]
        x = x_enc.permute(0, 2, 1).contiguous()   # [B, L, C]

        # 2) your normalization
        x_norm, mean, std = self._norm(x)
        # print(f"x_norm.shape: {x_norm.shape}") # [128, 96, 7]

        # 3) PatchTST expects [B, L, C] then permute to [B, C, L]
        # x_norm = x_norm.permute(0, 2, 1).contiguous()   # [B, C, L]

        # 4) Patch embedding: output enc_out [B*C, patch_num, d_model]
        # x_input = x_norm.permute(0, 2, 1)               # [B, L, C]
        enc_out, n_vars = self.patch_embedding(x_norm)
        
        # print(f"enc_out.shape: {enc_out.shape}")

        # 5) Encoder
        enc_out, _ = self.encoder(enc_out)

        # 6) reshape to [B, C, d_model, patch_num]
        # print(f"enc_out.shape: {enc_out.shape}") # [12288, 1, 512]
        enc_out = enc_out.reshape(B, C, enc_out.shape[-2], enc_out.shape[-1])
        
        enc_out = enc_out.permute(0, 1, 3, 2)           # [B, C, d_model, patch_num]

        # 7) Prediction head → [B, C, pred_len]
        dec_y = self.head(enc_out)
        dec_y = dec_y.permute(0, 2, 1)   # → [B, pred_len, C]

        # 8) denormalize
        dec_y = dec_y * std + mean       # broadcast [B, 1, C]

        # 9) return [B, C, pred_len]
        return dec_y.permute(0, 2, 1).contiguous()


class iTransformerForecaster(DownstreamModelBase):

    def _build_model(self):
        cfg = self.configs
        args = self.downstream_args
        
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        
        # Hyperparameters
        self.d_model = args.get("d_model", getattr(cfg, "d_model", 128))
        self.embed = args.get("embed", getattr(cfg, "embed", "timeF"))
        self.freq = args.get("freq", getattr(cfg, "freq", "h"))
        self.dropout = args.get("dropout", getattr(cfg, "dropout", 0.1))
        self.factor = args.get("factor", getattr(cfg, "factor", 3))
        self.n_heads = args.get("n_heads", getattr(cfg, "n_heads", 8))
        self.d_ff = args.get("d_ff", getattr(cfg, "d_ff", 128))
        self.activation = args.get("activation", getattr(cfg, "activation", "gelu"))
        self.e_layers = args.get("e_layers", getattr(cfg, "e_layers", 2))

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, self.d_model, self.embed, self.freq, self.dropout)
        
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        
        # Decoder
        self.projection = nn.Linear(self.d_model, self.pred_len, bias=True)

    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        # x_enc: [B, C, L]
        
        # Permute to [B, L, C]
        x_enc = x_enc.permute(0, 2, 1).contiguous()
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # print(f"x_enc: {x_enc.shape}, x_mark_enc: {x_mark_enc.shape}")
        # exit(0)

        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection [B, C, D] -> [B, C, pred_len]
        dec_out = self.projection(enc_out)
        
        # Permute to [B, pred_len, C] for denormalization
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]
        
        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # Return [B, C, pred_len]
        return dec_out.permute(0, 2, 1).contiguous()


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]



class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNetForecaster(DownstreamModelBase):

    def _build_model(self):
        cfg = self.configs
        args = self.downstream_args
        
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        
        self.top_k = args.get("top_k", getattr(cfg, "top_k", 3))
        self.d_model = args.get("d_model", getattr(cfg, "d_model", 16))
        self.d_ff = args.get("d_ff", getattr(cfg, "d_ff", 32))
        self.num_kernels = args.get("num_kernels", getattr(cfg, "num_kernels", 6))
        self.e_layers = args.get("e_layers", getattr(cfg, "e_layers", 2))
        self.enc_in = self.n_channels
        self.embed = args.get("embed", getattr(cfg, "embed", 'timeF'))
        self.freq = args.get("freq", getattr(cfg, "freq", 'h'))
        self.dropout = args.get("dropout", getattr(cfg, "dropout", 0.1))

        self.model = nn.ModuleList([TimesBlock(self)
                                    for _ in range(self.e_layers)])
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq,
                                           self.dropout)
        self.layer = self.e_layers
        self.layer_norm = nn.LayerNorm(self.d_model)

        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            self.d_model, cfg.n_channels, bias=True)
        

    def forward(self, x_enc, x_dec, x_mark_enc=None, x_mark_dec=None):
        
        # Permute to [B, L, C]
        x_enc = x_enc.permute(0, 2, 1).contiguous()

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # print(f"x_enc: {x_enc.shape}, x_mark_enc: {x_mark_enc.shape}")
        # exit(0)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        out = dec_out[:, -self.pred_len:, :]  # [B, L, D]
        return out.permute(0, 2, 1).contiguous()
    



AVAILABLE_LONG_TERM_FORECASTER = {
    # "RNN":LSTMClassifier,
    # "CNN":CNNClassifier,
    # "Transformer":TransformerClassifier,
    # "Transformer1":Transformer1Classifier,
    "RNN":RNNForecaster,
    "PatchTST":PatchTSTForecaster,
    "iTransformer":iTransformerForecaster,
    "TimesNet":TimesNetForecaster
}

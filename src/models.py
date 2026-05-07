"""
PyTorch model definitions.

Both models take encoded latent features [batch, window, latent_dim] as input.
The SDAE encoder is applied upstream (in windowing or training loop).

GRUForecaster:
  [B, T, latent] -> GRU(hidden=44, layers=2) -> last hidden -> MLP -> [B, 6]

MiniTransformerForecaster:
  [B, T, latent] -> Linear(latent->64) -> SinPE -> TransformerEncoder -> mean pool -> MLP -> [B, 6]
"""
import math
import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(in_dim // 2, out_dim),
        )
    def forward(self, x): return self.net(x)


class GRUForecaster(nn.Module):
    """Unidirectional GRU encoder -> MLP head. Input: [B, T, latent_dim]."""
    def __init__(self, n_features, hidden_dim=44, n_layers=2, dropout=0.1, n_outputs=6):
        super().__init__()
        self.gru  = nn.GRU(n_features, hidden_dim, n_layers,
                           batch_first=True, dropout=dropout if n_layers>1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = MLPHead(hidden_dim, n_outputs, dropout)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(self.drop(h_n[-1]))


class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])


class MiniTransformerForecaster(nn.Module):
    """Compact Transformer. Input: [B, T, latent_dim]."""
    def __init__(self, n_features, d_model=64, nhead=4, ff_dim=128,
                 n_layers=1, dropout=0.1, n_outputs=6):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe   = SinusoidalPE(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout,
                                               batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = MLPHead(d_model, n_outputs, dropout)

    def forward(self, x):
        x = self.pe(self.proj(x))
        return self.head(self.enc(x).mean(dim=1))


def build_model(arch: str, n_features: int, cfg) -> nn.Module:
    if arch == "gru":
        return GRUForecaster(n_features, cfg.GRU_HIDDEN, cfg.GRU_LAYERS, cfg.GRU_DROPOUT)
    elif arch == "transformer":
        return MiniTransformerForecaster(n_features, cfg.TRANSFORMER_D_MODEL,
                                         cfg.TRANSFORMER_NHEAD, cfg.TRANSFORMER_FF_DIM,
                                         cfg.TRANSFORMER_LAYERS, cfg.TRANSFORMER_DROPOUT)
    raise ValueError(f"Unknown arch: {arch!r}")

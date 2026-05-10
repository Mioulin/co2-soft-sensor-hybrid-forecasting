"""
Stacked Denoising Autoencoder (SDAE) for process variable compression.

Architecture (95 features -> 16 latent):
  Encoder: Linear(95->64) -> ReLU -> Dropout -> Linear(64->32) -> ReLU -> Dropout -> Linear(32->16)
  Decoder: Linear(16->32) -> ReLU -> Linear(32->64) -> ReLU -> Linear(64->95)

Input: 89 numeric process variables + 6 sampling-point one-hot indicators = 95 features.
During training: Gaussian noise injected at input (AWGN, std=0.1 after normalization).
This forces the encoder to learn robust, denoised representations.

Pretraining is unsupervised (reconstruction loss). The encoder is then
frozen and used as a feature extractor for GRU/Transformer forecasting.

Reference: Chai et al. (2026) - SDAE-GRU hybrid CO2 soft sensor.
"""
import torch
import torch.nn as nn
import numpy as np
from src.config import SDAE_HIDDEN_DIMS, SDAE_LATENT_DIM, SDAE_NOISE_STD, SDAE_DROPOUT


class SDAE(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dims: list = SDAE_HIDDEN_DIMS,
        latent_dim: int   = SDAE_LATENT_DIM,
        noise_std: float  = SDAE_NOISE_STD,
        dropout: float    = SDAE_DROPOUT,
    ):
        super().__init__()
        self.noise_std  = noise_std
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        in_dim = n_features
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        # Inject AWGN noise only during training
        if self.training and self.noise_std > 0:
            x_noisy = x + torch.randn_like(x) * self.noise_std
        else:
            x_noisy = x
        z     = self.encode(x_noisy)
        x_hat = self.decode(z)
        return x_hat, z


def train_sdae(
    model: SDAE,
    X_train: np.ndarray,
    X_val:   np.ndarray,
    checkpoint_path,
    epochs: int    = 150,
    lr: float      = 6.785e-4,
    batch_size: int = 32,
    device = None,
    verbose: bool  = True,
) -> dict:
    """Pretrain SDAE on reconstruction loss (unsupervised)."""
    from torch.utils.data import DataLoader, TensorDataset
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds_tr = TensorDataset(torch.from_numpy(X_train).float())
    ds_va = TensorDataset(torch.from_numpy(X_val).float())
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (xb,) in loader_tr:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in loader_va:
                xb = xb.to(device)
                x_hat, _ = model(xb)
                val_losses.append(criterion(x_hat, xb).item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), checkpoint_path)
        if verbose and epoch % 30 == 0:
            print(f"  SDAE epoch {epoch:>3d} | train={tl:.5f} val={vl:.5f}")

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return history


@torch.no_grad()
def encode_dataset(model: SDAE, X: np.ndarray, device=None, batch_size=256) -> np.ndarray:
    """Encode feature matrix X -> latent Z. Shape: (n, latent_dim)."""
    from torch.utils.data import DataLoader, TensorDataset
    if device is None:
        device = torch.device("cpu")
    model.eval().to(device)
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    zs = []
    for (xb,) in loader:
        zs.append(model.encode(xb.to(device)).cpu().numpy())
    return np.concatenate(zs)

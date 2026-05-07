"""Training loop with early stopping. Loss = full_MSE + alpha * obs_MSE."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src import config as cfg


def combined_loss(pred, target, mask, alpha=cfg.LOSS_ALPHA):
    full_mse = nn.functional.mse_loss(pred, target)
    obs = mask > 0.5
    obs_mse = ((pred[obs] - target[obs])**2).mean() if obs.sum() > 0 else torch.tensor(0.0, device=pred.device)
    return full_mse + alpha * obs_mse, full_mse, obs_mse


def make_loader(X, y, mask, kinetic, batch_size=cfg.BATCH_SIZE, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float(),
                       torch.from_numpy(mask).float(), torch.from_numpy(kinetic).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model, train_loader, val_loader, checkpoint_path, device=None, verbose=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.MAX_EPOCHS)
    history = {k: [] for k in ["train_loss","val_loss","train_full_mse","val_full_mse",
                                 "train_obs_mse","val_obs_mse"]}
    best_val, patience_ctr = float("inf"), 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        model.train(); tl, tf, to_ = [], [], []
        for Xb, yb, mb, _ in train_loader:
            Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            loss, fm, om = combined_loss(model(Xb), yb, mb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP); opt.step()
            tl.append(loss.item()); tf.append(fm.item()); to_.append(om.item())
        sched.step()

        model.eval(); vl, vf, vo = [], [], []
        with torch.no_grad():
            for Xb, yb, mb, _ in val_loader:
                Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
                loss, fm, om = combined_loss(model(Xb), yb, mb)
                vl.append(loss.item()); vf.append(fm.item()); vo.append(om.item())

        tl_m, vl_m = np.mean(tl), np.mean(vl)
        for k, v in zip(["train_loss","val_loss","train_full_mse","val_full_mse","train_obs_mse","val_obs_mse"],
                         [tl_m, vl_m, np.mean(tf), np.mean(vf), np.mean(to_), np.mean(vo)]):
            history[k].append(v)

        if vl_m < best_val - cfg.ES_MIN_DELTA:
            best_val = vl_m; patience_ctr = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_ctr += 1

        if verbose and epoch % 20 == 0:
            print(f"  ep{epoch:>3d} train={tl_m:.5f} val={vl_m:.5f} patience={patience_ctr}/{cfg.ES_PATIENCE}")
        if patience_ctr >= cfg.ES_PATIENCE:
            if verbose: print(f"  Early stop ep{epoch}")
            break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return history


@torch.no_grad()
def predict(model, loader, device=None):
    if device is None: device = torch.device("cpu")
    model.eval().to(device)
    preds, targets, masks = [], [], []
    for Xb, yb, mb, _ in loader:
        preds.append(model(Xb.to(device)).cpu().numpy())
        targets.append(yb.numpy()); masks.append(mb.numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)

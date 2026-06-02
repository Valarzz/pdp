"""
Unified PDP trajectory-encoder training. Domain-agnostic: all four RLBench domains use
this same script and differ only in their config (config/<domain>/encoder.yaml).

Run from the repo root:
    python train_encoder.py --config config/close_drawer/encoder.yaml
    # smoke test (few epochs):
    python train_encoder.py --config config/close_drawer/encoder.yaml --epochs 3
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from dataset import TrajectoryDataset
from model.encoder import TrajectoryVAE, TrajectoryVAENormalizedEE
from soft_dtw import SoftDTW
from trajectory_utils import (
    extract_endeffector_position, make_trajectory_relative, normalize_ee_trajectory,
    normalize_ee_trajectory_with_length,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def reconstruction_loss(pred_states, pred_actions, gt_states, gt_actions):
    return F.mse_loss(pred_actions, gt_actions) + F.mse_loss(pred_states, gt_states[:, 1:, :])


def kl_loss(z_mean, z_logvar):
    return (-0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)).mean()


def _geometry_dtw(z, geom, sdtw):
    """Align z-space L2 distances with soft-DTW distances between EE geometries `geom`
    (B,T,*). This is what makes the latent semantically structured; uses rolled batch pairs."""
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device, requires_grad=True)
    dz = torch.sqrt(torch.sum((z - torch.roll(z, 1, 0)) ** 2, dim=1) + 1e-8)
    traj_d = sdtw(geom, torch.roll(geom, 1, 0))
    dz = dz / (dz.max() + 1e-8)
    traj_d = traj_d / (traj_d.max() + 1e-8)
    return F.mse_loss(dz, traj_d)


def dtw_loss_full(z, states, sdtw):
    """Geometry loss from the FULL (raw relative) end-effector path. Default; used by the
    single-stage domains (CloseDrawer, PickUpCup)."""
    return _geometry_dtw(z, extract_endeffector_position(states, relative=True), sdtw)


def dtw_loss_ee(z, states, sdtw):
    """Geometry loss from the shape-NORMALIZED end-effector path ([progress, perp1, perp2]).
    Scale/endpoint-invariant; used by the two-stage domains (PlaceBlock, MeatOffGrill)."""
    return _geometry_dtw(z, normalize_ee_trajectory(states), sdtw)


DTW_LOSSES = {"full": dtw_loss_full, "ee": dtw_loss_ee}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_normalized_ee(cfg, epochs, batch_size, device, save_dir):
    """Encoder variant for the 2-stage tasks (place_block / meat_off_grill): encode the
    shape-normalized EE path; decode the full state+action trajectory from (z, s0, length).
    Trained on RAW (un-normalized) data, matching the research code."""
    enc, tr = cfg["encoder"], cfg["train"]
    dd = os.path.join(REPO_ROOT, cfg["data"]["dir"])
    d = np.load(os.path.join(dd, cfg["data"]["training"]))
    nz = np.load(os.path.join(dd, "normalizer.npz"))
    S, A, lens = d["states"], d["actions"], d["traj_lengths"]
    un_s = lambda x: (x + 1) / 2 * (nz["obs_max"] - nz["obs_min"]) + nz["obs_min"]
    un_a = lambda x: (x + 1) / 2 * (nz["action_max"] - nz["action_min"]) + nz["action_min"]
    H = cfg["horizon"]
    EE, S0, LEN, SR, AR = [], [], [], [], []
    i = 0
    for L in lens:
        L = int(L)
        if L == H:
            s = torch.tensor(un_s(S[i:i + L]), dtype=torch.float32).unsqueeze(0)  # raw (1,T,22)
            a = torch.tensor(un_a(A[i:i + L]), dtype=torch.float32).unsqueeze(0)  # raw (1,T,8)
            ee, length = normalize_ee_trajectory_with_length(s)
            sr, ar = make_trajectory_relative(s, a)
            EE.append(ee[0]); S0.append(s[0, 0]); LEN.append(length); SR.append(sr[0]); AR.append(ar[0])
        i += L
    ds = TensorDataset(torch.stack(EE), torch.stack(S0), torch.cat(LEN).unsqueeze(1),
                       torch.stack(SR), torch.stack(AR))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) >= batch_size)
    print(f"[{cfg['domain']}] normalized_ee: {len(ds)} trajectories (horizon {H}), device {device}")

    model_cfg = dict(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"], hidden_dim=enc["hidden_dim"],
                     num_layers=enc["num_layers"], num_heads=enc["num_heads"],
                     latent_dim=enc["latent_dim"], horizon=H)
    model = TrajectoryVAENormalizedEE(**model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sdtw = SoftDTW(gamma=tr.get("dtw_gamma", 1e-5), normalize=True)
    os.makedirs(save_dir, exist_ok=True)

    def save(tag):
        torch.save({"model_state_dict": model.state_dict(), "config": {**model_cfg, "type": "normalized_ee"}},
                   os.path.join(save_dir, f"{tag}.pt"))

    model.train()
    for epoch in range(1, epochs + 1):
        tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "dtw": 0.0}
        for ee, s0, length, sr, ar in loader:
            ee, s0, length, sr, ar = (t.to(device) for t in (ee, s0, length, sr, ar))
            out = model(ee, s0, length, sr, ar)
            recon = F.mse_loss(out["pred_actions"], ar) + F.mse_loss(out["pred_states"], sr[:, 1:])
            kl = kl_loss(out["z_mean"], out["z_logvar"])
            dtw = _geometry_dtw(out["z_mean"], ee, sdtw)
            loss = tr.get("recon_weight", 1.0) * recon + tr["kl_weight"] * kl + tr["dtw_weight"] * dtw
            opt.zero_grad(); loss.backward(); opt.step()
            for k, v in zip(tot, [loss, recon, kl, dtw]):
                tot[k] += v.item()
        n = len(loader)
        print(f"epoch {epoch:>4}/{epochs}  loss {tot['loss']/n:.4f}  recon {tot['recon']/n:.4f}  "
              f"kl {tot['kl']/n:.6f}  dtw {tot['dtw']/n:.4f}")
        if epoch % tr.get("save_freq", 500) == 0 or epoch == epochs:
            save(f"checkpoint_epoch_{epoch}"); save("best_model")
    print(f"saved normalized_ee encoder to {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override train.num_epochs")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    import random
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tr = cfg["train"]
    epochs = args.epochs or tr["num_epochs"]
    batch_size = args.batch_size or tr["batch_size"]
    device = args.device or tr.get("device", "cuda:0")
    save_dir = args.save_dir or os.path.join(REPO_ROOT, tr["save_dir"])
    if torch.cuda.is_available() is False:
        device = "cpu"

    if cfg["encoder"].get("type", "full") == "normalized_ee":
        train_normalized_ee(cfg, epochs, batch_size, device, save_dir)
        return

    npz = os.path.join(REPO_ROOT, cfg["data"]["dir"], cfg["data"]["training"])
    ds = TrajectoryDataset(npz, cfg["horizon"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) >= batch_size)
    print(f"[{cfg['domain']}] {len(ds)} trajectories (horizon {cfg['horizon']}), "
          f"state {ds.state_dim} action {ds.action_dim}, device {device}")

    enc = cfg["encoder"]
    model_cfg = dict(state_dim=cfg["state_dim"], action_dim=cfg["action_dim"],
                     hidden_dim=enc["hidden_dim"], num_layers=enc["num_layers"],
                     num_heads=enc["num_heads"], latent_dim=enc["latent_dim"], horizon=cfg["horizon"])
    model = TrajectoryVAE(**model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    sdtw = SoftDTW(gamma=tr.get("dtw_gamma", 1e-5), normalize=True)
    geometry = tr.get("geometry", "full")  # 'full' = raw EE path; 'ee' = shape-normalized EE
    if geometry not in DTW_LOSSES:
        raise ValueError(f"train.geometry must be one of {list(DTW_LOSSES)}, got {geometry!r}")
    dtw_fn = DTW_LOSSES[geometry]
    print(f"  geometry loss: {geometry}")

    ckpt_dir = save_dir  # e.g. results/<domain>/encoder
    os.makedirs(ckpt_dir, exist_ok=True)

    def save(tag):
        torch.save({"model_state_dict": model.state_dict(), "config": model_cfg},
                   os.path.join(ckpt_dir, f"{tag}.pt"))

    model.train()
    for epoch in range(1, epochs + 1):
        tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "dtw": 0.0}
        for states, actions in loader:
            states, actions = states.to(device), actions.to(device)
            states_rel, actions_rel = make_trajectory_relative(states, actions)
            out = model(states_rel, actions_rel)
            recon = reconstruction_loss(out["pred_states"], out["pred_actions"], states_rel, actions_rel)
            kl = kl_loss(out["z_mean"], out["z_logvar"])
            dtw = dtw_fn(out["z_mean"], states_rel, sdtw)
            loss = tr.get("recon_weight", 1.0) * recon + tr["kl_weight"] * kl + tr["dtw_weight"] * dtw
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k, v in zip(tot, [loss, recon, kl, dtw]):
                tot[k] += v.item()
        n = len(loader)
        print(f"epoch {epoch:>4}/{epochs}  loss {tot['loss']/n:.4f}  recon {tot['recon']/n:.4f}  "
              f"kl {tot['kl']/n:.6f}  dtw {tot['dtw']/n:.4f}")
        if epoch % tr.get("save_freq", 500) == 0 or epoch == epochs:
            save(f"checkpoint_epoch_{epoch}")
            save("best_model")
    print(f"saved checkpoints to {ckpt_dir}")


if __name__ == "__main__":
    main()

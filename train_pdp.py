"""
Unified PDP (parameterized diffusion policy) training. Domain-agnostic: all four RLBench
domains use this same script and differ only in their config (config/<domain>/pdp.yaml).

The trajectory encoder (trained by train_encoder.py) is frozen and used to precompute a
latent z for every training trajectory; the diffusion policy is then trained by behavior
cloning to denoise the action trajectory conditioned on (first observation, z).

Run from the repo root:
    python train_pdp.py --config config/close_drawer/pdp.yaml
    python train_pdp.py --config config/close_drawer/pdp.yaml --epochs 5   # smoke test
"""
import argparse
import copy
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from model.diffusion import FiLMDiffusionMLP, PDPDiffusion
from encoder_io import encode_demo_z, load_trajectory_encoder

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_z_dataset(data_dir, training_file, horizon, encoder, etype, device):
    """For each training trajectory: cond = first obs, target = action seq, z = encoder(traj).
    The z extraction dispatches on the encoder variant (see encoder_io.encode_demo_z)."""
    data = np.load(os.path.join(data_dir, training_file))
    states, actions, lengths = data["states"], data["actions"], data["traj_lengths"]
    S0, A, Z = [], [], []
    i = 0
    for L in lengths:
        L = int(L)
        if L == horizon:
            s = torch.from_numpy(states[i:i + L]).float().to(device)
            a = torch.from_numpy(actions[i:i + L]).float().to(device)
            z = encode_demo_z(encoder, etype, s, a, data_dir)   # (z_dim,)
            S0.append(s[0])                                      # (obs_dim,) first observation
            A.append(a)                                          # (T, action_dim)
            Z.append(z)
        i += L
    if not A:
        raise ValueError(f"No trajectories of length {horizon} in {data_dir}/{training_file}")
    return TensorDataset(torch.stack(A), torch.stack(S0), torch.stack(Z))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr, m = cfg["train"], cfg["model"]
    epochs = args.epochs or tr["num_epochs"]
    batch_size = args.batch_size or tr["batch_size"]
    device = args.device or tr.get("device", "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"
    save_dir = args.save_dir or os.path.join(REPO_ROOT, tr["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    encoder, etype = load_trajectory_encoder(os.path.join(REPO_ROOT, cfg["encoder_checkpoint"]), device)
    data_dir = os.path.join(REPO_ROOT, cfg["data"]["dir"])
    ds = build_z_dataset(data_dir, cfg["data"]["training"], cfg["horizon"], encoder, etype, device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=len(ds) >= batch_size)
    print(f"[{cfg['domain']}] {len(ds)} trajectories, horizon {cfg['horizon']}, "
          f"z_dim {m['z_dim']}, device {device}")

    net_cfg = dict(action_dim=cfg["action_dim"], horizon_steps=cfg["horizon"],
                   cond_dim=cfg["state_dim"], z_dim=m["z_dim"], time_dim=m["time_dim"],
                   mlp_dims=m["mlp_dims"], film_at_layers=m.get("film_at_layers", "all"))
    diff_cfg = dict(horizon_steps=cfg["horizon"], action_dim=cfg["action_dim"],
                    denoising_steps=m["denoising_steps"], use_ddim=m.get("use_ddim", True),
                    ddim_steps=m.get("ddim_steps", 8), device=device)
    model = PDPDiffusion(network=FiLMDiffusionMLP(**net_cfg), **diff_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"], weight_decay=tr.get("weight_decay", 1e-6))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=tr.get("min_lr", 1e-5))
    ema = EMA(model.network, tr.get("ema_decay", 0.995))

    save_cfg = {**net_cfg, **{k: diff_cfg[k] for k in ("denoising_steps", "use_ddim", "ddim_steps")}}

    def save(tag):
        torch.save({"model_state_dict": ema.shadow.state_dict(), "config": save_cfg},
                   os.path.join(save_dir, f"{tag}.pt"))

    best = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for actions, cond_state, z in loader:
            loss = model.loss(actions, {"state": cond_state}, z)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model.network)
            losses.append(loss.item())
        sched.step()
        avg = float(np.mean(losses))
        print(f"epoch {epoch:>5}/{epochs}  loss {avg:.5f}  lr {opt.param_groups[0]['lr']:.2e}")
        if avg < best:
            best = avg
            save("best_model")
        if epoch % tr.get("save_freq", 1000) == 0 or epoch == epochs:
            save("best_model")
    print(f"saved PDP policy to {save_dir}")


if __name__ == "__main__":
    main()

"""Shared trajectory-encoder loading + latent extraction, used by train_pdp.py and adapt.py.

The four domains use two encoder variants (see model/encoder.py):
  - "full"          (close_drawer, pick_up_cup): encodes the relative state+action path;
  - "normalized_ee" (place_block, meat_off_grill): encodes the shape-normalized EE path,
                     which is computed from the RAW (un-normalized) trajectory.
The saved checkpoint records which one in config["type"] ("full" if absent).
"""
import os

import numpy as np
import torch

from model.encoder import TrajectoryVAE, TrajectoryVAENormalizedEE
from trajectory_utils import make_trajectory_relative, normalize_ee_trajectory


def load_trajectory_encoder(ckpt_path, device):
    """Load the right encoder class for the checkpoint. Returns (encoder, encoder_type)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["config"])
    etype = cfg.pop("type", "full")
    cls = TrajectoryVAENormalizedEE if etype == "normalized_ee" else TrajectoryVAE
    enc = cls(**cfg).to(device).eval()
    enc.load_state_dict(ck["model_state_dict"])
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, etype


@torch.no_grad()
def encode_demo_z(encoder, etype, states_norm, actions_norm, data_dir):
    """Latent z (z_dim,) for one trajectory given its NORMALIZED states/actions ([-1,1]).

    full:          z = encoder(relative state+action path).
    normalized_ee: un-normalize states -> raw -> shape-normalized EE path -> z = encoder(ee).
    """
    if etype == "normalized_ee":
        nz = np.load(os.path.join(data_dir, "normalizer.npz"))
        raw = (states_norm.cpu().numpy() + 1) / 2 * (nz["obs_max"] - nz["obs_min"]) + nz["obs_min"]
        ee = normalize_ee_trajectory(
            torch.tensor(raw, dtype=torch.float32, device=states_norm.device).unsqueeze(0))
        return encoder.encode(ee)[0]
    s_rel, a_rel = make_trajectory_relative(states_norm.unsqueeze(0), actions_norm.unsqueeze(0))
    return encoder.encode(s_rel, a_rel)[0]

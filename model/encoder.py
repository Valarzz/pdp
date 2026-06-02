"""
Trajectory VAE that defines PDP's structured latent space.

Encodes a full (state, action) trajectory (in relative coordinates, so s0 = 0) into a
continuous latent z. Trained with reconstruction + KL + a DTW-geometry loss that aligns
distances in z-space with soft-DTW distances between end-effector paths -- this is what
makes z-space distances reflect physical-trajectory similarity, enabling behavior steering.

Shared across all four RLBench domains; only horizon / hyperparameters change (via config).
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):  # up to 250 steps x 2 tokens
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: (B, seq_len, d_model)
        return x + self.pe[: x.size(1), :].unsqueeze(0)


class TrajectoryEncoder(nn.Module):
    """(s0,a0,...,s_{T-1},a_{T-1}) -> z via a bidirectional transformer + reparameterization."""

    def __init__(self, state_dim=22, action_dim=8, hidden_dim=128, num_layers=4, num_heads=4, latent_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=500)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.to_latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, states, actions):
        B, T, _ = states.shape
        tokens = torch.stack([self.state_proj(states), self.action_proj(actions)], dim=2)
        tokens = tokens.reshape(B, T * 2, self.hidden_dim)
        encoded = self.transformer(self.pos_encoding(tokens))
        traj_repr = encoded[:, -1, :]  # last token = trajectory representation
        z_mean = self.to_latent_mean(traj_repr)
        z_logvar = self.to_latent_logvar(traj_repr)
        if self.training:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar)
        else:
            z = z_mean
        return z_mean, z_logvar, z


class TrajectoryDecoder(nn.Module):
    """z (+ per-step time) -> full trajectory in one shot (no teacher forcing; all info via z)."""

    def __init__(self, state_dim=22, action_dim=8, hidden_dim=128, num_layers=4, num_heads=4, latent_dim=2, horizon=4):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + action_dim),
        )

    def forward(self, z, states=None, actions=None):
        B = z.shape[0]
        t = torch.linspace(0, 1, self.horizon, device=z.device).unsqueeze(0).unsqueeze(2).expand(B, -1, -1)
        z_rep = z.unsqueeze(1).expand(-1, self.horizon, -1)
        out = self.mlp(torch.cat([z_rep, t], dim=-1))
        pred_states = out[:, 1:, : self.state_dim]    # s1..s_{T-1} (s0 is 0 in relative coords)
        pred_actions = out[:, :, self.state_dim:]     # a0..a_{T-1}
        return pred_states, pred_actions


class TrajectoryVAE(nn.Module):
    def __init__(self, state_dim=22, action_dim=8, hidden_dim=128, num_layers=4, num_heads=4, latent_dim=2, horizon=4):
        super().__init__()
        self.encoder = TrajectoryEncoder(state_dim, action_dim, hidden_dim, num_layers, num_heads, latent_dim)
        self.decoder = TrajectoryDecoder(state_dim, action_dim, hidden_dim, num_layers, num_heads, latent_dim, horizon)
        self.state_dim, self.action_dim, self.horizon = state_dim, action_dim, horizon

    def forward(self, states, actions):
        z_mean, z_logvar, z = self.encoder(states, actions)
        pred_states, pred_actions = self.decoder(z)
        return {"z_mean": z_mean, "z_logvar": z_logvar, "z": z,
                "pred_states": pred_states, "pred_actions": pred_actions}

    @torch.no_grad()
    def encode(self, states, actions):
        return self.encoder(states, actions)[0]  # z_mean


# ======================================================================================
# Normalized-EE variant (place_block / meat_off_grill, the two-stage tasks).
# The encoder ingests the shape-normalized EE path ([progress, perp1, perp2]) -- identical
# for reach and carry of the same control point -- and the decoder reconstructs the full
# state+action trajectory conditioned on (z, s0, trajectory_length). Same z-structuring
# idea, different I/O; ported to match the research checkpoints by key.
# ======================================================================================
class NormalizedEEEncoder(nn.Module):
    def __init__(self, ee_dim=3, hidden_dim=128, num_layers=4, num_heads=4, latent_dim=2):
        super().__init__()
        self.ee_proj = nn.Linear(ee_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=500)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.to_latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, ee_normalized):
        encoded = self.transformer(self.pos_encoding(self.ee_proj(ee_normalized)))
        traj_repr = encoded.mean(dim=1)  # mean-pool over time
        z_mean, z_logvar = self.to_latent_mean(traj_repr), self.to_latent_logvar(traj_repr)
        z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_logvar) if self.training else z_mean
        return z_mean, z_logvar, z


class FullTrajectoryDecoder(nn.Module):
    def __init__(self, state_dim=22, action_dim=8, hidden_dim=256, latent_dim=2, horizon=64):
        super().__init__()
        self.state_dim, self.horizon = state_dim, horizon
        input_dim = latent_dim + state_dim + 1 + 1  # z + s0 + length + time
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + action_dim))

    def forward(self, z, s0, trajectory_length):
        B = z.shape[0]
        t = torch.linspace(0, 1, self.horizon, device=z.device).unsqueeze(0).unsqueeze(2).expand(B, -1, -1)
        z_rep = z.unsqueeze(1).expand(-1, self.horizon, -1)
        s0_rep = s0.unsqueeze(1).expand(-1, self.horizon, -1)
        len_rep = trajectory_length.unsqueeze(1).expand(-1, self.horizon, -1)
        out = self.mlp(torch.cat([z_rep, s0_rep, len_rep, t], dim=-1))
        return out[:, 1:, :self.state_dim], out[:, :, self.state_dim:]


class TrajectoryVAENormalizedEE(nn.Module):
    def __init__(self, state_dim=22, action_dim=8, hidden_dim=128, num_layers=4, num_heads=4,
                 latent_dim=2, horizon=64):
        super().__init__()
        self.encoder = NormalizedEEEncoder(3, hidden_dim, num_layers, num_heads, latent_dim)
        self.decoder = FullTrajectoryDecoder(state_dim, action_dim, hidden_dim * 2, latent_dim, horizon)
        self.state_dim, self.action_dim, self.horizon = state_dim, action_dim, horizon

    def forward(self, ee_normalized, s0, trajectory_length, states_rel=None, actions_rel=None):
        z_mean, z_logvar, z = self.encoder(ee_normalized)
        pred_states, pred_actions = self.decoder(z, s0, trajectory_length)
        return {"z_mean": z_mean, "z_logvar": z_logvar, "z": z,
                "pred_states": pred_states, "pred_actions": pred_actions}

    @torch.no_grad()
    def encode(self, ee_normalized):
        return self.encoder(ee_normalized)[0]

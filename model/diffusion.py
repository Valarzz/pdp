"""
Parameterized Diffusion Policy (PDP): a DDPM action-trajectory model whose denoiser is
FiLM-conditioned on the trajectory latent z (from the trajectory encoder). Same model and
training across all four domains; configs only change horizon / net size.

Ported (and trimmed) from the DPPO DiffusionModel + FiLMDiffusionMLP.
"""
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- schedule helpers -----------------------------
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    ac = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.tensor(np.clip(betas, 0, 0.999), dtype=dtype)


def extract(a, t, x_shape):
    b = t.shape[0]
    return a.gather(-1, t).reshape(b, *((1,) * (len(x_shape) - 1)))


def make_timesteps(batch_size, i, device):
    return torch.full((batch_size,), i, device=device, dtype=torch.long)


class MLP(nn.Module):
    """Small MLP matching the research-code module naming (moduleList[i].linear_1 / act_1),
    so checkpoints trained with the original FiLMDiffusionMLP load by key directly.
    dims=[in,h1,...,out]; activation between layers, out_activation after the last."""

    def __init__(self, dims, activation=nn.Mish, out_activation=nn.Identity):
        super().__init__()
        self.moduleList = nn.ModuleList()
        n = len(dims) - 1
        for i in range(n):
            act = activation if i < n - 1 else out_activation
            self.moduleList.append(nn.Sequential(OrderedDict([
                ("linear_1", nn.Linear(dims[i], dims[i + 1])),
                ("act_1", act()),
            ])))

    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# ----------------------------- FiLM denoiser -----------------------------
class FiLMLayer(nn.Module):
    """Feature-wise linear modulation: x * (1 + scale(z)) + shift(z), zero-init = identity."""

    def __init__(self, feature_dim, conditioning_dim):
        super().__init__()
        self.scale_net = nn.Linear(conditioning_dim, feature_dim)
        self.shift_net = nn.Linear(conditioning_dim, feature_dim)
        for net in (self.scale_net, self.shift_net):
            nn.init.zeros_(net.weight)
            nn.init.zeros_(net.bias)

    def forward(self, x, z):
        return x * (1 + self.scale_net(z)) + self.shift_net(z)


class FiLMDiffusionMLP(nn.Module):
    """Diffusion denoiser; z conditions every block via FiLM and is also concatenated to the
    input. Predicts epsilon over the flattened action trajectory."""

    def __init__(self, action_dim, horizon_steps, cond_dim, z_dim, time_dim=16,
                 mlp_dims=(768, 768, 768, 768, 768), z_mlp_dims=(16, 32),
                 use_layernorm=True, residual_style=True, film_at_layers="all"):
        super().__init__()
        mlp_dims = list(mlp_dims)
        self.time_dim = time_dim
        self.z_dim = z_dim
        self.use_layernorm = use_layernorm
        self.residual_style = residual_style

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim), nn.Linear(time_dim, time_dim * 2),
            nn.Mish(), nn.Linear(time_dim * 2, time_dim),
        )
        if z_mlp_dims is not None:
            self.z_processor = MLP([z_dim] + list(z_mlp_dims))
            z_proc_dim = list(z_mlp_dims)[-1]
        else:
            self.z_processor = nn.Identity()
            z_proc_dim = z_dim

        input_dim = time_dim + action_dim * horizon_steps + cond_dim + z_proc_dim
        num_layers = len(mlp_dims)
        film_idx = list(range(num_layers)) if film_at_layers == "all" else (
            [num_layers // 2] if film_at_layers == "middle" else list(film_at_layers))

        self.input_proj = nn.Linear(input_dim, mlp_dims[0])
        self.blocks = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.film_layers = nn.ModuleDict()
        for i in range(num_layers):
            in_dim = mlp_dims[i - 1] if i > 0 else mlp_dims[0]
            self.blocks.append(nn.Linear(in_dim, mlp_dims[i]))
            if use_layernorm:
                self.layer_norms.append(nn.LayerNorm(mlp_dims[i]))
            if i in film_idx:
                self.film_layers[str(i)] = FiLMLayer(mlp_dims[i], z_proc_dim)
        self.output_proj = nn.Linear(mlp_dims[-1], action_dim * horizon_steps)
        self.activation = nn.Mish()
        self.register_buffer("z_empty", torch.zeros(1, z_dim))

    def forward(self, x, time, cond, z=None, **kwargs):
        B, Ta, Da = x.shape
        if z is None:
            z = self.z_empty.expand(B, -1)
        x = x.view(B, -1)
        state = cond["state"].view(B, -1)
        z_proc = self.z_processor(z)
        time_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        x = self.input_proj(torch.cat([x, time_emb, state, z_proc], dim=-1))
        for i, block in enumerate(self.blocks):
            residual = x
            x = block(x)
            if self.use_layernorm and i < len(self.layer_norms):
                x = self.layer_norms[i](x)
            if str(i) in self.film_layers:
                x = self.film_layers[str(i)](x, z_proc)
            if i < len(self.blocks) - 1:
                x = self.activation(x)
            if self.residual_style and residual.shape == x.shape:
                x = x + residual
        return self.output_proj(x).view(B, Ta, Da)


# ----------------------------- DDPM / DDIM wrapper -----------------------------
class PDPDiffusion(nn.Module):
    def __init__(self, network, horizon_steps, action_dim, denoising_steps=20,
                 predict_epsilon=True, denoised_clip_value=1.0, randn_clip_value=10,
                 use_ddim=True, ddim_steps=8, device="cuda:0"):
        super().__init__()
        self.device = device
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.denoising_steps = int(denoising_steps)
        self.predict_epsilon = predict_epsilon
        self.denoised_clip_value = denoised_clip_value
        self.randn_clip_value = randn_clip_value
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.network = network.to(device)

        betas = cosine_beta_schedule(self.denoising_steps).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, axis=0)
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
        self.alphas_cumprod = acp
        self.sqrt_alphas_cumprod = torch.sqrt(acp)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - acp)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / acp)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / acp - 1)
        self.ddpm_var = betas * (1.0 - acp_prev) / (1.0 - acp)
        self.ddpm_logvar_clipped = torch.log(torch.clamp(self.ddpm_var, min=1e-20))
        self.ddpm_mu_coef1 = betas * torch.sqrt(acp_prev) / (1.0 - acp)
        self.ddpm_mu_coef2 = (1.0 - acp_prev) * torch.sqrt(alphas) / (1.0 - acp)

        if use_ddim:
            step_ratio = self.denoising_steps // ddim_steps
            ddim_t = torch.arange(0, ddim_steps, device=device) * step_ratio
            self.ddim_alphas = acp[ddim_t].clone().float()
            self.ddim_alphas_prev = torch.cat([torch.ones(1, device=device).float(), acp[ddim_t[:-1]]])
            self.ddim_sqrt_one_minus_alphas = (1.0 - self.ddim_alphas) ** 0.5
            self.ddim_t = torch.flip(ddim_t, [0])
            self.ddim_alphas = torch.flip(self.ddim_alphas, [0])
            self.ddim_alphas_prev = torch.flip(self.ddim_alphas_prev, [0])
            self.ddim_sqrt_one_minus_alphas = torch.flip(self.ddim_sqrt_one_minus_alphas, [0])

    # ---- training ----
    def q_sample(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def loss(self, actions, cond, z):
        B = actions.shape[0]
        t = torch.randint(0, self.denoising_steps, (B,), device=actions.device).long()
        noise = torch.randn_like(actions)
        x_noisy = self.q_sample(actions, t, noise)
        pred = self.network(x_noisy, t, cond=cond, z=z)
        target = noise if self.predict_epsilon else actions
        return F.mse_loss(pred, target)

    # ---- sampling ----
    def _p_mean(self, x, t, index, cond, z):
        noise = self.network(x, t, cond=cond, z=z)
        if self.use_ddim:
            alpha = extract(self.ddim_alphas, index, x.shape)
            alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
            sqrt_1m_alpha = extract(self.ddim_sqrt_one_minus_alphas, index, x.shape)
            x_recon = (x - sqrt_1m_alpha * noise) / (alpha ** 0.5)
            if self.denoised_clip_value is not None:
                x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)
                noise = (x - alpha ** 0.5 * x_recon) / sqrt_1m_alpha
            mu = (alpha_prev ** 0.5) * x_recon + (1.0 - alpha_prev).sqrt() * noise
            return mu, torch.zeros_like(mu)
        x_recon = (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                   - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise)
        if self.denoised_clip_value is not None:
            x_recon = x_recon.clamp(-self.denoised_clip_value, self.denoised_clip_value)
        mu = (extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
              + extract(self.ddpm_mu_coef2, t, x.shape) * x)
        return mu, extract(self.ddpm_logvar_clipped, t, x.shape)

    @torch.no_grad()
    def sample(self, cond, z, x_init=None):
        """Generate an action trajectory conditioned on (state, z)."""
        B = cond["state"].shape[0]
        x = x_init if x_init is not None else torch.randn(
            (B, self.horizon_steps, self.action_dim), device=self.device)
        t_all = self.ddim_t if self.use_ddim else list(reversed(range(self.denoising_steps)))
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, int(t), self.device)
            mean, logvar = self._p_mean(x, t_b, make_timesteps(B, i, self.device), cond, z)
            if self.use_ddim or t == 0:
                std = torch.zeros_like(mean)
            else:
                std = torch.exp(0.5 * logvar).clip(min=1e-3)
            noise = torch.randn_like(x).clamp_(-self.randn_clip_value, self.randn_clip_value)
            x = mean + std * noise
        return x

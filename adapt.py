"""
Unified PDP test-time adaptation (FTL-IGM) + RLBench evaluation.

Given a single demonstration of a shifted scene, PDP adapts WITHOUT updating any policy
weights: it optimizes only the latent z so the frozen diffusion policy reproduces the demo,
then rolls the adapted policy out in RLBench and checks the domain constraint.

Steps (mirrors agent/finetune + agent/eval in the research code):
  1. warm-start z from the demo via the encoder, then optimize z by denoising-score-matching
     the frozen PDP policy against the demo actions (FTL-IGM);
  2. generate the adapted action trajectory with the optimized z;
  3. roll out in RLBench to get the executed end-effector path + task outcome;
  4. domain success = task outcome AND geometric constraint (env/constraints.py).

RLBench/CoppeliaSim is NOT bundled in this repo. Steps 1-2 run standalone; step 3 requires a
RLBench install + the task scenes -- see README ("RLBench setup") for how to build the engine
from the original RLBench repo. `make_rlbench_env` below is the single integration point.

Run from the repo root:
    python adapt.py --config config/close_drawer/eval.yaml            # uses adapt.demo from config
    python adapt.py --config config/close_drawer/eval.yaml --demo shiftdemo2 --style 2
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from model.diffusion import FiLMDiffusionMLP, PDPDiffusion
from encoder_io import encode_demo_z, load_trajectory_encoder
from env.constraints import constraint_satisfied

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_policy(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck["config"]
    net = FiLMDiffusionMLP(action_dim=c["action_dim"], horizon_steps=c["horizon_steps"],
                           cond_dim=c["cond_dim"], z_dim=c["z_dim"], time_dim=c["time_dim"],
                           mlp_dims=c["mlp_dims"], film_at_layers=c["film_at_layers"])
    diff = PDPDiffusion(network=net, horizon_steps=c["horizon_steps"], action_dim=c["action_dim"],
                        denoising_steps=c["denoising_steps"], use_ddim=c["use_ddim"],
                        ddim_steps=c["ddim_steps"], device=device)
    diff.network.load_state_dict(ck["model_state_dict"])
    diff.eval()
    for p in diff.parameters():
        p.requires_grad_(False)
    return diff, c


def ftl_igm(diff, z_init, demo_actions, cond_state, n_epochs, lr, n_timestep_samples):
    """Optimize z (only) so the frozen policy denoises the demo action trajectory."""
    device = demo_actions.device
    z = nn.Parameter(z_init.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([z], lr=lr)
    K = n_timestep_samples
    demo_K = demo_actions.expand(K, -1, -1)
    cond_K = {"state": cond_state.expand(K, -1)}
    for _ in range(n_epochs):
        opt.zero_grad()
        t = torch.randint(0, diff.denoising_steps, (K,), device=device).long()
        noise = torch.randn(K, demo_actions.shape[1], demo_actions.shape[2], device=device)
        x_noisy = diff.q_sample(demo_K, t, noise)
        with torch.enable_grad():
            pred = diff.network(x_noisy, t, cond=cond_K, z=z.unsqueeze(0).expand(K, -1))
        loss = F.mse_loss(pred, noise)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([z], 1.0)
        opt.step()
    return z.detach()


def make_rlbench_env(domain, style, cfg):
    """Build the RLBench task environment for `domain` at scene `style`.

    NOT implemented in this repo (RLBench/CoppeliaSim is an external dependency). Build it from
    the original RLBench repo per the README, then implement this to return an env exposing
    reset()/step(action) and the executed end-effector path + task-success flag, mirroring the
    research code's eval agents (agent/eval/eval_{close_drawer,pick_place,grasp}_agent.py)."""
    raise NotImplementedError(
        "RLBench rollout is an external dependency -- see README 'RLBench setup'. "
        "Implement make_rlbench_env() to return the task env for this domain/style.")


def rollout(diff, z, cond_state, domain, style, cfg):
    """Roll the adapted policy out in RLBench; return (ee_world (T,3), task_success)."""
    env = make_rlbench_env(domain, style, cfg)
    obs = env.reset()
    actions = diff.sample({"state": cond_state}, z).cpu().numpy()[0]  # (T, action_dim)
    ee_path, task_success = env.execute(actions)  # interface defined by the user's RLBench env
    return np.asarray(ee_path), bool(task_success)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--demo", default=None, help="override adapt.demo (shiftdemo1|2|3)")
    ap.add_argument("--style", type=int, default=None, help="scene style for the RLBench rollout")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_rollout", action="store_true", help="adapt + generate only (skip RLBench)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    a = cfg["adapt"]
    device = args.device or a.get("device", "cuda:0")
    if not torch.cuda.is_available():
        device = "cpu"
    domain = cfg["domain"]
    demo_name = args.demo or a["demo"]
    style = args.style if args.style is not None else int(demo_name.replace("shiftdemo", ""))

    encoder, etype = load_trajectory_encoder(os.path.join(REPO_ROOT, cfg["encoder_checkpoint"]), device)
    diff, _ = load_policy(os.path.join(REPO_ROOT, cfg["policy_checkpoint"]), device)

    data_dir = os.path.join(REPO_ROOT, cfg["data"]["dir"])
    demo = np.load(os.path.join(data_dir, f"{demo_name}.npz"))
    states = torch.from_numpy(demo["states"]).float().to(device)        # (T, obs_dim) normalized
    actions = torch.from_numpy(demo["actions"]).float().to(device)      # (T, action_dim) normalized
    H = cfg["horizon"]
    states, actions = states[:H], actions[:H]

    # 1. warm-start z from the demo (encoder variant-aware), then FTL-IGM
    z0 = encode_demo_z(encoder, etype, states, actions, data_dir)
    cond_state = states[0:1]  # first observation (1, obs_dim)
    z = ftl_igm(diff, z0, actions.unsqueeze(0), cond_state,
                a.get("ftl_igm_epochs", 300), a.get("lr", 3e-3), a.get("n_timestep_samples", 16))
    print(f"[{domain}/{demo_name}] z warm-start {z0.cpu().numpy()} -> optimized {z.cpu().numpy()}")

    # 2. generate adapted trajectory + report match to the demo
    gen = diff.sample({"state": cond_state}, z.unsqueeze(0))
    dtw_mse = F.mse_loss(gen[0], actions).item()
    print(f"  adapted-vs-demo action MSE: {dtw_mse:.5f}")

    if args.no_rollout:
        print("  (--no_rollout) skipping RLBench; adapted z + trajectory ready.")
        return

    # 3-4. RLBench rollout + domain success
    ee_world, task_success = rollout(diff, z.unsqueeze(0), cond_state, domain, style, cfg)
    ok, info = constraint_satisfied(domain, style, ee_world)
    success = task_success and ok
    print(f"  task_success={task_success}  constraint_ok={ok} {info}  =>  SUCCESS={success}")


if __name__ == "__main__":
    main()

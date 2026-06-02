"""
Loads a domain's normalized multimodal dataset (training.npz) as full trajectories.

training.npz: states (N, 22), actions (N, 8), traj_lengths (M,). Trajectories are
concatenated along axis 0; we split them back and keep those whose length == horizon
(all trajectories in the provided datasets share a fixed per-domain horizon).
"""
import numpy as np
import torch


class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, horizon):
        data = np.load(npz_path)
        states, actions, lengths = data["states"], data["actions"], data["traj_lengths"]
        self.state_dim = states.shape[1]
        self.action_dim = actions.shape[1]
        self.trajectories = []
        i = 0
        for L in lengths:
            L = int(L)
            if L == horizon:
                self.trajectories.append((
                    torch.from_numpy(states[i:i + L]).float(),
                    torch.from_numpy(actions[i:i + L]).float(),
                ))
            i += L
        if not self.trajectories:
            raise ValueError(f"No trajectories of length {horizon} in {npz_path} "
                             f"(found lengths {sorted(set(int(x) for x in lengths))})")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        s, a = self.trajectories[idx]
        return s, a

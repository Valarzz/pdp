"""
Shared trajectory transforms used by both encoder training and adaptation.

State layout (22-D, all four RLBench domains):
  joint_pos(7) | joint_vel(7) | gripper_open(1) | gripper_pose(7 = xyz + quat)
Action layout (8-D): joint_pos(7) | gripper_open(1)
End-effector xyz lives at state indices 15:18.
"""
import torch

EE_SLICE = slice(15, 18)


def make_trajectory_relative(states, actions):
    """Express a trajectory relative to its first frame (s0 -> 0), so the encoder learns
    trajectory *shape/mode* rather than absolute location. Joint positions/velocities and
    EE xyz are offset by their s0 values; gripper-open and quaternion are left as-is."""
    states_rel = states.clone()
    actions_rel = actions.clone()
    s0_joint_pos = states[:, 0:1, 0:7]
    s0_joint_vel = states[:, 0:1, 7:14]
    s0_ee_xyz = states[:, 0:1, 15:18]
    states_rel[:, :, 0:7] = states[:, :, 0:7] - s0_joint_pos
    states_rel[:, :, 7:14] = states[:, :, 7:14] - s0_joint_vel
    states_rel[:, :, 15:18] = states[:, :, 15:18] - s0_ee_xyz
    actions_rel[:, :, 0:7] = actions[:, :, 0:7] - s0_joint_pos
    return states_rel, actions_rel


def extract_endeffector_position(states, relative=False):
    """Return EE xyz (B, T, 3); if relative, offset so the first frame is the origin."""
    ee_pos = states[:, :, EE_SLICE]
    if relative:
        ee_pos = ee_pos - ee_pos[:, 0:1, :]
    return ee_pos


# --------------------------------------------------------------------------------------
# Normalized end-effector trajectory (used by the 'ee' geometry loss).
# Reparameterizes the EE path into a start->end trajectory frame: [progress, perp1, perp2].
# This is invariant to absolute start/end and overall scale, so it captures path *shape*
# -- helpful for the long-horizon, two-stage tasks (PlaceBlock / MeatOffGrill).
# --------------------------------------------------------------------------------------

def build_local_frame_batched(start_pos, end_pos):
    """Orthonormal trajectory frame (line_dir, perp1, perp2) + start->end distance."""
    line_vec = end_pos - start_pos
    dist = torch.clamp(torch.norm(line_vec, dim=1, keepdim=True), min=1e-6)
    line_dir = line_vec / dist

    world_up = torch.tensor([0.0, 0.0, 1.0], device=start_pos.device, dtype=start_pos.dtype)
    world_up = world_up.unsqueeze(0).expand(start_pos.shape[0], -1)
    dot = torch.sum(world_up * line_dir, dim=1, keepdim=True)
    perp1 = world_up - dot * line_dir
    perp1_len = torch.norm(perp1, dim=1, keepdim=True)

    nearly_vertical = perp1_len.squeeze(1) < 1e-6
    if nearly_vertical.any():  # fall back to world-forward when the path is ~vertical
        world_fwd = torch.tensor([0.0, 1.0, 0.0], device=start_pos.device, dtype=start_pos.dtype)
        world_fwd = world_fwd.unsqueeze(0).expand(start_pos.shape[0], -1)
        dot_fwd = torch.sum(world_fwd * line_dir, dim=1, keepdim=True)
        perp1_alt = world_fwd - dot_fwd * line_dir
        perp1_alt = perp1_alt / torch.clamp(torch.norm(perp1_alt, dim=1, keepdim=True), min=1e-6)
        perp1 = torch.where(nearly_vertical.unsqueeze(1), perp1_alt, perp1 / torch.clamp(perp1_len, min=1e-6))
    else:
        perp1 = perp1 / perp1_len

    perp2 = torch.cross(line_dir, perp1, dim=1)
    perp2 = perp2 / torch.norm(perp2, dim=1, keepdim=True).clamp(min=1e-6)
    return line_dir, perp1, perp2, dist.squeeze(1)


def normalize_ee_trajectory_with_length(states):
    """Map the EE path to trajectory-frame coords [progress, perp1_offset, perp2_offset]
    and also return the start->end distance (B,) used as the decoder's length conditioning."""
    ee_pos = states[:, :, EE_SLICE]
    start_pos, end_pos = ee_pos[:, 0, :], ee_pos[:, -1, :]
    line_dir, perp1, perp2, dist = build_local_frame_batched(start_pos, end_pos)
    dist_e = dist.unsqueeze(1).unsqueeze(2)
    vec = ee_pos - start_pos.unsqueeze(1)
    progress = torch.sum(vec * line_dir.unsqueeze(1), dim=2, keepdim=True) / dist_e
    point_on_line = start_pos.unsqueeze(1) + progress * (end_pos - start_pos).unsqueeze(1)
    offset = ee_pos - point_on_line
    perp1_off = torch.sum(offset * perp1.unsqueeze(1), dim=2, keepdim=True) / dist_e
    perp2_off = torch.sum(offset * perp2.unsqueeze(1), dim=2, keepdim=True) / dist_e
    return torch.cat([progress, perp1_off, perp2_off], dim=2), dist  # (B,T,3), (B,)


def normalize_ee_trajectory(states):
    """Map the EE path to trajectory-frame coords [progress, perp1_offset, perp2_offset]."""
    return normalize_ee_trajectory_with_length(states)[0]

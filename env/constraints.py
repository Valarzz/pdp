"""
Per-domain success/constraint dispatch over an end-effector trajectory (world coords).

Mirrors the constraint checks the paper's evaluation applies to the policy's rolled-out EE
path: wall collision for close_drawer / place_block / meat_off_grill, and approach-angle /
height validity for pick_up_cup. (Full task success also requires the RLBench task reward;
this module covers the geometric domain constraint.)
"""
import numpy as np

from env.close_drawer.wall_collision import WALL_STYLES as CD_WALL_STYLES
from env.close_drawer.wall_collision import check_trajectory_wall_collision as cd_check
from env.grasp.blocked_zone import BLOCKED_ZONE_STYLES, check_grasp_valid
from env.trajectory_wall.wall_config import check_trajectory_wall_collision as tw_check
from env.trajectory_wall.wall_config import get_wall_config

WALL_DOMAINS = ("close_drawer", "place_block", "meat_off_grill")


def constraint_satisfied(domain, style, ee_world, radius=0.02, grasp_height=None):
    """Return (ok, info). `ee_world`: (T,3) world-frame end-effector path.

    - close_drawer: world-coordinate wall (WALL_STYLES[style]); ok = collision-free.
    - place_block / meat_off_grill: trajectory-relative corner wall (get_wall_config);
      ok = collision-free (start/end taken from the path endpoints).
    - pick_up_cup: ok = grasp approach angle (+ height) valid for the blocked-zone style.
    """
    if domain == "close_drawer":
        collided, idx, _ = cd_check(np.asarray(ee_world), CD_WALL_STYLES[style])
        return (not collided), {"collision_idx": idx}
    if domain in ("place_block", "meat_off_grill"):
        task = "placeblock" if domain == "place_block" else "meatoffgrill"
        wc = get_wall_config(task, style)
        start, end = np.asarray(ee_world)[0], np.asarray(ee_world)[-1]
        collided, idx = tw_check(np.asarray(ee_world), start, end, radius, wc)
        return (not collided), {"collision_idx": idx}
    if domain == "pick_up_cup":
        angle = _approach_angle_from_ee(np.asarray(ee_world))
        h = grasp_height if grasp_height is not None else float(np.asarray(ee_world)[-1, 2])
        ok, reason = check_grasp_valid(angle, h, BLOCKED_ZONE_STYLES[style])
        return ok, {"approach_angle_deg": np.degrees(angle) % 360, "reason": reason}
    raise ValueError(f"unknown domain {domain!r}")


def _approach_angle_from_ee(ee_world):
    """Approach angle (rad) from the final descent direction toward the grasp, in the xy-plane."""
    v = ee_world[-1] - ee_world[max(0, len(ee_world) - 6)]
    return float(np.arctan2(v[1], v[0]))

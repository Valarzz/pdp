"""
Wall configuration for trajectory-relative collision detection.

This module provides wall collision detection that is relative to the start→end
trajectory of the task. The wall is defined perpendicular to the trajectory line,
positioned at a specific point along the trajectory (default: midpoint at pos_frac=0.5).

Wall parameters use the SAME coordinate system as control points:
  - angle: 0-360 degrees, where 0 deg is "up" relative to trajectory
  - distance: fraction of (radius * path_length), same as CP dist_frac

Supported tasks:
  - placeblock (stack_blocks): Pick up block and place it
  - meatoffgrill (meat_off_grill): Remove meat from grill
"""
import numpy as np


def build_local_frame(start_pos, target_pos):
    """
    Build an orthonormal frame from the trajectory direction.

    Args:
        start_pos: np.ndarray(3,), trajectory start position
        target_pos: np.ndarray(3,), trajectory target/end position

    Returns:
        line_vec_norm: unit vector along trajectory
        perp1: first perpendicular vector (up direction in local frame)
        perp2: second perpendicular vector (right direction in local frame)
    """
    line_vec = target_pos - start_pos
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    world_up = np.array([0.0, 0.0, 1.0])
    dot = np.dot(world_up, line_vec_norm)
    perp1 = world_up - dot * line_vec_norm
    perp1_len = np.linalg.norm(perp1)

    if perp1_len < 1e-6:
        world_forward = np.array([0.0, 1.0, 0.0])
        dot = np.dot(world_forward, line_vec_norm)
        perp1 = world_forward - dot * line_vec_norm
        perp1_len = np.linalg.norm(perp1)

    perp1 = perp1 / perp1_len
    perp2 = np.cross(line_vec_norm, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    return line_vec_norm, perp1, perp2


def compute_wall_corners(start_pos, end_pos, radius, wall_config, offset=None):
    """
    Compute the 4 corners of a wall in world coordinates.

    Args:
        start_pos: np.ndarray(3,), trajectory start
        end_pos: np.ndarray(3,), trajectory end
        radius: float, same radius used for control points
        wall_config: dict with wall parameters
        offset: [perp1_offset, perp2_offset] or None

    Returns:
        corners: np.ndarray(4, 3), four corners in world coordinates
    """
    line_vec = end_pos - start_pos
    path_length = np.linalg.norm(line_vec)
    line_vec_norm, perp1, perp2 = build_local_frame(start_pos, end_pos)

    pos_frac = wall_config.get("pos_frac", 0.5)
    corner_angle = wall_config.get("corner_angle", 0.0)
    corner_dist = wall_config.get("corner_dist", 0.0)
    width = wall_config.get("width", 1.0)
    height = wall_config.get("height", 1.0)

    offset_perp1 = 0.0
    offset_perp2 = 0.0
    if offset is not None:
        offset_perp1, offset_perp2 = offset

    base_pos = start_pos + pos_frac * line_vec

    angle_rad = np.deg2rad(corner_angle)
    corner_offset_dist = corner_dist * radius * path_length
    corner_offset = corner_offset_dist * (np.cos(angle_rad) * perp1 + np.sin(angle_rad) * perp2)
    offset_world = (offset_perp1 * radius * path_length) * perp1 + (offset_perp2 * radius * path_length) * perp2
    corner_pos = base_pos + corner_offset + offset_world

    wall_width = width * radius * path_length
    wall_height = height * radius * path_length

    corners = np.array([
        corner_pos,
        corner_pos + wall_height * perp2,
        corner_pos + wall_height * perp2 - wall_width * perp1,
        corner_pos - wall_width * perp1,
    ])

    return corners


def check_trajectory_wall_collision(ee_positions, start_pos, end_pos, radius, wall_config, offset=None):
    """
    Check if EE trajectory crosses through the wall.

    Args:
        ee_positions: np.ndarray(N, 3), EE trajectory positions
        start_pos: trajectory start position
        end_pos: trajectory end position
        radius: same radius used for control points
        wall_config: dict with wall parameters
        offset: [perp1_offset, perp2_offset] or None

    Returns:
        collision: bool, True if collision detected
        collision_idx: int or None, index where collision occurred
    """
    line_vec = end_pos - start_pos
    path_length = np.linalg.norm(line_vec)
    line_vec_norm, perp1, perp2 = build_local_frame(start_pos, end_pos)

    pos_frac = wall_config.get("pos_frac", 0.5)
    wall_center = start_pos + pos_frac * line_vec

    corner_angle = wall_config.get("corner_angle", 0.0)
    corner_dist = wall_config.get("corner_dist", 0.0)
    width = wall_config.get("width", 1.0)
    height = wall_config.get("height", 1.0)

    offset_perp1 = 0.0
    offset_perp2 = 0.0
    if offset is not None:
        offset_perp1, offset_perp2 = offset

    angle_rad = np.deg2rad(corner_angle)
    corner_perp1 = corner_dist * radius * np.cos(angle_rad) + offset_perp1 * radius
    corner_perp2 = corner_dist * radius * np.sin(angle_rad) + offset_perp2 * radius

    wall_width_world = width * radius * path_length
    wall_height_world = height * radius * path_length

    perp1_max = corner_perp1 * path_length
    perp1_min = perp1_max - wall_width_world
    perp2_min = corner_perp2 * path_length
    perp2_max = perp2_min + wall_height_world

    opening = wall_config.get("opening", None)
    has_opening = opening is not None

    if has_opening:
        dist_to_corner = opening.get("distance_to_corner", 0.0) * radius * path_length
        opening_length = opening.get("length", 0.0) * radius * path_length
        opening_perp1_max = perp1_max - dist_to_corner
        opening_perp1_min = opening_perp1_max - opening_length
        opening_perp2_min = perp2_min + dist_to_corner
        opening_perp2_max = opening_perp2_min + opening_length

    for i in range(1, len(ee_positions)):
        prev_pos = ee_positions[i - 1]
        curr_pos = ee_positions[i]

        prev_t = np.dot(prev_pos - start_pos, line_vec_norm) / path_length
        curr_t = np.dot(curr_pos - start_pos, line_vec_norm) / path_length

        if (prev_t < pos_frac and curr_t >= pos_frac) or (prev_t > pos_frac and curr_t <= pos_frac):
            if abs(curr_t - prev_t) > 1e-6:
                alpha = (pos_frac - prev_t) / (curr_t - prev_t)
                cross_pos = prev_pos + alpha * (curr_pos - prev_pos)
            else:
                cross_pos = curr_pos

            rel_pos = cross_pos - wall_center
            cross_perp1 = np.dot(rel_pos, perp1)
            cross_perp2 = np.dot(rel_pos, perp2)

            if perp1_min <= cross_perp1 <= perp1_max and perp2_min <= cross_perp2 <= perp2_max:
                if has_opening:
                    in_opening = (opening_perp1_min <= cross_perp1 <= opening_perp1_max and
                                  opening_perp2_min <= cross_perp2 <= opening_perp2_max)
                    if in_opening:
                        continue
                return True, i

    return False, None


def get_wall_config(task_name, style):
    """
    Get wall configuration for a specific task and style.

    Args:
        task_name: str, either "placeblock" or "meatoffgrill"
        style: int, wall style index (0, 1, 2, 3)

    Returns:
        dict with wall configuration parameters
    """
    if task_name == "placeblock":
        return PLACEBLOCK_WALL_STYLES.get(style, PLACEBLOCK_WALL_STYLES[0])
    elif task_name == "meatoffgrill":
        return MEATOFFGRILL_WALL_STYLES.get(style, MEATOFFGRILL_WALL_STYLES[0])
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ============================================================================
# Predefined Wall Configurations for PlaceBlock (stack_blocks)
# ============================================================================

PLACEBLOCK_DEFAULT_CONFIG = {
    "pos_frac": 0.5,
    "corner_angle": 315.0,
    "corner_dist": 0.5,
    "width": 0.5,
    "height": 0.5,
    "reach_offset": [-0.2, 0.2],
    "carry_offset": [-1.4, 0.1],
}

PLACEBLOCK_STYLE_1 = {
    "pos_frac": 0.5,
    "corner_angle": 330.0,
    "corner_dist": 1.4577,
    "width": 2.5,
    "height": 2.5,
    "reach_offset": [-0.2, 0.0],
    "carry_offset": [-1.3, -0.1],
}

PLACEBLOCK_STYLE_2 = {
    "pos_frac": 0.5,
    "corner_angle": 300.0,
    "corner_dist": 1.4577,
    "width": 2.5,
    "height": 2.5,
    "reach_offset": [-0.1, 0.0],
    "carry_offset": [-1.4, -0.1],
}

PLACEBLOCK_STYLE_3 = {
    "pos_frac": 0.5,
    "corner_angle": 315.0,
    "corner_dist": 1.7675,
    "width": 2.5,
    "height": 2.5,
    "opening": {
        "distance_to_corner": 1.5,
        "length": 1.0,
    },
    "reach_offset": [-0.1, 0.0],
    "carry_offset": [-1.5, 0.0],
}

PLACEBLOCK_WALL_STYLES = {
    0: PLACEBLOCK_DEFAULT_CONFIG,
    1: PLACEBLOCK_STYLE_1,
    2: PLACEBLOCK_STYLE_2,
    3: PLACEBLOCK_STYLE_3,
}


# ============================================================================
# Predefined Wall Configurations for MeatOffGrill (meat_off_grill)
# ============================================================================

MEATOFFGRILL_DEFAULT_CONFIG = {
    "pos_frac": 0.5,
    "corner_angle": 315.0,
    "corner_dist": 0.5,
    "width": 0.5,
    "height": 0.5,
    "reach_offset": [-0.2, 0.2],
    "carry_offset": [-1.4, 0.1],
}

MEATOFFGRILL_STYLE_1 = {
    "pos_frac": 0.5,
    "corner_angle": 330.0,
    "corner_dist": 1.4577,
    "width": 2.5,
    "height": 2.5,
    "reach_offset": [-0.2, 0.0],
    "carry_offset": [-1.3, -0.1],
}

MEATOFFGRILL_STYLE_2 = {
    "pos_frac": 0.5,
    "corner_angle": 300.0,
    "corner_dist": 1.4577,
    "width": 2.5,
    "height": 2.5,
    "reach_offset": [-0.1, 0.0],
    "carry_offset": [-1.4, -0.1],
}

MEATOFFGRILL_STYLE_3 = {
    "pos_frac": 0.5,
    "corner_angle": 315.0,
    "corner_dist": 1.7675,
    "width": 2.5,
    "height": 2.5,
    "opening": {
        "distance_to_corner": 1.5,
        "length": 1.0,
    },
    "reach_offset": [-0.1, 0.0],
    "carry_offset": [-1.5, 0.0],
}

MEATOFFGRILL_WALL_STYLES = {
    0: MEATOFFGRILL_DEFAULT_CONFIG,
    1: MEATOFFGRILL_STYLE_1,
    2: MEATOFFGRILL_STYLE_2,
    3: MEATOFFGRILL_STYLE_3,
}

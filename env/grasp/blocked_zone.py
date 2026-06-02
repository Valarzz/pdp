"""
Blocked zone definitions for grasp task evaluation.

Defines constraints on valid grasp configurations (approach_angle, grasp_height)
for different evaluation styles.

Style 1: Only 0 degree approach works (cup against wall)
Style 2: Only 180 degree approach works (opposite wall)
Style 3: Novel 45 degree approach (adaptation test)
"""
import numpy as np


# ============================================================================
# Evaluation Style Definitions
# ============================================================================

STYLE_1 = {
    "name": "single_angle_0",
    "description": "Only 0 degree approach works (cup against wall)",
    "valid_angles_deg": [0],
    "valid_heights": [0.12],
    "angle_tolerance_deg": 15.0,
    "height_tolerance": 0.02,
}

STYLE_2 = {
    "name": "single_angle_180",
    "description": "Only 180 degree approach works (opposite wall)",
    "valid_angles_deg": [180],
    "valid_heights": [0.12],
    "angle_tolerance_deg": 15.0,
    "height_tolerance": 0.02,
}

STYLE_3 = {
    "name": "novel_angle",
    "description": "Novel 45 degree approach - not in training (adaptation test)",
    "valid_angles_deg": [45],
    "valid_heights": [0.12],
    "angle_tolerance_deg": 10.0,
    "height_tolerance": 0.02,
}

BLOCKED_ZONE_STYLES = {
    1: STYLE_1,
    2: STYLE_2,
    3: STYLE_3,
}


# ============================================================================
# Validation Functions
# ============================================================================

def check_grasp_valid(approach_angle_rad, grasp_height, style_config):
    """
    Check if a grasp configuration is valid for a given blocked zone style.

    Args:
        approach_angle_rad: float, approach angle in radians
        grasp_height: float, grasp height in meters
        style_config: dict, style configuration

    Returns:
        is_valid: bool, True if configuration is valid
        reason: str, explanation of why invalid (if not valid)
    """
    approach_angle_deg = np.degrees(approach_angle_rad) % 360

    valid_angles_deg = style_config["valid_angles_deg"]
    valid_heights = style_config["valid_heights"]
    angle_tol = style_config.get("angle_tolerance_deg", 15.0)
    height_tol = style_config.get("height_tolerance", 0.008)

    angle_valid = False
    min_angle_diff = float('inf')
    for valid_angle in valid_angles_deg:
        diff = min(abs(approach_angle_deg - valid_angle),
                   360 - abs(approach_angle_deg - valid_angle))
        min_angle_diff = min(min_angle_diff, diff)
        if diff <= angle_tol:
            angle_valid = True
            break

    height_valid = False
    min_height_diff = float('inf')
    for valid_height in valid_heights:
        diff = abs(grasp_height - valid_height)
        min_height_diff = min(min_height_diff, diff)
        if diff <= height_tol:
            height_valid = True
            break

    if angle_valid and height_valid:
        return True, "Valid configuration"
    elif not angle_valid and not height_valid:
        return False, f"Invalid angle ({approach_angle_deg:.1f} deg) and height ({grasp_height:.4f}m)"
    elif not angle_valid:
        return False, f"Invalid angle: {approach_angle_deg:.1f} deg (tolerance={angle_tol} deg)"
    else:
        return False, f"Invalid height: {grasp_height:.4f}m (tolerance={height_tol}m)"


def get_valid_modes_for_style(canonical_params, style_config):
    """
    Find which training modes are valid for a given blocked zone style.

    Args:
        canonical_params: np.ndarray of (angle, height) pairs from training
        style_config: dict, style configuration

    Returns:
        valid_indices: list of valid mode indices
        valid_params: list of (angle, height) tuples that are valid
    """
    valid_indices = []
    valid_params = []

    for idx, (angle, height) in enumerate(canonical_params):
        is_valid, _ = check_grasp_valid(angle, height, style_config)
        if is_valid:
            valid_indices.append(idx)
            valid_params.append((angle, height))

    return valid_indices, valid_params


def get_target_config_for_style(style_config):
    """Get the target (ideal) configuration for a blocked zone style."""
    target_angle_deg = style_config["valid_angles_deg"][0]
    target_height = style_config["valid_heights"][0]
    return np.radians(target_angle_deg), target_height


def sample_valid_config_for_style(style_config, rng=None):
    """Sample a valid (angle, height) configuration for a blocked zone style."""
    if rng is None:
        rng = np.random.default_rng()

    valid_angles_deg = style_config["valid_angles_deg"]
    valid_heights = style_config["valid_heights"]
    angle_tol = style_config.get("angle_tolerance_deg", 15.0)
    height_tol = style_config.get("height_tolerance", 0.008)

    base_angle = rng.choice(valid_angles_deg)
    angle_noise = rng.uniform(-angle_tol * 0.5, angle_tol * 0.5)
    angle_deg = base_angle + angle_noise

    base_height = rng.choice(valid_heights)
    height_noise = rng.uniform(-height_tol * 0.5, height_tol * 0.5)
    height = base_height + height_noise

    return np.radians(angle_deg), height

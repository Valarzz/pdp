"""
Wall collision detection for close_drawer task.

The wall is a 2D plane (zero-width) positioned between the robot and drawer handle.
Trajectories that cause any part of the robot arm to cross this plane are rejected.
"""
import numpy as np

# ============================================================================
# Default Wall Configuration
# ============================================================================

DEFAULT_WALL_CONFIG = {
    "wall_y": -0.13,
    "wall_min_x": 0.30,
    "wall_max_x": 0.35,
    "wall_min_z": 1.15,
    "wall_max_z": 1.2,
    "wall_thickness": 0.002,
    "wall_color": [1.0, 0.2, 0.2],
    "wall_transparency": 0.6,
    "opening": None,
}

DEFAULT_OPENING_CONFIG = {
    "min_x": 0.2,
    "max_x": 0.3,
    "min_z": 1.2,
    "max_z": 1.3,
}

# ============================================================================
# Predefined Wall Styles
# ============================================================================

# Style 1: Only 0 degree could success (no opening)
WALL_STYLE_1 = DEFAULT_WALL_CONFIG.copy()
WALL_STYLE_1["wall_min_x"] = 0.25
WALL_STYLE_1["wall_max_x"] = 0.55
WALL_STYLE_1["wall_max_z"] = 1.3
WALL_STYLE_1["wall_min_z"] = 0.5
WALL_STYLE_1["opening"] = None

# Style 2: Only 270 degree could success (no opening)
WALL_STYLE_2 = DEFAULT_WALL_CONFIG.copy()
WALL_STYLE_2["wall_min_x"] = 0.2
WALL_STYLE_2["wall_max_x"] = 0.5
WALL_STYLE_2["wall_max_z"] = 1.23
WALL_STYLE_2["wall_min_z"] = 0.5
WALL_STYLE_2["opening"] = None

# Style 3: With opening enabled
WALL_STYLE_3 = DEFAULT_WALL_CONFIG.copy()
WALL_STYLE_3["wall_min_x"] = 0.2
WALL_STYLE_3["wall_max_x"] = 0.5
WALL_STYLE_3["wall_max_z"] = 1.3
WALL_STYLE_3["wall_min_z"] = 0.5
WALL_STYLE_3["opening"] = DEFAULT_OPENING_CONFIG.copy()

WALL_STYLES = {
    1: WALL_STYLE_1,
    2: WALL_STYLE_2,
    3: WALL_STYLE_3,
}


# ============================================================================
# Collision Detection
# ============================================================================

def check_wall_collision(task_env, wall_config=None):
    """
    Check if any robot arm link has crossed the wall plane.

    Args:
        task_env: RLBench task environment
        wall_config: dict with wall configuration

    Returns:
        collision: bool, True if collision detected
        collision_link: str or None, name of the link that collided
        collision_pos: np.ndarray or None, position of the colliding link
    """
    if wall_config is None:
        wall_config = DEFAULT_WALL_CONFIG.copy()

    wall_y = wall_config["wall_y"]
    min_x = wall_config["wall_min_x"]
    max_x = wall_config["wall_max_x"]
    min_z = wall_config["wall_min_z"]
    max_z = wall_config["wall_max_z"]
    opening = wall_config.get("opening", None)

    link_positions = get_robot_link_positions(task_env)

    for link_name, pos in link_positions:
        if not (min_x <= pos[0] <= max_x and min_z <= pos[2] <= max_z):
            continue

        if opening is not None:
            if (opening["min_x"] <= pos[0] <= opening["max_x"] and
                opening["min_z"] <= pos[2] <= opening["max_z"]):
                continue

        if pos[1] < wall_y:
            return True, link_name, pos

    return False, None, None


def get_robot_link_positions(task_env):
    """Get positions of all robot arm links and gripper."""
    robot = task_env._scene.robot
    arm = robot.arm
    gripper = robot.gripper

    link_positions = []

    for i, joint in enumerate(arm.joints):
        pos = np.array(joint.get_position())
        link_positions.append((f"joint_{i}", pos))

    tip = arm.get_tip()
    tip_pos = np.array(tip.get_position())
    link_positions.append(("tip", tip_pos))

    try:
        for i, gripper_joint in enumerate(gripper.joints):
            pos = np.array(gripper_joint.get_position())
            link_positions.append((f"gripper_{i}", pos))
    except Exception:
        pass

    return link_positions


def check_trajectory_wall_collision(ee_positions, wall_config=None):
    """
    Check if an end-effector trajectory crosses the wall.

    Args:
        ee_positions: list or array of EE positions (Nx3)
        wall_config: dict with wall configuration

    Returns:
        collision: bool, True if trajectory crosses wall
        collision_idx: int or None, index of first collision point
        collision_pos: np.ndarray or None, position at collision
    """
    if wall_config is None:
        wall_config = DEFAULT_WALL_CONFIG.copy()

    wall_y = wall_config["wall_y"]
    min_x = wall_config["wall_min_x"]
    max_x = wall_config["wall_max_x"]
    min_z = wall_config["wall_min_z"]
    max_z = wall_config["wall_max_z"]
    opening = wall_config.get("opening", None)

    for i, pos in enumerate(ee_positions):
        pos = np.array(pos)

        if not (min_x <= pos[0] <= max_x and min_z <= pos[2] <= max_z):
            continue

        if opening is not None:
            if (opening["min_x"] <= pos[0] <= opening["max_x"] and
                opening["min_z"] <= pos[2] <= opening["max_z"]):
                continue

        if pos[1] < wall_y:
            return True, i, pos

    return False, None, None


class WallCollisionTracker:
    """Tracker for wall collisions during trajectory execution."""

    def __init__(self, task_env, wall_config=None):
        self.task_env = task_env
        self.wall_config = wall_config if wall_config else DEFAULT_WALL_CONFIG.copy()
        self.reset()

    def reset(self):
        self.has_collision = False
        self.collision_step = None
        self.collision_link = None
        self.collision_pos = None
        self.current_step = 0

    def check_and_update(self):
        if self.has_collision:
            return True

        collision, link, pos = check_wall_collision(self.task_env, self.wall_config)

        if collision:
            self.has_collision = True
            self.collision_step = self.current_step
            self.collision_link = link
            self.collision_pos = pos

        self.current_step += 1
        return collision

    def get_collision_info(self):
        return {
            "has_collision": self.has_collision,
            "collision_step": self.collision_step,
            "collision_link": self.collision_link,
            "collision_pos": self.collision_pos.tolist() if self.collision_pos is not None else None,
        }

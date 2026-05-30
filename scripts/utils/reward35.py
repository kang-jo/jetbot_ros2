# utils/reward_3sonar5bin_aruco.py
#
# Reward function for SARSA with:
# - 3 ultrasonic sensors: front, left_1, right_1
# - 5-bin sonar discretization
# - 7-state ArUco camera state
#
# Designed for a corridor environment where the path can become narrow.
# The reward does NOT treat left/right close as always bad; if both sides are
# close but balanced, the robot can still be considered centered in a narrow corridor.

from typing import Dict, Tuple, Union


# ============================================================
# State definitions
# ============================================================

# Sonar state order:
#   sonar_state = (front, left_1, right_1)
#
# Sonar bins:
#   0 = danger      distance <= 0.20 m
#   1 = close       distance <= 0.35 m
#   2 = medium      distance <= 0.60 m
#   3 = clear       distance <= 1.00 m
#   4 = very clear  distance >  1.00 m

# Camera state:
CAM_NONE = 0
CAM_LEFT_FAR = 1
CAM_CENTER_FAR = 2
CAM_RIGHT_FAR = 3
CAM_LEFT_NEAR = 4
CAM_CENTER_NEAR = 5
CAM_RIGHT_NEAR = 6

CAMERA_STATE_NAMES = {
    CAM_NONE: "NO_TAG",
    CAM_LEFT_FAR: "LEFT_FAR",
    CAM_CENTER_FAR: "CENTER_FAR",
    CAM_RIGHT_FAR: "RIGHT_FAR",
    CAM_LEFT_NEAR: "LEFT_NEAR",
    CAM_CENTER_NEAR: "CENTER_NEAR",
    CAM_RIGHT_NEAR: "RIGHT_NEAR",
}

# Action mapping:
ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_LEFT_BIT = 3
ACTION_RIGHT_BIT = 4


# ============================================================
# Terminal rewards
# ============================================================

GOAL_REWARD = 300.0
CRASH_PENALTY = -180.0
TIMEOUT_PENALTY = -120.0


# ============================================================
# Basic movement reward
# ============================================================

STEP_REWARD = -0.10

# Encourage moving forward when the front path is safe.
FORWARD_SAFE_BONUS = 1.00

# Penalize large turns when unnecessary, but allow small corrections.
BIG_TURN_PENALTY = -0.35
SMALL_TURN_PENALTY = -0.05


# ============================================================
# Sonar safety reward
# ============================================================

FRONT_DANGER_PENALTY = -14.0
FRONT_CLOSE_PENALTY = -6.0
FRONT_CLEAR_BONUS = 0.40

SIDE_DANGER_PENALTY = -10.0
SIDE_CLOSE_PENALTY = -2.0

# Important for narrow corridors:
# left=1 and right=1 can mean the robot is centered in a narrow section.
NARROW_BALANCED_BONUS = 1.00

# Penalize large left/right imbalance, because that means the robot is too close
# to one side or angled poorly.
SIDE_IMBALANCE_PENALTY = -2.50

# Action-aware sonar penalties.
WRONG_SIDE_ACTION_PENALTY = -8.0
FORWARD_INTO_OBSTACLE_PENALTY = -8.0


# ============================================================
# Camera reward
# ============================================================

ARUCO_SIDE_FAR_BONUS = 0.40
ARUCO_SIDE_NEAR_BONUS = 0.80

ARUCO_CENTER_FAR_BONUS = 2.00
ARUCO_CENTER_NEAR_BONUS = 6.00
ARUCO_CENTER_NEAR_FRONT_OK_BONUS = 10.00

# Camera action shaping: make the policy more goal-locking.
CAMERA_CORRECT_ACTION_BONUS = 2.00
CAMERA_WRONG_ACTION_PENALTY = -3.00
CAMERA_FORWARD_WHEN_SIDE_PENALTY = -0.40
CAMERA_CENTER_FORWARD_BONUS = 2.00
CAMERA_CENTER_SMALL_TURN_BONUS = 0.40
CAMERA_CENTER_BIG_TURN_PENALTY = -1.50


RewardReturn = Union[Tuple[float, bool], Tuple[float, bool, Dict[str, float]]]


def _add_term(terms: Dict[str, float], name: str, value: float) -> None:
    """Store reward term only when value is non-zero."""
    if value != 0.0:
        terms[name] = terms.get(name, 0.0) + float(value)


def compute_reward(
    sonar_state,
    prev_action: int,
    camera_state: int = CAM_NONE,
    front_ok: bool = False,
    crash: bool = False,
    reached_goal: bool = False,
    timeout: bool = False,
    return_debug: bool = False,
) -> RewardReturn:
    """
    Compute reward for SARSA 3-sonar 5-bin + ArUco camera state.

    Parameters
    ----------
    sonar_state:
        Tuple/list of length 3: (front, left_1, right_1), each value 0..4.

    prev_action:
        Action that was executed before entering this state.
        0=forward, 1=turn left, 2=turn right, 3=left bit, 4=right bit.

    camera_state:
        One of CAM_NONE, CAM_LEFT_FAR, CAM_CENTER_FAR, CAM_RIGHT_FAR,
        CAM_LEFT_NEAR, CAM_CENTER_NEAR, CAM_RIGHT_NEAR.

    front_ok:
        True when front ultrasonic distance satisfies the goal front threshold.
        Usually used together with CAM_CENTER_NEAR.

    crash, reached_goal, timeout:
        Terminal flags.

    return_debug:
        If True, return (reward, done, terms). Otherwise return (reward, done).

    Notes
    -----
    Priority:
    1. Goal is rewarded immediately.
    2. Crash and timeout are terminal penalties.
    3. Sonar safety remains primary.
    4. Narrow but balanced corridor is not over-penalized.
    5. Camera state gives direction-locking reward when the robot is not in danger.
    """

    terms: Dict[str, float] = {}

    # --------------------------------------------------------
    # Terminal cases
    # --------------------------------------------------------
    if reached_goal:
        terms["terminal_goal"] = GOAL_REWARD
        if return_debug:
            return GOAL_REWARD, True, terms
        return GOAL_REWARD, True

    if crash:
        terms["terminal_crash"] = CRASH_PENALTY
        if return_debug:
            return CRASH_PENALTY, True, terms
        return CRASH_PENALTY, True

    if timeout:
        terms["terminal_timeout"] = TIMEOUT_PENALTY
        if return_debug:
            return TIMEOUT_PENALTY, True, terms
        return TIMEOUT_PENALTY, True

    if len(sonar_state) != 3:
        raise ValueError(f"sonar_state must have length 3, got {len(sonar_state)}: {sonar_state}")

    front, left, right = [int(v) for v in sonar_state]
    prev_action = int(prev_action)
    camera_state = int(camera_state)

    reward = STEP_REWARD
    _add_term(terms, "step", STEP_REWARD)

    # --------------------------------------------------------
    # 1. Front safety
    # --------------------------------------------------------
    if front == 0:
        reward += FRONT_DANGER_PENALTY
        _add_term(terms, "front_danger", FRONT_DANGER_PENALTY)
    elif front == 1:
        reward += FRONT_CLOSE_PENALTY
        _add_term(terms, "front_close", FRONT_CLOSE_PENALTY)
    elif front >= 3:
        reward += FRONT_CLEAR_BONUS
        _add_term(terms, "front_clear", FRONT_CLEAR_BONUS)

    # --------------------------------------------------------
    # 2. Side safety
    # --------------------------------------------------------
    if left == 0:
        reward += SIDE_DANGER_PENALTY
        _add_term(terms, "left_danger", SIDE_DANGER_PENALTY)
    elif left == 1:
        reward += SIDE_CLOSE_PENALTY
        _add_term(terms, "left_close", SIDE_CLOSE_PENALTY)

    if right == 0:
        reward += SIDE_DANGER_PENALTY
        _add_term(terms, "right_danger", SIDE_DANGER_PENALTY)
    elif right == 1:
        reward += SIDE_CLOSE_PENALTY
        _add_term(terms, "right_close", SIDE_CLOSE_PENALTY)

    # --------------------------------------------------------
    # 3. Corridor-friendly balance
    # --------------------------------------------------------
    # Narrow corridor case: both sides close but not danger.
    # This should not be treated as a bad state if the front is still safe.
    if left == 1 and right == 1 and front >= 2:
        reward += NARROW_BALANCED_BONUS
        _add_term(terms, "narrow_balanced", NARROW_BALANCED_BONUS)

    # Penalize strong imbalance.
    # Example: left=1, right=4 or left=4, right=1.
    if abs(left - right) >= 2:
        reward += SIDE_IMBALANCE_PENALTY
        _add_term(terms, "side_imbalance", SIDE_IMBALANCE_PENALTY)

    # --------------------------------------------------------
    # 4. Action-aware sonar shaping
    # --------------------------------------------------------
    if front <= 1 and prev_action == ACTION_FORWARD:
        reward += FORWARD_INTO_OBSTACLE_PENALTY
        _add_term(terms, "forward_into_obstacle", FORWARD_INTO_OBSTACLE_PENALTY)

    if left <= 1 and prev_action in (ACTION_TURN_LEFT, ACTION_LEFT_BIT):
        reward += WRONG_SIDE_ACTION_PENALTY
        _add_term(terms, "wrong_left_action", WRONG_SIDE_ACTION_PENALTY)

    if right <= 1 and prev_action in (ACTION_TURN_RIGHT, ACTION_RIGHT_BIT):
        reward += WRONG_SIDE_ACTION_PENALTY
        _add_term(terms, "wrong_right_action", WRONG_SIDE_ACTION_PENALTY)

    # --------------------------------------------------------
    # 5. Forward preference when generally safe
    # --------------------------------------------------------
    # If front is safe and neither side is in danger, prefer forward motion.
    if front >= 2 and left >= 1 and right >= 1:
        if prev_action == ACTION_FORWARD:
            reward += FORWARD_SAFE_BONUS
            _add_term(terms, "forward_safe", FORWARD_SAFE_BONUS)
        elif prev_action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
            reward += BIG_TURN_PENALTY
            _add_term(terms, "unneeded_big_turn", BIG_TURN_PENALTY)
        elif prev_action in (ACTION_LEFT_BIT, ACTION_RIGHT_BIT):
            reward += SMALL_TURN_PENALTY
            _add_term(terms, "small_turn", SMALL_TURN_PENALTY)

    # --------------------------------------------------------
    # 6. Camera base reward
    # --------------------------------------------------------
    # Camera reward is allowed when front is not close/danger and side sensors
    # are not in danger. left/right close is still allowed because the corridor
    # may be narrow.
    camera_allowed = front >= 2 and left >= 1 and right >= 1

    if camera_allowed:
        cam_base = 0.0

        if camera_state in (CAM_LEFT_FAR, CAM_RIGHT_FAR):
            cam_base = ARUCO_SIDE_FAR_BONUS
        elif camera_state in (CAM_LEFT_NEAR, CAM_RIGHT_NEAR):
            cam_base = ARUCO_SIDE_NEAR_BONUS
        elif camera_state == CAM_CENTER_FAR:
            cam_base = ARUCO_CENTER_FAR_BONUS
        elif camera_state == CAM_CENTER_NEAR:
            cam_base = ARUCO_CENTER_NEAR_FRONT_OK_BONUS if front_ok else ARUCO_CENTER_NEAR_BONUS

        reward += cam_base
        _add_term(terms, "camera_base", cam_base)

    # --------------------------------------------------------
    # 7. Camera action shaping
    # --------------------------------------------------------
    # This makes the robot more likely to lock onto the marker direction.
    if camera_allowed:
        cam_action = 0.0

        if camera_state in (CAM_LEFT_FAR, CAM_LEFT_NEAR):
            if prev_action in (ACTION_TURN_LEFT, ACTION_LEFT_BIT):
                cam_action += CAMERA_CORRECT_ACTION_BONUS
            elif prev_action in (ACTION_TURN_RIGHT, ACTION_RIGHT_BIT):
                cam_action += CAMERA_WRONG_ACTION_PENALTY
            elif prev_action == ACTION_FORWARD:
                cam_action += CAMERA_FORWARD_WHEN_SIDE_PENALTY

        elif camera_state in (CAM_RIGHT_FAR, CAM_RIGHT_NEAR):
            if prev_action in (ACTION_TURN_RIGHT, ACTION_RIGHT_BIT):
                cam_action += CAMERA_CORRECT_ACTION_BONUS
            elif prev_action in (ACTION_TURN_LEFT, ACTION_LEFT_BIT):
                cam_action += CAMERA_WRONG_ACTION_PENALTY
            elif prev_action == ACTION_FORWARD:
                cam_action += CAMERA_FORWARD_WHEN_SIDE_PENALTY

        elif camera_state in (CAM_CENTER_FAR, CAM_CENTER_NEAR):
            if prev_action == ACTION_FORWARD:
                cam_action += CAMERA_CENTER_FORWARD_BONUS
            elif prev_action in (ACTION_LEFT_BIT, ACTION_RIGHT_BIT):
                cam_action += CAMERA_CENTER_SMALL_TURN_BONUS
            elif prev_action in (ACTION_TURN_LEFT, ACTION_TURN_RIGHT):
                cam_action += CAMERA_CENTER_BIG_TURN_PENALTY

        reward += cam_action
        _add_term(terms, "camera_action", cam_action)

    if return_debug:
        return float(reward), False, terms

    return float(reward), False
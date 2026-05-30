from typing import Dict, Tuple, Union

# Sonar bins:
#   0 = danger      distance <= 0.20 m
#   1 = close       distance <= 0.35 m
#   2 = medium      distance <= 0.60 m
#   3 = clear       distance <= 1.00 m
#   4 = very clear  distance >  1.00 m

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

ACTION_FORWARD = 0
ACTION_TURN_LEFT = 1
ACTION_TURN_RIGHT = 2
ACTION_LEFT_BIT = 3
ACTION_RIGHT_BIT = 4

GOAL_REWARD = 300.0
CRASH_PENALTY = -180.0
TIMEOUT_PENALTY = -120.0

STEP_REWARD = -0.10

FORWARD_SAFE_BONUS = 1.00

BIG_TURN_PENALTY = -0.35
SMALL_TURN_PENALTY = -0.05

FRONT_DANGER_PENALTY = -14.0
FRONT_CLOSE_PENALTY = -6.0
FRONT_CLEAR_BONUS = 0.40

SIDE_DANGER_PENALTY = -10.0
SIDE_CLOSE_PENALTY = -2.0

NARROW_BALANCED_BONUS = 1.00

SIDE_IMBALANCE_PENALTY = -2.50

WRONG_SIDE_ACTION_PENALTY = -8.0
FORWARD_INTO_OBSTACLE_PENALTY = -8.0

ARUCO_SIDE_FAR_BONUS = 0.40
ARUCO_SIDE_NEAR_BONUS = 0.80

ARUCO_CENTER_FAR_BONUS = 2.00
ARUCO_CENTER_NEAR_BONUS = 6.00
ARUCO_CENTER_NEAR_FRONT_OK_BONUS = 10.00

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

    terms: Dict[str, float] = {}

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

    if front == 0:
        reward += FRONT_DANGER_PENALTY
        _add_term(terms, "front_danger", FRONT_DANGER_PENALTY)
    elif front == 1:
        reward += FRONT_CLOSE_PENALTY
        _add_term(terms, "front_close", FRONT_CLOSE_PENALTY)
    elif front >= 3:
        reward += FRONT_CLEAR_BONUS
        _add_term(terms, "front_clear", FRONT_CLEAR_BONUS)

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

    if left == 1 and right == 1 and front >= 2:
        reward += NARROW_BALANCED_BONUS
        _add_term(terms, "narrow_balanced", NARROW_BALANCED_BONUS)
    
    if abs(left - right) >= 2:
        reward += SIDE_IMBALANCE_PENALTY
        _add_term(terms, "side_imbalance", SIDE_IMBALANCE_PENALTY)

    if front <= 1 and prev_action == ACTION_FORWARD:
        reward += FORWARD_INTO_OBSTACLE_PENALTY
        _add_term(terms, "forward_into_obstacle", FORWARD_INTO_OBSTACLE_PENALTY)

    if left <= 1 and prev_action in (ACTION_TURN_LEFT, ACTION_LEFT_BIT):
        reward += WRONG_SIDE_ACTION_PENALTY
        _add_term(terms, "wrong_left_action", WRONG_SIDE_ACTION_PENALTY)

    if right <= 1 and prev_action in (ACTION_TURN_RIGHT, ACTION_RIGHT_BIT):
        reward += WRONG_SIDE_ACTION_PENALTY
        _add_term(terms, "wrong_right_action", WRONG_SIDE_ACTION_PENALTY)

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
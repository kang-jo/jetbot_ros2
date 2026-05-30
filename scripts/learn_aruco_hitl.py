#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning_node_3sonar5bin_aruco_hitl_sim.py

SARSA tabular untuk simulasi JetBot dengan dukungan HITL (Human-In-The-Loop) override action kapan saja.

State RL:
  - 3 ultrasonic: front, left_1, right_1
  - ultrasonic dibuat 5-bin agar kondisi miring/mepet lebih terbaca
  - 1 state kamera ArUco diskrit: no tag / left-center-right x far-near

Total state:
  ultrasonic: 5^3 = 125
  camera: 7
  total: 125 * 7 = 875

Action mengikuti utils/control.py:
  0 = forward
  1 = turn left wide
  2 = turn right wide
  3 = left bit
  4 = right bit
"""

import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Bool, Float32, Int32
from std_srvs.srv import Empty

from utils.spawn import declare_random_spawn_params, create_random_spawn_manager
from utils.aruco_detect import GoalArucoConfig, GoalDetectorAruco
from utils.control import do_action, stop
from utils.live_logger import LivePlotLogger
from utils.reward35 import GOAL_REWARD, CRASH_PENALTY, TIMEOUT_PENALTY, STEP_REWARD
from utils.sonar import MAX_SONAR, MIN_SONAR


# =============================
# Camera state definition
# =============================
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

ACTION_NAMES = {
    0: "FORWARD",
    1: "LEFT_WIDE",
    2: "RIGHT_WIDE",
    3: "LEFT_BIT",
    4: "RIGHT_BIT",
}


HITL_COMMAND_TO_ACTION = {
    "e": 0,  # forward
    "a": 1,  # left wide
    "f": 2,  # right wide
    "s": 3,  # left bit
    "d": 4,  # right bit
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}

SONAR_KEYS_3 = ["front", "left_1", "right_1"]
SONAR_LEVELS = 5
CAMERA_STATES = 7
N_SONAR_STATES = SONAR_LEVELS ** 3
N_TOTAL_STATES = N_SONAR_STATES * CAMERA_STATES

DEFAULT_Q_TABLE_PATH = "data/Q_table_35_aruco_sim.csv"


class Sonar5BinDiscretizer:
    """
    Discretizer 5-bin khusus 3 sonar.

    Default bin:
      0 = danger      d <= 0.20
      1 = close       0.20 < d <= 0.35
      2 = medium      0.35 < d <= 0.60
      3 = clear       0.60 < d <= 1.00
      4 = very clear  d > 1.00
    """

    def __init__(
        self,
        danger_m: float = 0.20,
        close_m: float = 0.35,
        medium_m: float = 0.60,
        clear_m: float = 1.00,
    ):
        self.danger_m = float(danger_m)
        self.close_m = float(close_m)
        self.medium_m = float(medium_m)
        self.clear_m = float(clear_m)

    def normalize(self, raw: Optional[float]) -> float:
        if raw is None:
            return MAX_SONAR
        return float(np.clip(float(raw), MIN_SONAR, MAX_SONAR))

    def discretize(self, raw: Optional[float]) -> int:
        d = self.normalize(raw)
        if d <= self.danger_m:
            return 0
        if d <= self.close_m:
            return 1
        if d <= self.medium_m:
            return 2
        if d <= self.clear_m:
            return 3
        return 4

    def process(self, raw_sonar: Dict[str, float]) -> Tuple[int, int, int]:
        return tuple(self.discretize(raw_sonar.get(k, MAX_SONAR)) for k in SONAR_KEYS_3)

    def thresholds_text(self) -> str:
        return (
            f"0<= {self.danger_m:.2f}, 1<= {self.close_m:.2f}, "
            f"2<= {self.medium_m:.2f}, 3<= {self.clear_m:.2f}, 4> {self.clear_m:.2f}"
        )


class TabularSARSAAgent:
    """SARSA tabular dengan jumlah state eksplisit."""

    def __init__(
        self,
        n_states: int,
        n_actions: int = 5,
        alpha: float = 0.12,
        gamma: float = 0.92,
        epsilon: float = 0.7,
        save_path: Optional[str] = None,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.save_path = save_path
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.visit_count = np.zeros(self.n_states, dtype=np.int64)

    def touch_state(self, state_idx: int) -> None:
        self.visit_count[int(state_idx)] += 1

    def choose_action_no_count(self, state_idx: int) -> int:
        state_idx = int(state_idx)
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.Q[state_idx, :]))

    def choose_action(self, state_idx: int) -> int:
        self.touch_state(state_idx)
        return self.choose_action_no_count(state_idx)

    def update(self, s: int, a: int, r: float, s2: int, a2: int) -> None:
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2, a2] - self.Q[s, a])

    def terminal_update(self, s: int, a: int, r: float) -> None:
        self.Q[s, a] += self.alpha * (r - self.Q[s, a])

    def save_qtable(self, path: Optional[str] = None) -> None:
        p = path if path is not None else self.save_path
        if not p:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pd.DataFrame(self.Q).to_csv(p, index=False, header=False)

    def load_qtable(self, path: Optional[str]) -> bool:
        if not path:
            return False
        try:
            q = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        except FileNotFoundError:
            return False
        except Exception:
            return False

        if q.shape != (self.n_states, self.n_actions):
            return False

        self.Q = q
        return True


class LearningNode3Sonar5BinAruco(Node):
    def __init__(self):
        super().__init__("sarsa_learning_node_3sonar5bin_aruco_sim")

        # =============================
        # ROS parameters
        # =============================
        self.declare_parameter("sensor_prefix", "/jetbotV21")
        self.declare_parameter("q_table_path", DEFAULT_Q_TABLE_PATH)
        declare_random_spawn_params(self)

        self.declare_parameter("hitl.enabled", True)
        self.declare_parameter("hitl.terminal_enabled", True)
        self.declare_parameter("hitl.action_topic", "/hitl_action")
        self.declare_parameter("hitl.manual_mode_topic", "/hitl_manual_mode")
        self.declare_parameter("hitl.imitation_enabled", True)
        self.declare_parameter("hitl.imitation_alpha", 0.25)
        self.declare_parameter("hitl.imitation_target_q", 4.0)
        self.declare_parameter("hitl.log_every_override", 1)

        self.declare_parameter("control_period", 0.2)
        self.declare_parameter("max_step", 240)
        self.declare_parameter("alpha", 0.12)
        self.declare_parameter("gamma", 0.92)
        self.declare_parameter("epsilon", 0.5)
        self.declare_parameter("epsilon_min", 0.05)
        self.declare_parameter("epsilon_decay", 0.995)
        self.declare_parameter("resume_epsilon_from_log", True)

        self.declare_parameter("goal_id", 23)
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("goal_front_threshold_m", 0.40)
        self.declare_parameter("goal_min_streak", 2)
        self.declare_parameter("aruco_near_area_ratio", 0.002)

        self.declare_parameter("camera_left_boundary_ratio", 0.35)
        self.declare_parameter("camera_right_boundary_ratio", 0.65)

        self.declare_parameter("front_crash_threshold_m", 0.25)
        self.declare_parameter("side_crash_threshold_m", 0.18)

        self.declare_parameter("sonar_bin_danger_m", 0.20)
        self.declare_parameter("sonar_bin_close_m", 0.35)
        self.declare_parameter("sonar_bin_medium_m", 0.60)
        self.declare_parameter("sonar_bin_clear_m", 1.00)

        self.declare_parameter("front_danger_penalty", -12.0)
        self.declare_parameter("front_close_penalty", -5.0)
        self.declare_parameter("front_clear_bonus", 0.3)
        self.declare_parameter("side_danger_penalty", -10.0)
        self.declare_parameter("side_close_penalty", -4.0)
        self.declare_parameter("center_good_bonus", 0.7)
        self.declare_parameter("center_ok_bonus", 0.3)
        self.declare_parameter("imbalance_penalty", -2.0)
        self.declare_parameter("wrong_side_action_penalty", -8.0)

        self.declare_parameter("disable_camera_bonus_when_sonar_close", True)

        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_every", 1)
        self.declare_parameter("camera_debug_log", True)
        self.declare_parameter("camera_debug_log_every", 15)
        self.declare_parameter("q_debug_log_every", 25)

        self.sensor_prefix = str(self.get_parameter("sensor_prefix").value)
        self.q_table_path = str(self.get_parameter("q_table_path").value)

        self.hitl_enabled = bool(self.get_parameter("hitl.enabled").value)
        self.hitl_terminal_enabled = bool(self.get_parameter("hitl.terminal_enabled").value)
        self.hitl_action_topic = str(self.get_parameter("hitl.action_topic").value)
        self.hitl_manual_mode_topic = str(self.get_parameter("hitl.manual_mode_topic").value)
        self.hitl_imitation_enabled = bool(self.get_parameter("hitl.imitation_enabled").value)
        self.hitl_imitation_alpha = float(self.get_parameter("hitl.imitation_alpha").value)
        self.hitl_imitation_target_q = float(self.get_parameter("hitl.imitation_target_q").value)
        self.hitl_log_every_override = max(1, int(self.get_parameter("hitl.log_every_override").value))

        self.control_period = float(self.get_parameter("control_period").value)
        self.max_step = int(self.get_parameter("max_step").value)
        self.epsilon_min = float(self.get_parameter("epsilon_min").value)
        self.epsilon_decay = float(self.get_parameter("epsilon_decay").value)

        self.goal_front_threshold_m = float(self.get_parameter("goal_front_threshold_m").value)
        self.goal_min_streak = int(self.get_parameter("goal_min_streak").value)
        self.aruco_near_area_ratio = float(self.get_parameter("aruco_near_area_ratio").value)
        self.camera_left_boundary_ratio = float(self.get_parameter("camera_left_boundary_ratio").value)
        self.camera_right_boundary_ratio = float(self.get_parameter("camera_right_boundary_ratio").value)
        self.front_crash_threshold_m = float(self.get_parameter("front_crash_threshold_m").value)
        self.side_crash_threshold_m = float(self.get_parameter("side_crash_threshold_m").value)

        self.front_danger_penalty = float(self.get_parameter("front_danger_penalty").value)
        self.front_close_penalty = float(self.get_parameter("front_close_penalty").value)
        self.front_clear_bonus = float(self.get_parameter("front_clear_bonus").value)
        self.side_danger_penalty = float(self.get_parameter("side_danger_penalty").value)
        self.side_close_penalty = float(self.get_parameter("side_close_penalty").value)
        self.center_good_bonus = float(self.get_parameter("center_good_bonus").value)
        self.center_ok_bonus = float(self.get_parameter("center_ok_bonus").value)
        self.imbalance_penalty = float(self.get_parameter("imbalance_penalty").value)
        self.wrong_side_action_penalty = float(self.get_parameter("wrong_side_action_penalty").value)
        self.disable_camera_bonus_when_sonar_close = bool(
            self.get_parameter("disable_camera_bonus_when_sonar_close").value
        )

        self.publish_debug_image_enabled = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_every = max(1, int(self.get_parameter("debug_image_every").value))
        self.camera_debug_log = bool(self.get_parameter("camera_debug_log").value)
        self.camera_debug_log_every = max(1, int(self.get_parameter("camera_debug_log_every").value))
        self.q_debug_log_every = max(1, int(self.get_parameter("q_debug_log_every").value))

        if self.camera_left_boundary_ratio >= self.camera_right_boundary_ratio:
            self.get_logger().warning(
                "camera_left_boundary_ratio >= camera_right_boundary_ratio, fallback to 0.35/0.65"
            )
            self.camera_left_boundary_ratio = 0.35
            self.camera_right_boundary_ratio = 0.65

        self.sonar_discretizer = Sonar5BinDiscretizer(
            danger_m=float(self.get_parameter("sonar_bin_danger_m").value),
            close_m=float(self.get_parameter("sonar_bin_close_m").value),
            medium_m=float(self.get_parameter("sonar_bin_medium_m").value),
            clear_m=float(self.get_parameter("sonar_bin_clear_m").value),
        )

        # =============================
        # Logger
        # =============================
        self.logger = LivePlotLogger(enable_plot=False)

        # =============================
        # ROS pubs/subs
        # =============================
        self.vel_pub = self.create_publisher(Twist, f"{self.sensor_prefix}/cmd_vel", 10)
        self.reward_pub = self.create_publisher(Float32, f"{self.sensor_prefix}/reward", 10)
        self.debug_img_pub = self.create_publisher(Image, "/aruco/debug_image", 10)
        self.spawn_cb_group = ReentrantCallbackGroup()
        self.reset_client = self.create_client(
            Empty,
            "reset_world",
            callback_group=self.spawn_cb_group,
        )
        self.spawn_manager = create_random_spawn_manager(
            self,
            callback_group=self.spawn_cb_group,
        )

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.image_fresh = False
        self.image_sub = self.create_subscription(
            Image,
            f"{self.sensor_prefix}/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.hitl_cmd_queue: "queue.Queue[str]" = queue.Queue()
        self.hitl_stop_event = threading.Event()
        self.hitl_manual_mode = False
        self.hitl_manual_action: Optional[int] = None
        self.hitl_one_shot_action: Optional[int] = None
        self.hitl_pending_actions: "queue.Queue[int]" = queue.Queue()
        self.hitl_last_source = "RL"
        self.hitl_override_step_counter = 0

        self.hitl_action_sub = self.create_subscription(
            Int32,
            self.hitl_action_topic,
            self.hitl_action_callback,
            10,
        )
        self.hitl_manual_mode_sub = self.create_subscription(
            Bool,
            self.hitl_manual_mode_topic,
            self.hitl_manual_mode_callback,
            10,
        )

        self.hitl_input_thread = None
        if self.hitl_enabled and self.hitl_terminal_enabled:
            self.hitl_input_thread = threading.Thread(target=self.hitl_terminal_loop, daemon=True)
            self.hitl_input_thread.start()

        self.sensor_topics = {
            "front": f"{self.sensor_prefix}/ultrasonic_front",
            "left_1": f"{self.sensor_prefix}/ultrasonic_left_1",
            "right_1": f"{self.sensor_prefix}/ultrasonic_right_1",
        }
        self.raw_sonar = {k: MAX_SONAR for k in self.sensor_topics.keys()}
        self.sensor_fresh = {k: False for k in self.sensor_topics.keys()}

        for key, topic in self.sensor_topics.items():
            self.create_subscription(
                Range,
                topic,
                lambda msg, k=key: self.sonar_cb(msg, k),
                qos_profile_sensor_data,
            )

        # =============================
        # ArUco detector
        # =============================
        self.goal_detector = GoalDetectorAruco(
            GoalArucoConfig(
                goal_id=int(self.get_parameter("goal_id").value),
                dictionary_name=str(self.get_parameter("aruco_dictionary").value),
                front_threshold_m=self.goal_front_threshold_m,
                min_streak=self.goal_min_streak,
                use_area_check=False,
                min_area_ratio=self.aruco_near_area_ratio,
                use_center_check=False,
                center_tolerance_ratio=0.20,
                debug=False,
            )
        )
        self.goal_streak = 0

        # =============================
        # RL setup
        # =============================
        self.n_sonar_states = N_SONAR_STATES
        self.n_camera_states = CAMERA_STATES
        self.n_states = N_TOTAL_STATES
        self.agent = TabularSARSAAgent(
            n_states=self.n_states,
            n_actions=5,
            alpha=float(self.get_parameter("alpha").value),
            gamma=float(self.get_parameter("gamma").value),
            epsilon=float(self.get_parameter("epsilon").value),
            save_path=self.q_table_path,
        )

        loaded = self.agent.load_qtable(self.q_table_path)
        if loaded:
            self.get_logger().warning(f"Loaded Q-table: {self.q_table_path}")
        else:
            self.get_logger().warning(
                f"Start with empty Q-table shape=({self.n_states}, 5). Path: {self.q_table_path}"
            )

        if bool(self.get_parameter("resume_epsilon_from_log").value):
            try:
                resume = self.logger.get_resume_state()
                if resume.get("has_history", False) and resume.get("last_epsilon") is not None:
                    self.agent.epsilon = float(resume["last_epsilon"])
                    self.get_logger().warning(
                        f"Resume epsilon from training log: {self.agent.epsilon:.3f} "
                        f"at logger episode {resume.get('last_episode')}"
                    )
            except Exception as exc:
                self.get_logger().warning(f"Could not resume epsilon from logger: {exc}")

        self.prev_state_idx: Optional[int] = None
        self.prev_action: Optional[int] = None
        self.episode = 0
        self.step_in_episode = 0
        self.hitl_count_episode = 0
        self.cumulated_reward = 0.0

        self.wait_after_reset = True
        self.reset_ready_time = time.time() + 1.0
        self.last_wait_warn_time = 0.0
        self.debug_counter = 0

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info("SARSA 3-sonar 5-bin + ArUco learning node initialized")
        self.get_logger().info(
            f"State size: sonar={self.n_sonar_states}, camera={self.n_camera_states}, total={self.n_states}; "
            f"epsilon={self.agent.epsilon:.3f}, max_step={self.max_step}"
        )
        self.get_logger().info(f"Sonar 5-bin thresholds: {self.sonar_discretizer.thresholds_text()}")
        self.get_logger().info(
            "Camera state: 0=NO_TAG, 1=LEFT_FAR, 2=CENTER_FAR, 3=RIGHT_FAR, "
            "4=LEFT_NEAR, 5=CENTER_NEAR, 6=RIGHT_NEAR"
        )
        self.get_logger().warning(
            f"HITL enabled={self.hitl_enabled}, terminal={self.hitl_terminal_enabled}, "
            f"action_topic={self.hitl_action_topic}, manual_mode_topic={self.hitl_manual_mode_topic}"
        )
        if self.hitl_enabled and self.hitl_terminal_enabled:
            self.get_logger().warning(
                "HITL keyboard (tanpa ENTER): e=forward, a=left_wide, f=right_wide, s=left_bit, d=right_bit, m=manual latch, r=back to RL, h=help"
            )

    # =============================
    # Callbacks
    # =============================
    def sonar_cb(self, msg: Range, key: str) -> None:
        try:
            d = float(msg.range)
        except Exception:
            return

        if d != d or d in (float("inf"), float("-inf")):
            self.get_logger().warning(f"Invalid sonar {key}: {d}")
            return

        self.raw_sonar[key] = float(max(min(d, MAX_SONAR), MIN_SONAR))
        self.sensor_fresh[key] = True

    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_fresh = True
        except Exception as exc:
            self.get_logger().warning(f"Image callback failed: {exc}")
            self.latest_frame = None
            self.image_fresh = False


    # =============================
    # HITL control
    # =============================
    def hitl_terminal_loop(self) -> None:
        """Read keyboard directly from terminal.

        Behavior:
        - If stdin is a TTY, use single-key capture (no Enter needed, no local echo).
        - If stdin is not a TTY, fall back to line mode.
        """
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setcbreak(fd)
                    while not self.hitl_stop_event.is_set():
                        ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                        if not ready:
                            continue
                        ch = sys.stdin.read(1)
                        if not ch:
                            continue
                        if ch in ("\n", "\r", "\x03", "\x04"):
                            continue
                        self.hitl_cmd_queue.put(ch.lower())
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                return
        except Exception as exc:
            self.get_logger().warning(f"HITL raw keyboard disabled, fallback to line mode: {exc}")

        while not self.hitl_stop_event.is_set():
            try:
                line = sys.stdin.readline()
                if line == "":
                    time.sleep(0.1)
                    continue
                for ch in line.strip().lower():
                    self.hitl_cmd_queue.put(ch)
            except Exception:
                time.sleep(0.1)

    def hitl_action_callback(self, msg: Int32) -> None:
        if not self.hitl_enabled:
            return
        action = int(msg.data)
        if 0 <= action <= 4:
            if self.hitl_manual_mode:
                self.hitl_manual_action = action
            else:
                self.hitl_pending_actions.put(action)

    def hitl_manual_mode_callback(self, msg: Bool) -> None:
        if not self.hitl_enabled:
            return
        self.hitl_manual_mode = bool(msg.data)
        if not self.hitl_manual_mode:
            self.hitl_manual_action = None
            while not self.hitl_pending_actions.empty():
                try:
                    self.hitl_pending_actions.get_nowait()
                except queue.Empty:
                    break

    def process_hitl_terminal_commands(self) -> None:
        if not self.hitl_enabled:
            return

        while True:
            try:
                cmd = self.hitl_cmd_queue.get_nowait()
            except queue.Empty:
                break

            if cmd in ("h", "?"):
                self.get_logger().warning(
                    "HITL keyboard (tanpa ENTER): e=forward, a=left_wide, f=right_wide, s=left_bit, d=right_bit, m=manual latch, r=back to RL"
                )
                continue

            if cmd in ("m",):
                self.hitl_manual_mode = True
                self.get_logger().warning("HITL manual mode ENABLED")
                continue

            if cmd in ("r", "x"):
                self.hitl_manual_mode = False
                self.hitl_manual_action = None
                self.hitl_one_shot_action = None
                while not self.hitl_pending_actions.empty():
                    try:
                        self.hitl_pending_actions.get_nowait()
                    except queue.Empty:
                        break
                self.get_logger().warning("HITL manual mode DISABLED -> back to RL")
                continue

            action = HITL_COMMAND_TO_ACTION.get(cmd)
            if action is None:
                self.get_logger().warning(f"Unknown HITL key: {cmd}")
                continue

            if self.hitl_manual_mode:
                self.hitl_manual_action = action
                self.get_logger().warning(
                    f"HITL latched manual action={action} {ACTION_NAMES.get(action)}"
                )
            else:
                self.hitl_pending_actions.put(action)
                self.get_logger().warning(
                    f"HITL queued one-shot action={action} {ACTION_NAMES.get(action)}"
                )

    def apply_hitl_imitation(self, state_idx: int, action: int) -> None:
        if not self.hitl_imitation_enabled:
            return
        self.agent.Q[state_idx, action] += self.hitl_imitation_alpha * (
            self.hitl_imitation_target_q - self.agent.Q[state_idx, action]
        )

    def choose_action_with_hitl(self, state_idx: int) -> Tuple[int, str]:
        self.agent.touch_state(state_idx)
        self.process_hitl_terminal_commands()

        if self.hitl_enabled:
            if self.hitl_manual_mode and self.hitl_manual_action is not None:
                action = int(self.hitl_manual_action)
                self.apply_hitl_imitation(state_idx, action)
                self.hitl_last_source = "HITL_MANUAL"
                self.hitl_count_episode += 1
                self.hitl_override_step_counter += 1
                return action, self.hitl_last_source

            try:
                action = int(self.hitl_pending_actions.get_nowait())
                self.apply_hitl_imitation(state_idx, action)
                self.hitl_last_source = "HITL_ONESHOT"
                self.hitl_count_episode += 1
                self.hitl_override_step_counter += 1
                return action, self.hitl_last_source
            except queue.Empty:
                pass

        action = self.agent.choose_action_no_count(state_idx)
        self.hitl_last_source = "RL"
        return action, self.hitl_last_source

    # =============================
    # State encoding
    # =============================
    def sonar_state_3(self) -> Tuple[int, int, int]:
        """Return discrete state order: (front, left_1, right_1), each in 0..4."""
        return self.sonar_discretizer.process(self.raw_sonar)

    @staticmethod
    def sonar_state_to_index(sonar_state: Tuple[int, int, int]) -> int:
        front, left_1, right_1 = sonar_state
        return int(front) * 25 + int(left_1) * 5 + int(right_1)

    @staticmethod
    def index_to_sonar_state(idx: int) -> Tuple[int, int, int]:
        front = idx // 25
        rem = idx % 25
        left_1 = rem // 5
        right_1 = rem % 5
        return int(front), int(left_1), int(right_1)

    def combined_state_to_index(self, sonar_state: Tuple[int, int, int], camera_state: int) -> int:
        sonar_idx = self.sonar_state_to_index(sonar_state)
        return int(sonar_idx * self.n_camera_states + int(camera_state))

    def index_to_combined_state(self, idx: int) -> Tuple[Tuple[int, int, int], int]:
        sonar_idx = int(idx) // self.n_camera_states
        camera_state = int(idx) % self.n_camera_states
        return self.index_to_sonar_state(sonar_idx), camera_state

    # =============================
    # Crash and reward
    # =============================
    def is_crash_3sonar(self) -> bool:
        front = float(self.raw_sonar.get("front", MAX_SONAR))
        left_1 = float(self.raw_sonar.get("left_1", MAX_SONAR))
        right_1 = float(self.raw_sonar.get("right_1", MAX_SONAR))
        return (
            front < self.front_crash_threshold_m
            or left_1 < self.side_crash_threshold_m
            or right_1 < self.side_crash_threshold_m
        )

    def compute_reward_3sonar(
        self,
        sonar_state: Tuple[int, int, int],
        prev_action: Optional[int],
        crash: bool = False,
        reached_goal: bool = False,
        timeout: bool = False,
    ) -> Tuple[float, bool, str]:
        """
        Reward 5-bin:
        - state 0 dihukum berat
        - state 1 tetap dihukum sebagai rawan
        - action yang mengarah ke sisi yang rawan diberi penalti tambahan
        """
        if reached_goal:
            return float(GOAL_REWARD), True, "GOAL"

        if crash:
            return float(CRASH_PENALTY), True, "CRASH"

        if timeout:
            return float(TIMEOUT_PENALTY), True, "TIMEOUT"

        front, left_1, right_1 = sonar_state
        r = float(STEP_REWARD)
        parts = [f"step={STEP_REWARD:.2f}"]

        if front == 0:
            r += self.front_danger_penalty
            parts.append(f"front_danger={self.front_danger_penalty:.1f}")
        elif front == 1:
            r += self.front_close_penalty
            parts.append(f"front_close={self.front_close_penalty:.1f}")
        elif front >= 3:
            r += self.front_clear_bonus
            parts.append(f"front_clear=+{self.front_clear_bonus:.1f}")

        if left_1 == 0:
            r += self.side_danger_penalty
            parts.append(f"left_danger={self.side_danger_penalty:.1f}")
        elif left_1 == 1:
            r += self.side_close_penalty
            parts.append(f"left_close={self.side_close_penalty:.1f}")

        if right_1 == 0:
            r += self.side_danger_penalty
            parts.append(f"right_danger={self.side_danger_penalty:.1f}")
        elif right_1 == 1:
            r += self.side_close_penalty
            parts.append(f"right_close={self.side_close_penalty:.1f}")

        if left_1 >= 2 and right_1 >= 2:
            diff = abs(left_1 - right_1)
            if diff == 0:
                r += self.center_good_bonus
                parts.append(f"center_good=+{self.center_good_bonus:.1f}")
            elif diff == 1:
                r += self.center_ok_bonus
                parts.append(f"center_ok=+{self.center_ok_bonus:.1f}")
            else:
                r += self.imbalance_penalty
                parts.append(f"imbalance={self.imbalance_penalty:.1f}")

        if prev_action is not None:
            wrong = False
            if front <= 1 and prev_action == 0:
                wrong = True
                parts.append("wrong_forward_when_front_close")
            if left_1 <= 1 and prev_action in (1, 3):
                wrong = True
                parts.append("wrong_left_when_left_close")
            if right_1 <= 1 and prev_action in (2, 4):
                wrong = True
                parts.append("wrong_right_when_right_close")
            if wrong:
                r += self.wrong_side_action_penalty
                parts.append(f"action_penalty={self.wrong_side_action_penalty:.1f}")

        return float(r), False, " ".join(parts)

    def compute_camera_bonus(
        self,
        camera_state: int,
        goal_dbg: Dict,
        sonar_state: Tuple[int, int, int],
        reached_goal: bool,
        crash: bool,
    ) -> float:
        if reached_goal or crash:
            return 0.0

        if camera_state == CAM_NONE:
            return 0.0

        front, left_1, right_1 = sonar_state
        if self.disable_camera_bonus_when_sonar_close and (front <= 1 or left_1 <= 1 or right_1 <= 1):
            return 0.0

        if camera_state == CAM_CENTER_NEAR:
            if bool(goal_dbg.get("front_ok", False)):
                return 5.0
            return 3.0

        if camera_state == CAM_CENTER_FAR:
            return 1.2

        if camera_state in (CAM_LEFT_NEAR, CAM_RIGHT_NEAR):
            return 0.6

        return 0.25

    # =============================
    # Camera and goal handling
    # =============================
    def classify_camera_state(self, info: Dict) -> int:
        if not info.get("goal_id_found", False):
            return CAM_NONE

        w = float(info.get("frame_width", 0.0))
        cx = float(info.get("center_x", -1.0))
        area_ratio = float(info.get("area_ratio", 0.0))
        if w <= 0.0 or cx < 0.0:
            return CAM_NONE

        x_ratio = cx / w
        is_near = area_ratio >= self.aruco_near_area_ratio

        if x_ratio < self.camera_left_boundary_ratio:
            return CAM_LEFT_NEAR if is_near else CAM_LEFT_FAR
        if x_ratio > self.camera_right_boundary_ratio:
            return CAM_RIGHT_NEAR if is_near else CAM_RIGHT_FAR
        return CAM_CENTER_NEAR if is_near else CAM_CENTER_FAR

    def read_camera_goal(self) -> Tuple[int, bool, Dict]:
        """Return (camera_state, reached_goal, debug_info)."""
        if self.latest_frame is None:
            self.goal_streak = 0
            return CAM_NONE, False, {
                "goal_id_found": False,
                "camera_state": CAM_NONE,
                "camera_state_name": CAMERA_STATE_NAMES[CAM_NONE],
                "streak": 0,
                "goal_reached": False,
            }

        try:
            found, info = self.goal_detector.detect_goal_marker(self.latest_frame)
        except Exception as exc:
            self.get_logger().warning(f"ArUco detection failed: {exc}")
            self.goal_streak = 0
            return CAM_NONE, False, {
                "goal_id_found": False,
                "camera_state": CAM_NONE,
                "camera_state_name": CAMERA_STATE_NAMES[CAM_NONE],
                "streak": 0,
                "goal_reached": False,
            }

        front_distance = float(self.raw_sonar.get("front", MAX_SONAR))
        info["front_distance_m"] = front_distance
        info["front_ok"] = 0.0 < front_distance <= self.goal_front_threshold_m

        camera_state = self.classify_camera_state(info)
        center_ok = camera_state in (CAM_CENTER_FAR, CAM_CENTER_NEAR)
        near_ok = camera_state in (CAM_LEFT_NEAR, CAM_CENTER_NEAR, CAM_RIGHT_NEAR)
        success_candidate = bool(camera_state == CAM_CENTER_NEAR and info["front_ok"])

        if success_candidate:
            self.goal_streak += 1
        else:
            self.goal_streak = 0

        reached_goal = self.goal_streak >= self.goal_min_streak

        info["goal_id_found"] = bool(found and info.get("goal_id_found", False))
        info["camera_state"] = camera_state
        info["camera_state_name"] = CAMERA_STATE_NAMES.get(camera_state, "UNKNOWN")
        info["center_ok"] = center_ok
        info["near_ok"] = near_ok
        info["success_candidate"] = success_candidate
        info["valid"] = success_candidate
        info["streak"] = self.goal_streak
        info["goal_reached"] = reached_goal

        if self.publish_debug_image_enabled:
            self.debug_counter += 1
            if self.debug_counter % self.debug_image_every == 0:
                self.publish_debug_image(info)

        return camera_state, reached_goal, info

    def publish_debug_image(self, info: Dict) -> None:
        if self.latest_frame is None:
            return

        try:
            vis = self.draw_camera_debug(self.latest_frame, info)
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_img_pub.publish(msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to publish debug image: {exc}")

    def draw_camera_debug(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        x_left = int(self.camera_left_boundary_ratio * w)
        x_right = int(self.camera_right_boundary_ratio * w)
        x_mid = int(0.5 * w)

        cv2.line(vis, (x_left, 0), (x_left, h), (255, 0, 0), 2)
        cv2.line(vis, (x_right, 0), (x_right, h), (255, 0, 0), 2)
        cv2.line(vis, (x_mid, 0), (x_mid, h), (255, 255, 0), 1)

        cv2.putText(vis, "LEFT", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(vis, "CENTER", (max(x_left + 10, 15), h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(vis, "RIGHT", (max(x_right + 10, 15), h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

        state_name = str(info.get("camera_state_name", CAMERA_STATE_NAMES[CAM_NONE]))
        line1 = (
            f"cam={info.get('camera_state', CAM_NONE)} {state_name} | "
            f"front={info.get('front_distance_m', 0.0):.2f}m th={self.goal_front_threshold_m:.2f} "
            f"front_ok={info.get('front_ok', False)} | "
            f"streak={info.get('streak', 0)}/{self.goal_min_streak} "
            f"goal={info.get('goal_reached', False)}"
        )
        cv2.putText(vis, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        sonar_state = self.sonar_state_3()
        line2 = (
            f"sonar5(front,left1,right1)={sonar_state} raw="
            f"({self.raw_sonar['front']:.2f},{self.raw_sonar['left_1']:.2f},{self.raw_sonar['right_1']:.2f})"
        )
        cv2.putText(vis, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

        line2b = f"bins: {self.sonar_discretizer.thresholds_text()}"
        cv2.putText(vis, line2b, (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if not info.get("goal_id_found", False):
            seen_ids = info.get("ids", [])
            cv2.putText(
                vis,
                f"goal_id not found | seen={seen_ids}",
                (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            return vis

        pts = info.get("corners", None)
        if pts is not None:
            pts_i = pts.astype(np.int32)
            cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)

        cx = int(info.get("center_x", 0))
        cy = int(info.get("center_y", 0))
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)

        area_ratio = float(info.get("area_ratio", 0.0))
        x_ratio = float(info.get("center_x", 0.0)) / float(w) if w > 0 else 0.0
        line3 = (
            f"id={info.get('target_id', '?')} x_ratio={x_ratio:.3f} "
            f"area={area_ratio:.4f} near_th={self.aruco_near_area_ratio:.4f} "
            f"center={info.get('center_ok', False)} near={info.get('near_ok', False)}"
        )
        cv2.putText(vis, line3, (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        return vis

    # =============================
    # Reset / readiness
    # =============================
    def hard_stop(self, repeat: int = 5, dt: float = 0.05) -> None:
        for _ in range(repeat):
            stop(self.vel_pub)
            time.sleep(dt)

    def begin_reset_wait(self) -> None:
        self.wait_after_reset = True
        self.reset_ready_time = time.time() + 0.8
        self.sensor_fresh = {k: False for k in self.sensor_topics.keys()}
        self.raw_sonar = {k: MAX_SONAR for k in self.sensor_topics.keys()}
        self.latest_frame = None
        self.image_fresh = False
        self.last_wait_warn_time = 0.0
        self.goal_streak = 0
        self.goal_detector.reset()

    def ready_after_reset(self) -> bool:
        if not self.wait_after_reset:
            return True

        now = time.time()
        if now < self.reset_ready_time:
            stop(self.vel_pub)
            return False

        missing_sensors = [k for k, fresh in self.sensor_fresh.items() if not fresh]
        missing_image = not self.image_fresh

        if missing_sensors or missing_image:
            stop(self.vel_pub)
            if now - self.last_wait_warn_time > 1.0:
                parts = []
                if missing_sensors:
                    parts.append(f"sensors={missing_sensors}")
                if missing_image:
                    parts.append("camera=waiting")
                self.get_logger().warning("Waiting fresh data after reset: " + ", ".join(parts))
                self.last_wait_warn_time = now
            return False

        self.wait_after_reset = False
        return True

    def reset_world(self) -> None:
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            self.reset_client.call_async(Empty.Request())

    def reset_episode_world(self, force_new_candidate: bool = False) -> None:
        """
        Wrapper for episode reset so the learning node stays clean.

        Behavior:
        - If random_spawn.enabled == True:
            uses GazeboRandomSpawnManager to:
            pause(optional) -> reset(optional) -> set chosen/random pose -> unpause(optional)
        - Otherwise:
            falls back to plain /reset_world
        """
        try:
            if getattr(self, "spawn_manager", None) is not None and self.spawn_manager.enabled:
                ok, candidate, changed = self.spawn_manager.reset_and_respawn(
                    force_new_candidate=force_new_candidate
                )
                if not ok:
                    self.get_logger().warning(
                        "[random_spawn] reset_and_respawn failed, fallback to plain /reset_world"
                    )
                    self.reset_world()
                return
        except Exception as exc:
            self.get_logger().warning(f"[random_spawn] reset wrapper failed: {exc}")

        self.reset_world()

    # =============================
    # Main RL loop
    # =============================
    def control_loop(self) -> None:
        try:
            self._control_loop_impl()
        except Exception as exc:
            self.get_logger().error(f"control_loop exception: {exc}")
            stop(self.vel_pub)

    def _control_loop_impl(self) -> None:
        if not self.ready_after_reset():
            return

        sonar_state = self.sonar_state_3()
        crash = self.is_crash_3sonar()

        if crash:
            camera_state = CAM_NONE
            reached_goal = False
            goal_dbg = {
                "goal_id_found": False,
                "camera_state": CAM_NONE,
                "camera_state_name": CAMERA_STATE_NAMES[CAM_NONE],
                "front_distance_m": float(self.raw_sonar.get("front", MAX_SONAR)),
                "front_ok": False,
                "area_ratio": 0.0,
                "streak": self.goal_streak,
                "goal_reached": False,
            }
        else:
            camera_state, reached_goal, goal_dbg = self.read_camera_goal()

        s_idx = self.combined_state_to_index(sonar_state, camera_state)

        if self.prev_state_idx is None:
            if crash:
                self.get_logger().warning("Spawn crash detected -> reset again, not counted as episode")
                self.hard_stop()
                try:
                    force_new = False
                    if getattr(self, "spawn_manager", None) is not None:
                        force_new = bool(self.spawn_manager.force_new_on_pre_episode_reset)
                    self.reset_episode_world(force_new_candidate=force_new)
                except Exception as exc:
                    self.get_logger().warning(f"Reset failed: {exc}")
                self.begin_reset_wait()
                return

            action, action_source = self.choose_action_with_hitl(s_idx)
            do_action(self.vel_pub, action)
            if action_source != "RL":
                self.get_logger().warning(
                    f"[HITL] first action override -> idx={s_idx} action={action} {ACTION_NAMES.get(action)} source={action_source}"
                )
            self.prev_state_idx = s_idx
            self.prev_action = action
            self.step_in_episode = 1
            return

        timeout = self.step_in_episode >= self.max_step
        evaluated_action = self.prev_action

        reward_value, done, reward_log = self.compute_reward_3sonar(
            sonar_state=sonar_state,
            prev_action=evaluated_action,
            crash=crash,
            reached_goal=reached_goal,
            timeout=timeout,
        )

        camera_bonus = self.compute_camera_bonus(
            camera_state=camera_state,
            goal_dbg=goal_dbg,
            sonar_state=sonar_state,
            reached_goal=reached_goal,
            crash=crash,
        )
        if not done:
            reward_value += camera_bonus

        self.cumulated_reward += reward_value

        if done:
            self.agent.terminal_update(self.prev_state_idx, self.prev_action, reward_value)
        else:
            next_action, action_source = self.choose_action_with_hitl(s_idx)
            self.agent.update(self.prev_state_idx, self.prev_action, reward_value, s_idx, next_action)
            do_action(self.vel_pub, next_action)

            if action_source != "RL" and (self.hitl_override_step_counter % self.hitl_log_every_override == 0):
                self.get_logger().warning(
                    f"[HITL] override -> step={self.step_in_episode} idx={s_idx} "
                    f"sonar={sonar_state} cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                    f"action={next_action} {ACTION_NAMES.get(next_action)} source={action_source}"
                )

            if self.step_in_episode % self.q_debug_log_every == 0:
                qrow = self.agent.Q[s_idx, :]
                self.get_logger().info(
                    f"[Q] step={self.step_in_episode} idx={s_idx} sonar={sonar_state} "
                    f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                    f"Q={[round(float(x), 3) for x in qrow.tolist()]} "
                    f"next_action={next_action} {ACTION_NAMES.get(next_action)}"
                )

            self.prev_state_idx = s_idx
            self.prev_action = next_action
            self.step_in_episode += 1

        rmsg = Float32()
        rmsg.data = float(reward_value)
        self.reward_pub.publish(rmsg)

        if camera_bonus > 0.0 and self.camera_debug_log:
            if self.step_in_episode % self.camera_debug_log_every == 0:
                self.get_logger().warning(
                    f"Camera bonus={camera_bonus:.2f} | "
                    f"state={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} | "
                    f"area={goal_dbg.get('area_ratio', 0.0):.4f} "
                    f"near_th={self.aruco_near_area_ratio:.4f} "
                    f"front={goal_dbg.get('front_distance_m', MAX_SONAR):.3f} "
                    f"front_th={self.goal_front_threshold_m:.3f} "
                    f"front_ok={goal_dbg.get('front_ok', False)} "
                    f"streak={goal_dbg.get('streak', 0)} | "
                    f"sonar5={sonar_state} raw=({self.raw_sonar['front']:.2f},"
                    f"{self.raw_sonar['left_1']:.2f},{self.raw_sonar['right_1']:.2f}) "
                    f"idx={s_idx}"
                )

        if self.step_in_episode % self.q_debug_log_every == 0 and not done:
            self.get_logger().info(
                f"[REWARD] r={reward_value:.2f} cam_bonus={camera_bonus:.2f} "
                f"evaluated_action={evaluated_action} {ACTION_NAMES.get(evaluated_action)} | {reward_log}"
            )

        if done:
            if timeout:
                self.get_logger().warning("Episode terminated cause Max Steps")
            self.finish_episode(reached_goal=reached_goal, crash=crash, timeout=timeout)

    def finish_episode(self, reached_goal: bool, crash: bool, timeout: bool) -> None:
        self.episode += 1

        if reached_goal:
            reason = "goal"
        elif crash:
            reason = "crash"
        elif timeout:
            reason = "timeout"
        else:
            reason = "done"

        visited = int(np.count_nonzero(self.agent.visit_count))
        self.get_logger().info(
            f"Episode {self.episode} finished, reason={reason}, "
            f"cum_reward={self.cumulated_reward:.2f}, steps={self.step_in_episode}, "
            f"epsilon={self.agent.epsilon:.3f}, visited_states={visited}/{self.n_states}"
        )

        self.hard_stop()

        total_msg = Float32()
        total_msg.data = float(self.cumulated_reward)
        self.reward_pub.publish(total_msg)

        try:
            self.reset_episode_world(force_new_candidate=False)
        except Exception as exc:
            self.get_logger().warning(f"Reset failed: {exc}")

        if self.agent.save_path:
            self.agent.save_qtable(self.agent.save_path)

        self.agent.epsilon = max(self.epsilon_min, self.agent.epsilon * self.epsilon_decay)

        try:
            self.logger.log_episode(
                episode=self.episode,
                total_reward=self.cumulated_reward,
                steps=self.step_in_episode,
                hitl_count=self.hitl_count_episode,
                success=reached_goal,
                epsilon=self.agent.epsilon,
            )
        except TypeError:
            self.logger.log_episode(
                self.episode,
                self.cumulated_reward,
                self.step_in_episode,
                self.hitl_count_episode,
                reached_goal,
                self.agent.epsilon,
            )
        except Exception as exc:
            self.get_logger().warning(f"Episode logger failed: {exc}")

        self.prev_state_idx = None
        self.prev_action = None
        self.hitl_one_shot_action = None
        self.cumulated_reward = 0.0
        self.step_in_episode = 0
        self.hitl_count_episode = 0
        self.goal_streak = 0
        self.goal_detector.reset()
        self.begin_reset_wait()


def main(args=None):
    rclpy.init(args=args)
    node = LearningNode3Sonar5BinAruco()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt — shutting down")
    finally:
        try:
            node.hitl_stop_event.set()
        except Exception:
            pass
        try:
            node.logger.close()
        except Exception:
            pass
        stop(node.vel_pub)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_node_3sonar5bin_aruco_exploit.py

Pure exploitation / evaluation node untuk model SARSA 3 ultrasonic 5-bin + ArUco.

Harus cocok dengan learning_node_3sonar5bin_aruco_sim.py:
  - sonar state: (front, left_1, right_1), masing-masing 0..4
  - camera state: 0..6
  - combined index: sonar_idx * 7 + camera_state
  - Q-table shape: 875 x 5

Action mengikuti utils/control.py:
  0 = forward
  1 = turn left wide
  2 = turn right wide
  3 = left bit
  4 = right bit
"""

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Range

from utils.aruco_detect import GoalArucoConfig, GoalDetectorAruco
from utils.control import do_action, stop
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

SONAR_KEYS_3 = ["front", "left_1", "right_1"]
SONAR_LEVELS = 5
CAMERA_STATES = 7
N_SONAR_STATES = SONAR_LEVELS ** 3
N_TOTAL_STATES = N_SONAR_STATES * CAMERA_STATES

DEFAULT_Q_TABLE_PATH = "data/Q_table_3sonar5bin_randomize_sim.csv"


class Sonar5BinDiscretizer:
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


class QTablePolicy:
    """Greedy policy dari Q-table dengan jumlah state eksplisit."""

    def __init__(self, n_states: int = N_TOTAL_STATES, n_actions: int = 5):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.loaded = False

    def load(self, path: str) -> bool:
        try:
            q = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        except FileNotFoundError:
            return False
        except Exception:
            return False

        if q.shape != (self.n_states, self.n_actions):
            self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
            self.loaded = False
            return False

        self.Q = q
        self.loaded = True
        return True

    def qrow(self, state_idx: int) -> np.ndarray:
        return self.Q[int(state_idx), :]

    def is_untrained(self, state_idx: int) -> bool:
        return bool(np.allclose(self.qrow(state_idx), 0.0))

    def choose_greedy(self, state_idx: int) -> int:
        return int(np.argmax(self.qrow(state_idx)))


class ControlNode3Sonar5BinArucoExploit(Node):
    def __init__(self):
        super().__init__("sarsa_control_node_3sonar5bin_aruco_exploit")

        # =============================
        # ROS parameters
        # =============================
        self.declare_parameter("sensor_prefix", "/jetbotV21")
        self.declare_parameter("q_table_path", DEFAULT_Q_TABLE_PATH)
        self.declare_parameter("control_period", 0.1)

        self.declare_parameter("goal_id", 23)
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("goal_front_threshold_m", 0.40)
        self.declare_parameter("goal_min_streak", 2)
        self.declare_parameter("aruco_near_area_ratio", 0.002)

        self.declare_parameter("camera_left_boundary_ratio", 0.35)
        self.declare_parameter("camera_right_boundary_ratio", 0.65)

        self.declare_parameter("front_crash_threshold_m", 0.25)
        self.declare_parameter("side_crash_threshold_m", 0.10)

        self.declare_parameter("sonar_bin_danger_m", 0.20)
        self.declare_parameter("sonar_bin_close_m", 0.35)
        self.declare_parameter("sonar_bin_medium_m", 0.60)
        self.declare_parameter("sonar_bin_clear_m", 1.00)

        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_image_every", 1)
        self.declare_parameter("log_every", 1)

        # Penting untuk evaluasi:
        # Kalau Q-row semua nol, jangan otomatis maju karena argmax([0,0,0,0,0]) = 0.
        self.declare_parameter("stop_on_untrained_state", False)
        self.declare_parameter("stop_if_qtable_invalid", True)
        self.declare_parameter("max_control_steps", 0)

        self.sensor_prefix = str(self.get_parameter("sensor_prefix").value)
        self.q_table_path = str(self.get_parameter("q_table_path").value)
        self.control_period = float(self.get_parameter("control_period").value)

        self.goal_front_threshold_m = float(self.get_parameter("goal_front_threshold_m").value)
        self.goal_min_streak = int(self.get_parameter("goal_min_streak").value)
        self.aruco_near_area_ratio = float(self.get_parameter("aruco_near_area_ratio").value)
        self.camera_left_boundary_ratio = float(self.get_parameter("camera_left_boundary_ratio").value)
        self.camera_right_boundary_ratio = float(self.get_parameter("camera_right_boundary_ratio").value)
        self.front_crash_threshold_m = float(self.get_parameter("front_crash_threshold_m").value)
        self.side_crash_threshold_m = float(self.get_parameter("side_crash_threshold_m").value)
        self.publish_debug_image_enabled = bool(self.get_parameter("publish_debug_image").value)
        self.debug_image_every = max(1, int(self.get_parameter("debug_image_every").value))
        self.log_every = max(1, int(self.get_parameter("log_every").value))
        self.stop_on_untrained_state = bool(self.get_parameter("stop_on_untrained_state").value)
        self.stop_if_qtable_invalid = bool(self.get_parameter("stop_if_qtable_invalid").value)
        self.max_control_steps = int(self.get_parameter("max_control_steps").value)

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
        # ROS pubs/subs
        # =============================
        self.vel_pub = self.create_publisher(Twist, f"{self.sensor_prefix}/cmd_vel", 10)
        self.debug_img_pub = self.create_publisher(Image, "/aruco/debug_image", 10)

        self.bridge = CvBridge()
        self.latest_frame: Optional[np.ndarray] = None
        self.image_fresh = False
        self.image_sub = self.create_subscription(
            Image,
            f"{self.sensor_prefix}/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

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
                lambda msg, k=key: self.sonar_callback(msg, k),
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
        # Q-table policy
        # =============================
        self.n_sonar_states = N_SONAR_STATES
        self.n_camera_states = CAMERA_STATES
        self.n_states = N_TOTAL_STATES
        self.policy = QTablePolicy(n_states=self.n_states, n_actions=5)
        loaded = self.policy.load(self.q_table_path)

        self.qtable_valid = loaded
        if loaded:
            nonzero_rows = int(np.count_nonzero(np.any(np.abs(self.policy.Q) > 1e-12, axis=1)))
            nonzero_cells = int(np.count_nonzero(np.abs(self.policy.Q) > 1e-12))
            self.get_logger().warning(
                f"Loaded Q-table OK: {self.q_table_path} | shape={self.policy.Q.shape} | "
                f"nonzero_rows={nonzero_rows}/{self.n_states}, nonzero_cells={nonzero_cells}"
            )
        else:
            self.get_logger().error(
                f"Q-table invalid or shape mismatch. Expected {(self.n_states, 5)}: {self.q_table_path}"
            )
            if self.stop_if_qtable_invalid:
                self.get_logger().error("Control will stay stopped because stop_if_qtable_invalid=True")

        # Runtime state
        self.step_count = 0
        self.debug_counter = 0
        self.stopped = False
        self.stop_reason = ""
        self.wait_after_start = True
        self.start_ready_time = time.time() + 1.0
        self.last_wait_warn_time = 0.0

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info("SARSA 3-sonar 5-bin + ArUco exploitation node initialized")
        self.get_logger().info(
            f"State size: sonar={self.n_sonar_states}, camera={self.n_camera_states}, total={self.n_states}"
        )
        self.get_logger().info(f"Sonar 5-bin thresholds: {self.sonar_discretizer.thresholds_text()}")

    # =============================
    # Callbacks
    # =============================
    def image_callback(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_fresh = True
        except Exception as exc:
            self.get_logger().warning(f"Image callback failed: {exc}")
            self.latest_frame = None
            self.image_fresh = False

    def sonar_callback(self, msg: Range, key: str) -> None:
        try:
            d = float(msg.range)
        except Exception:
            return
        if d != d or d in (float("inf"), float("-inf")):
            self.get_logger().warning(f"Invalid sonar {key}: {d}")
            return
        self.raw_sonar[key] = float(max(min(d, MAX_SONAR), MIN_SONAR))
        self.sensor_fresh[key] = True

    # =============================
    # State encoding
    # =============================
    def sonar_state_3(self) -> Tuple[int, int, int]:
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

    # =============================
    # Camera and goal
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
        cv2.putText(vis, f"bins: {self.sonar_discretizer.thresholds_text()}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        if not info.get("goal_id_found", False):
            cv2.putText(vis, f"goal_id not found | seen={info.get('ids', [])}", (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return vis

        pts = info.get("corners", None)
        if pts is not None:
            cv2.polylines(vis, [pts.astype(np.int32)], True, (0, 255, 0), 2)
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
    # Safety / readiness
    # =============================
    def ready_after_start(self) -> bool:
        if not self.wait_after_start:
            return True

        now = time.time()
        if now < self.start_ready_time:
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
                self.get_logger().warning("Waiting fresh data before control: " + ", ".join(parts))
                self.last_wait_warn_time = now
            return False

        self.wait_after_start = False
        self.get_logger().warning("Fresh 3-sonar 5-bin and camera ready, exploitation started")
        return True

    def is_crash_3sonar(self) -> bool:
        front = float(self.raw_sonar.get("front", MAX_SONAR))
        left_1 = float(self.raw_sonar.get("left_1", MAX_SONAR))
        right_1 = float(self.raw_sonar.get("right_1", MAX_SONAR))
        return (
            front < self.front_crash_threshold_m
            or left_1 < self.side_crash_threshold_m
            or right_1 < self.side_crash_threshold_m
        )

    def stop_once(self, reason: str) -> None:
        stop(self.vel_pub)
        self.stopped = True
        self.stop_reason = reason
        self.get_logger().warning(reason)

    # =============================
    # Main loop
    # =============================
    def control_loop(self) -> None:
        try:
            self._control_loop_impl()
        except Exception as exc:
            stop(self.vel_pub)
            self.stopped = True
            self.get_logger().error(f"control_loop exception -> stopped: {exc}")

    def _control_loop_impl(self) -> None:
        if self.stopped:
            stop(self.vel_pub)
            return

        if self.stop_if_qtable_invalid and not self.qtable_valid:
            stop(self.vel_pub)
            return

        if not self.ready_after_start():
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
                "streak": self.goal_streak,
                "goal_reached": False,
            }
        else:
            camera_state, reached_goal, goal_dbg = self.read_camera_goal()

        s_idx = self.combined_state_to_index(sonar_state, camera_state)
        qrow = self.policy.qrow(s_idx)
        untrained = self.policy.is_untrained(s_idx)
        greedy_action = self.policy.choose_greedy(s_idx)

        if reached_goal:
            self.stop_once(
                f"GOAL reached -> stop | step={self.step_count} sonar5={sonar_state} "
                f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} idx={s_idx}"
            )
            return

        if crash:
            self.stop_once(
                f"CRASH detected -> stop | step={self.step_count} sonar5={sonar_state} "
                f"raw_front={self.raw_sonar['front']:.2f} raw_left1={self.raw_sonar['left_1']:.2f} "
                f"raw_right1={self.raw_sonar['right_1']:.2f} idx={s_idx}"
            )
            return

        if untrained and self.stop_on_untrained_state:
            self.stop_once(
                f"UNTRAINED STATE -> stop, not forcing argmax action 0 | "
                f"sonar5={sonar_state} cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                f"idx={s_idx} Q={qrow.tolist()}"
            )
            return

        do_action(self.vel_pub, greedy_action)
        self.step_count += 1

        if self.step_count % self.log_every == 0:
            self.get_logger().info(
                f"[EXPLOIT] step={self.step_count:04d} "
                f"sonar5={sonar_state} raw=({self.raw_sonar['front']:.2f},"
                f"{self.raw_sonar['left_1']:.2f},{self.raw_sonar['right_1']:.2f}) "
                f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                f"area={goal_dbg.get('area_ratio', 0.0):.4f} "
                f"front_ok={goal_dbg.get('front_ok', False)} "
                f"idx={s_idx} Q={[round(float(x), 3) for x in qrow.tolist()]} "
                f"action={greedy_action} {ACTION_NAMES.get(greedy_action)} "
                f"untrained={untrained} goal_streak={goal_dbg.get('streak', 0)}"
            )

        if self.max_control_steps > 0 and self.step_count >= self.max_control_steps:
            self.stop_once(f"Max control steps reached ({self.max_control_steps}) -> stop")


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode3Sonar5BinArucoExploit()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Control node stopped by user.")
    finally:
        try:
            stop(node.vel_pub)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
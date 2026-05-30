#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_node_3sonar5bin_aruco_real_udp_only.py

Node kontrol REAL ROBOT untuk model SARSA 3 ultrasonic 5-bin + ArUco.
Versi ini KHUSUS kamera UDP chunked JBF1.

Tidak memakai:
  - topic kamera ROS
  - sensor_msgs/Image
  - cv_bridge
  - reset_world

Alur:
  Host Jetson sender_chunked_jetson.py
      -> UDP JPEG chunked JBF1 port 5020
  Node ini di Docker
      -> reassemble JPEG
      -> cv2.imdecode
      -> deteksi ArUco
      -> gabung state sonar + camera
      -> pilih action dari Q-table
      -> publish cmd_vel

State:
  sonar_state = (front, left_1, right_1), masing-masing 0..4
  camera_state = 0..6
  combined_idx = sonar_idx * 7 + camera_state
  total state = 5^3 * 7 = 875

Action mengikuti utils/control.py:
  0 = forward
  1 = turn left wide
  2 = turn right wide
  3 = left bit
  4 = right bit
"""

import random
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Range
from std_msgs.msg import Float32

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

DEFAULT_Q_TABLE_PATH = "/home/pc/ros2_ws/src/jetbot_ros2/scripts/v2/data/Q_table_3sonar5bin_aruco_sim.csv"

# Chunked JPEG UDP protocol, compatible with sender_chunked_jetson.py / old receiver.py
MAGIC = b"JBF1"
HEADER_STRUCT = struct.Struct("!4sIHHH")
HEADER_SIZE = HEADER_STRUCT.size
MAX_FRAME_AGE_S = 1.0


class FrameReassembler:
    """Reassemble chunked UDP JPEG frames with JBF1 header."""

    def __init__(self, max_frame_age_s: float = MAX_FRAME_AGE_S):
        self.max_frame_age_s = float(max_frame_age_s)
        self.frames: Dict[int, Dict[str, Any]] = {}

    def add_packet(self, data: bytes) -> Optional[Tuple[int, bytes]]:
        if len(data) < HEADER_SIZE:
            return None

        try:
            magic, frame_id, chunk_idx, chunk_total, payload_len = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        except Exception:
            return None

        if magic != MAGIC:
            return None

        payload = data[HEADER_SIZE:]
        if len(payload) != payload_len:
            return None
        if chunk_total <= 0 or chunk_idx >= chunk_total:
            return None

        now = time.time()
        old_ids = [
            fid for fid, rec in self.frames.items()
            if now - float(rec.get("t", now)) > self.max_frame_age_s
        ]
        for fid in old_ids:
            self.frames.pop(fid, None)

        rec = self.frames.get(frame_id)
        if rec is None:
            rec = {"t": now, "total": int(chunk_total), "chunks": {}}
            self.frames[frame_id] = rec

        if int(rec.get("total", chunk_total)) != int(chunk_total):
            rec = {"t": now, "total": int(chunk_total), "chunks": {}}
            self.frames[frame_id] = rec

        rec["chunks"][int(chunk_idx)] = payload

        if len(rec["chunks"]) == int(rec["total"]):
            try:
                jpg = b"".join(rec["chunks"][i] for i in range(int(rec["total"])))
            except KeyError:
                return None
            self.frames.pop(frame_id, None)
            return int(frame_id), jpg

        return None


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
        try:
            d = float(raw)
        except Exception:
            return MAX_SONAR
        if d != d or d in (float("inf"), float("-inf")):
            return MAX_SONAR
        return float(np.clip(d, MIN_SONAR, MAX_SONAR))

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
    def __init__(self, n_states: int = N_TOTAL_STATES, n_actions: int = 5):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        self.loaded = False

    def load(self, path: str) -> Tuple[bool, str]:
        try:
            q = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        except FileNotFoundError:
            return False, f"Q-table file not found: {path}"
        except Exception as exc:
            return False, f"Failed reading Q-table: {exc}"

        if q.shape != (self.n_states, self.n_actions):
            self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
            self.loaded = False
            return False, f"Q-table shape mismatch. Got {q.shape}, expected {(self.n_states, self.n_actions)}"

        self.Q = q
        self.loaded = True
        return True, "OK"

    def qrow(self, state_idx: int) -> np.ndarray:
        return self.Q[int(state_idx), :]

    def is_untrained(self, state_idx: int) -> bool:
        return bool(np.allclose(self.qrow(state_idx), 0.0))

    def choose_greedy(self, state_idx: int) -> int:
        return int(np.argmax(self.qrow(state_idx)))


class RealControlNodeUdpOnly(Node):
    def __init__(self):
        super().__init__("sarsa_real_control_3sonar5bin_aruco_udp_only")

        # Basic params
        self.declare_parameter("sensor_prefix", "/jetbotV21")
        self.declare_parameter("q_table_path", DEFAULT_Q_TABLE_PATH)
        self.declare_parameter("control_period", 0.12)
        self.declare_parameter("max_control_steps", 0)

        # UDP camera only
        self.declare_parameter("udp_bind_host", "0.0.0.0")
        self.declare_parameter("udp_port", 5020)
        self.declare_parameter("udp_buffer_size", 65535)

        self.sensor_prefix = str(self.get_parameter("sensor_prefix").value)
        self.declare_parameter("cmd_vel_topic", f"jetbot/cmd_vel")
        self.declare_parameter("front_topic", f"{self.sensor_prefix}/ultrasonic_front")
        self.declare_parameter("left1_topic", f"{self.sensor_prefix}/ultrasonic_left_1")
        self.declare_parameter("right1_topic", f"{self.sensor_prefix}/ultrasonic_right_1")

        # Real robot sering pakai Float32, simulasi sering pakai Range.
        self.declare_parameter("sonar_msg_type", "float32")  # "float32" or "range"

        # Unit data sonar yang masuk dari topic.
        # Pilihan umum: "m" untuk meter, "cm" untuk centimeter, "mm" untuk millimeter.
        self.declare_parameter("sonar_input_unit", "m")

        # ArUco/camera params. Samakan dengan training bila memungkinkan.
        self.declare_parameter("goal_id", 23)
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("goal_front_threshold_m", 0.40)
        self.declare_parameter("goal_min_streak", 2)
        self.declare_parameter("aruco_near_area_ratio", 0.002)
        self.declare_parameter("camera_left_boundary_ratio", 0.35)
        self.declare_parameter("camera_right_boundary_ratio", 0.65)

        # Crash safety threshold real robot.
        self.declare_parameter("front_crash_threshold_m", 0.25)
        self.declare_parameter("side_crash_threshold_m", 0.20)

        # 5-bin sonar thresholds. Harus sama dengan training.
        self.declare_parameter("sonar_bin_danger_m", 0.20)
        self.declare_parameter("sonar_bin_close_m", 0.35)
        self.declare_parameter("sonar_bin_medium_m", 0.60)
        self.declare_parameter("sonar_bin_clear_m", 1.00)

        # Safety runtime.
        self.declare_parameter("startup_wait_sec", 1.0)
        self.declare_parameter("sensor_stale_timeout_sec", 0.75)
        self.declare_parameter("camera_stale_timeout_sec", 1.00)
        self.declare_parameter("stop_if_qtable_invalid", True)
        self.declare_parameter("enable_safety_action_filter", True)

        # Unknown-state policy: stop / safe_random / greedy_zero
        self.declare_parameter("unknown_state_policy", "stop")
        self.declare_parameter("log_every", 1)

        # Read params
        self.q_table_path = str(self.get_parameter("q_table_path").value)
        self.control_period = float(self.get_parameter("control_period").value)
        self.max_control_steps = int(self.get_parameter("max_control_steps").value)

        self.udp_bind_host = str(self.get_parameter("udp_bind_host").value)
        self.udp_port = int(self.get_parameter("udp_port").value)
        self.udp_buffer_size = int(self.get_parameter("udp_buffer_size").value)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.front_topic = str(self.get_parameter("front_topic").value)
        self.left1_topic = str(self.get_parameter("left1_topic").value)
        self.right1_topic = str(self.get_parameter("right1_topic").value)
        self.sonar_msg_type = str(self.get_parameter("sonar_msg_type").value).strip().lower()
        self.sonar_input_unit = str(self.get_parameter("sonar_input_unit").value).strip().lower()
        if self.sonar_input_unit not in ("m", "meter", "meters", "cm", "centimeter", "centimeters", "mm", "millimeter", "millimeters"):
            self.get_logger().warning(
                f"sonar_input_unit={self.sonar_input_unit} invalid; fallback to m"
            )
            self.sonar_input_unit = "m"

        self.goal_front_threshold_m = float(self.get_parameter("goal_front_threshold_m").value)
        self.goal_min_streak = int(self.get_parameter("goal_min_streak").value)
        self.aruco_near_area_ratio = float(self.get_parameter("aruco_near_area_ratio").value)
        self.camera_left_boundary_ratio = float(self.get_parameter("camera_left_boundary_ratio").value)
        self.camera_right_boundary_ratio = float(self.get_parameter("camera_right_boundary_ratio").value)
        self.front_crash_threshold_m = float(self.get_parameter("front_crash_threshold_m").value)
        self.side_crash_threshold_m = float(self.get_parameter("side_crash_threshold_m").value)

        self.startup_wait_sec = float(self.get_parameter("startup_wait_sec").value)
        self.sensor_stale_timeout_sec = float(self.get_parameter("sensor_stale_timeout_sec").value)
        self.camera_stale_timeout_sec = float(self.get_parameter("camera_stale_timeout_sec").value)
        self.stop_if_qtable_invalid = bool(self.get_parameter("stop_if_qtable_invalid").value)
        self.enable_safety_action_filter = bool(self.get_parameter("enable_safety_action_filter").value)
        self.unknown_state_policy = str(self.get_parameter("unknown_state_policy").value).strip().lower()
        self.log_every = max(1, int(self.get_parameter("log_every").value))

        if self.unknown_state_policy not in ("stop", "safe_random", "greedy_zero"):
            self.get_logger().warning(
                f"unknown_state_policy={self.unknown_state_policy} invalid; fallback to stop"
            )
            self.unknown_state_policy = "stop"

        if self.camera_left_boundary_ratio >= self.camera_right_boundary_ratio:
            self.get_logger().warning("Invalid camera boundary; fallback to 0.35/0.65")
            self.camera_left_boundary_ratio = 0.35
            self.camera_right_boundary_ratio = 0.65

        self.sonar_discretizer = Sonar5BinDiscretizer(
            danger_m=float(self.get_parameter("sonar_bin_danger_m").value),
            close_m=float(self.get_parameter("sonar_bin_close_m").value),
            medium_m=float(self.get_parameter("sonar_bin_medium_m").value),
            clear_m=float(self.get_parameter("sonar_bin_clear_m").value),
        )

        # ROS pub/sub: hanya cmd_vel dan ultrasonic. Tidak ada topic kamera.
        self.vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.sensor_topics = {
            "front": self.front_topic,
            "left_1": self.left1_topic,
            "right_1": self.right1_topic,
        }
        self.raw_sonar = {k: MAX_SONAR for k in self.sensor_topics.keys()}
        self.raw_sonar_input = {k: MAX_SONAR for k in self.sensor_topics.keys()}
        self.sensor_fresh = {k: False for k in self.sensor_topics.keys()}
        self.last_sensor_time = {k: 0.0 for k in self.sensor_topics.keys()}

        for key, topic in self.sensor_topics.items():
            if self.sonar_msg_type == "range":
                self.create_subscription(
                    Range,
                    topic,
                    lambda msg, k=key: self.sonar_range_callback(msg, k),
                    qos_profile_sensor_data,
                )
            else:
                self.create_subscription(
                    Float32,
                    topic,
                    lambda msg, k=key: self.sonar_float32_callback(msg, k),
                    qos_profile_sensor_data,
                )

        # UDP camera state
        self.latest_frame: Optional[np.ndarray] = None
        self.last_image_time = 0.0
        self.image_fresh = False
        self.latest_frame_id: Optional[int] = None
        self.decoded_frame_count = 0
        self.frame_reassembler = FrameReassembler()
        self.udp_sock: Optional[socket.socket] = None
        self.udp_stop_event: Optional[threading.Event] = None
        self.udp_thread: Optional[threading.Thread] = None
        self.start_udp_receiver()

        # ArUco detector
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

        # Q-table policy
        self.n_sonar_states = N_SONAR_STATES
        self.n_camera_states = CAMERA_STATES
        self.n_states = N_TOTAL_STATES
        self.policy = QTablePolicy(n_states=self.n_states, n_actions=5)
        loaded, load_msg = self.policy.load(self.q_table_path)
        self.qtable_valid = loaded

        if loaded:
            nonzero_rows = int(np.count_nonzero(np.any(np.abs(self.policy.Q) > 1e-12, axis=1)))
            nonzero_cells = int(np.count_nonzero(np.abs(self.policy.Q) > 1e-12))
            self.get_logger().warning(
                f"Loaded Q-table OK: {self.q_table_path} | shape={self.policy.Q.shape} | "
                f"nonzero_rows={nonzero_rows}/{self.n_states}, nonzero_cells={nonzero_cells}"
            )
        else:
            self.get_logger().error(f"Q-table invalid: {load_msg}")
            if self.stop_if_qtable_invalid:
                self.get_logger().error("Control will stay stopped because stop_if_qtable_invalid=True")

        # Runtime
        self.step_count = 0
        self.stopped = False
        self.stop_reason = ""
        self.wait_after_start = True
        self.start_ready_time = time.time() + self.startup_wait_sec
        self.last_wait_warn_time = 0.0
        self.last_stale_warn_time = 0.0

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().warning("REAL robot control UDP-only initialized: SARSA 3-sonar 5-bin + ArUco")
        self.get_logger().info(f"cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().warning(f"UDP image receiver listening on {self.udp_bind_host}:{self.udp_port} protocol=chunked_jbf1")
        self.get_logger().info(f"sonar_msg_type={self.sonar_msg_type}")
        self.get_logger().info(f"front_topic={self.front_topic}")
        self.get_logger().info(f"left1_topic={self.left1_topic}")
        self.get_logger().info(f"right1_topic={self.right1_topic}")
        self.get_logger().info(
            f"State size: sonar={self.n_sonar_states}, camera={self.n_camera_states}, total={self.n_states}"
        )
        self.get_logger().info(f"Sonar 5-bin thresholds: {self.sonar_discretizer.thresholds_text()}")
        self.get_logger().warning(
            f"unknown_state_policy={self.unknown_state_policy}, "
            f"safety_action_filter={self.enable_safety_action_filter}"
        )
        self.get_logger().warning(
            f"sonar_msg_type={self.sonar_msg_type}, sonar_input_unit={self.sonar_input_unit}; "
            "internal sonar values are meters"
        )

    # =============================
    # UDP camera receiver
    # =============================
    def start_udp_receiver(self) -> None:
        try:
            self.udp_stop_event = threading.Event()
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_sock.bind((self.udp_bind_host, self.udp_port))
            self.udp_sock.settimeout(0.5)
            self.udp_thread = threading.Thread(target=self.udp_image_loop, daemon=True)
            self.udp_thread.start()
        except Exception as exc:
            self.get_logger().error(f"Failed to start UDP image receiver: {exc}")
            self.udp_sock = None
            self.udp_stop_event = None
            self.udp_thread = None

    def stop_udp_receiver(self) -> None:
        try:
            if self.udp_stop_event is not None:
                self.udp_stop_event.set()
            if self.udp_sock is not None:
                self.udp_sock.close()
        except Exception:
            pass

    def udp_image_loop(self) -> None:
        last_error_log = 0.0
        while self.udp_stop_event is not None and not self.udp_stop_event.is_set():
            try:
                if self.udp_sock is None:
                    time.sleep(0.1)
                    continue

                data, _addr = self.udp_sock.recvfrom(self.udp_buffer_size)
                if not data:
                    continue

                ready = self.frame_reassembler.add_packet(data)
                if ready is None:
                    continue

                frame_id, jpg = ready
                np_buf = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
                if frame is None:
                    now = time.time()
                    if now - last_error_log > 2.0:
                        self.get_logger().warning(
                            f"UDP chunked image decode failed. frame_id={frame_id} bytes={len(jpg)}"
                        )
                        last_error_log = now
                    continue

                self.latest_frame = frame
                self.image_fresh = True
                self.last_image_time = time.time()
                self.latest_frame_id = int(frame_id)
                self.decoded_frame_count += 1

            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                now = time.time()
                if now - last_error_log > 2.0:
                    self.get_logger().warning(f"UDP image receiver error: {exc}")
                    last_error_log = now

    # =============================
    # Sonar callbacks
    # =============================
    def sonar_to_meters(self, value: float) -> float:
        unit = self.sonar_input_unit
        if unit in ("cm", "centimeter", "centimeters"):
            return float(value) * 0.01
        if unit in ("mm", "millimeter", "millimeters"):
            return float(value) * 0.001
        return float(value)

    def _set_sonar(self, key: str, value: float) -> None:
        try:
            raw_input = float(value)
        except Exception:
            return
        if raw_input != raw_input or raw_input in (float("inf"), float("-inf")):
            self.get_logger().warning(f"Invalid sonar {key}: {raw_input}")
            return

        d = self.sonar_to_meters(raw_input)
        if d != d or d in (float("inf"), float("-inf")):
            self.get_logger().warning(f"Invalid converted sonar {key}: raw={raw_input}, converted={d}")
            return

        self.raw_sonar_input[key] = raw_input
        self.raw_sonar[key] = float(max(min(d, MAX_SONAR), MIN_SONAR))
        self.sensor_fresh[key] = True
        self.last_sensor_time[key] = time.time()

    def sonar_range_callback(self, msg: Range, key: str) -> None:
        self._set_sonar(key, msg.range)

    def sonar_float32_callback(self, msg: Float32, key: str) -> None:
        self._set_sonar(key, msg.data)

    # =============================
    # State encoding
    # =============================
    def sonar_state_3(self) -> Tuple[int, int, int]:
        return self.sonar_discretizer.process(self.raw_sonar)

    @staticmethod
    def sonar_state_to_index(sonar_state: Tuple[int, int, int]) -> int:
        front, left_1, right_1 = sonar_state
        return int(front) * 25 + int(left_1) * 5 + int(right_1)

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
            return CAM_NONE, False, self._empty_goal_info()

        try:
            found, info = self.goal_detector.detect_goal_marker(self.latest_frame)
        except Exception as exc:
            self.get_logger().warning(f"ArUco detection failed: {exc}")
            self.goal_streak = 0
            return CAM_NONE, False, self._empty_goal_info()

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
        return camera_state, reached_goal, info

    def _empty_goal_info(self) -> Dict:
        return {
            "goal_id_found": False,
            "camera_state": CAM_NONE,
            "camera_state_name": CAMERA_STATE_NAMES[CAM_NONE],
            "front_distance_m": float(self.raw_sonar.get("front", MAX_SONAR)),
            "front_ok": False,
            "streak": 0,
            "goal_reached": False,
            "area_ratio": 0.0,
        }

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
                    parts.append("udp_camera=waiting")
                self.get_logger().warning("Waiting fresh data before real control: " + ", ".join(parts))
                self.last_wait_warn_time = now
            return False

        self.wait_after_start = False
        self.get_logger().warning("Fresh ultrasonic and UDP camera ready, REAL control started")
        return True

    def data_is_stale(self) -> bool:
        now = time.time()
        stale = []
        for k in SONAR_KEYS_3:
            if now - self.last_sensor_time.get(k, 0.0) > self.sensor_stale_timeout_sec:
                stale.append(k)
        camera_stale = now - self.last_image_time > self.camera_stale_timeout_sec

        if stale or camera_stale:
            stop(self.vel_pub)
            if now - self.last_stale_warn_time > 1.0:
                parts = []
                if stale:
                    parts.append(f"stale_sensors={stale}")
                if camera_stale:
                    parts.append("udp_camera=stale")
                self.get_logger().warning("Stale data -> stop: " + ", ".join(parts))
                self.last_stale_warn_time = now
            return True
        return False

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
    # Unknown / safety action helpers
    # =============================
    def safe_candidates(self, sonar_state: Tuple[int, int, int]) -> List[int]:
        front, left, right = sonar_state
        candidates = [0, 1, 2, 3, 4]

        # Jangan maju kalau depan danger/close.
        if front <= 1:
            candidates = [a for a in candidates if a != 0]

        # Untuk lorong sempit, left/right == 1 belum tentu bahaya terminal.
        # Block hanya saat bin 0.
        if left <= 0:
            candidates = [a for a in candidates if a not in (1, 3)]
        if right <= 0:
            candidates = [a for a in candidates if a not in (2, 4)]

        if candidates:
            return candidates

        if left > right:
            return [1, 3]
        if right > left:
            return [2, 4]
        return [1, 2]

    def choose_safe_random_action(self, sonar_state: Tuple[int, int, int], camera_state: int) -> int:
        candidates = self.safe_candidates(sonar_state)

        preferred: List[int] = []
        if camera_state in (CAM_CENTER_FAR, CAM_CENTER_NEAR):
            preferred = [0, 3, 4]
        elif camera_state in (CAM_LEFT_FAR, CAM_LEFT_NEAR):
            preferred = [3, 1]
        elif camera_state in (CAM_RIGHT_FAR, CAM_RIGHT_NEAR):
            preferred = [4, 2]

        preferred = [a for a in preferred if a in candidates]
        if preferred and random.random() < 0.75:
            return int(random.choice(preferred))
        return int(random.choice(candidates))

    def action_is_safe(self, action: int, sonar_state: Tuple[int, int, int]) -> bool:
        front, left, right = sonar_state
        if action == 0 and front <= 1:
            return False
        if action in (1, 3) and left <= 0:
            return False
        if action in (2, 4) and right <= 0:
            return False
        return True

    def choose_action(self, sonar_state: Tuple[int, int, int], camera_state: int, state_idx: int) -> Tuple[Optional[int], str, np.ndarray, bool]:
        qrow = self.policy.qrow(state_idx)
        untrained = self.policy.is_untrained(state_idx)
        greedy_action = self.policy.choose_greedy(state_idx)

        if untrained:
            if self.unknown_state_policy == "stop":
                return None, "UNTRAINED_STOP", qrow, True
            if self.unknown_state_policy == "safe_random":
                return self.choose_safe_random_action(sonar_state, camera_state), "UNTRAINED_SAFE_RANDOM", qrow, True
            return greedy_action, "UNTRAINED_GREEDY_ZERO", qrow, True

        action = greedy_action
        reason = "GREEDY"

        if self.enable_safety_action_filter and not self.action_is_safe(action, sonar_state):
            safe_action = self.choose_safe_random_action(sonar_state, camera_state)
            reason = f"GREEDY_BLOCKED_TO_SAFE_RANDOM({ACTION_NAMES.get(action)})"
            action = safe_action

        return int(action), reason, qrow, False

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

        if self.data_is_stale():
            return

        sonar_state = self.sonar_state_3()
        crash = self.is_crash_3sonar()

        if crash:
            camera_state = CAM_NONE
            reached_goal = False
            goal_dbg = self._empty_goal_info()
        else:
            camera_state, reached_goal, goal_dbg = self.read_camera_goal()

        state_idx = self.combined_state_to_index(sonar_state, camera_state)

        if reached_goal:
            self.stop_once(
                f"GOAL reached -> stop | step={self.step_count} sonar5={sonar_state} "
                f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} idx={state_idx}"
            )
            return

        if crash:
            self.stop_once(
                f"CRASH detected -> stop | step={self.step_count} sonar5={sonar_state} "
                f"sonar_m=({self.raw_sonar['front']:.2f},{self.raw_sonar['left_1']:.2f},{self.raw_sonar['right_1']:.2f}) "
                f"sonar_in=({self.raw_sonar_input['front']:.2f},{self.raw_sonar_input['left_1']:.2f},{self.raw_sonar_input['right_1']:.2f}) "
                f"idx={state_idx}"
            )
            return

        action, action_reason, qrow, untrained = self.choose_action(sonar_state, camera_state, state_idx)

        if action is None:
            self.stop_once(
                f"UNTRAINED STATE -> stop | sonar5={sonar_state} "
                f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                f"idx={state_idx} Q={qrow.tolist()}"
            )
            return

        do_action(self.vel_pub, action)
        self.step_count += 1

        if self.step_count % self.log_every == 0:
            self.get_logger().info(
                f"[REAL_CONTROL_UDP] step={self.step_count:04d} "
                f"frame_id={self.latest_frame_id} frames={self.decoded_frame_count} "
                f"sonar5={sonar_state} sonar_m=({self.raw_sonar['front']:.2f},"
                f"{self.raw_sonar['left_1']:.2f},{self.raw_sonar['right_1']:.2f}) "
                f"sonar_in=({self.raw_sonar_input['front']:.2f},"
                f"{self.raw_sonar_input['left_1']:.2f},{self.raw_sonar_input['right_1']:.2f}) "
                f"cam={camera_state} {CAMERA_STATE_NAMES.get(camera_state)} "
                f"area={goal_dbg.get('area_ratio', 0.0):.4f} "
                f"front_ok={goal_dbg.get('front_ok', False)} "
                f"idx={state_idx} Q={[round(float(x), 3) for x in qrow.tolist()]} "
                f"action={action} {ACTION_NAMES.get(action)} reason={action_reason} "
                f"untrained={untrained} goal_streak={goal_dbg.get('streak', 0)}"
            )

        if self.max_control_steps > 0 and self.step_count >= self.max_control_steps:
            self.stop_once(f"Max control steps reached ({self.max_control_steps}) -> stop")


def main(args=None):
    rclpy.init(args=args)
    node = RealControlNodeUdpOnly()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Real UDP-only control node stopped by user.")
    finally:
        try:
            stop(node.vel_pub)
        except Exception:
            pass
        try:
            node.stop_udp_receiver()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
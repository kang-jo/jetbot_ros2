#!/usr/bin/env python3
# learning_hitl_finetune.py
#
# Coverage-guided HITL fine-tuning for SARSA JetBot no-odom.
#
# Main idea:
# - Load baseline Q-table from pure SARSA.
# - Robot runs normally using current policy.
# - If robot enters an unseen / low-coverage state, robot stops temporarily.
# - Human provides action (E/A/S/D/F).
# - Executed human action is used in on-policy SARSA update.
#
# Notes:
# - State remains 7 ultrasonic sensors only.
# - Goal remains ArUco + front sonar, same as baseline node.
# - Camera bonus shaping is optional and OFF by default to stay closer to pure SARSA.

import csv
import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Float32
from std_srvs.srv import Empty

from utils.aruco_detect import GoalArucoConfig, GoalDetectorAruco
from utils.control import do_action, stop
from utils.live_logger import LivePlotLogger
from utils.reward import compute_reward
from utils.sarsa import SARSAAgent
from utils.sonar import SENSOR_KEYS, SonarDiscretizer


# ==============================
# Paths: adjust to your machine
# ==============================
BASE_Q_TABLE_PATH = "data/Q_table_camera_HITL_01.csv"
SAVE_Q_TABLE_PATH = "data/Q_table_camera_HITL_finetune1.csv"
STATE_COUNT_PATH = "data/state_count_HITL_finetune1.npy"
COVERAGE_LOG_PATH = "data/coverage_log_HITL_finetune1.csv"


@dataclass
class WaitContext:
    state_idx: int
    state_tuple: Tuple[int, ...]
    is_initial_step: bool
    pending_reward: float = 0.0
    previous_state_idx: Optional[int] = None
    previous_action: Optional[int] = None


class LearningNode(Node):
    def __init__(self):
        super().__init__("sarsa_hitl_finetune_no_odom")

        # ===== Logging =====
        self.logger_plot = LivePlotLogger(enable_plot=False)

        # ===== Config =====
        self.control_period = 0.2
        self.max_step = 1000
        self.enable_camera_bonus = False
        self.publish_debug_image_enabled = True
        self.debug_image_every = 4
        self.debug_counter = 0

        # HITL coverage trigger:
        # trigger when state_count < this threshold.
        # 1 means: only states never seen before will trigger HITL.
        self.hitl_trigger_threshold = 1

        # Baseline Q-table does not contain state counts.
        # We approximate explored states as rows whose Q values are not all zero.
        # Such states are initialized with this pseudo-count.
        self.known_state_initial_count = 1

        self.human_wait_timeout = 2.0
        self.fallback_to_agent_after_timeout = True
        self.fallback_to_safe_policy_after_timeout = False

        # E/A/S/D/F action mapping requested by user:
        # E maju, A kiri, S kiri dikit, D kanan dikit, F kanan
        self.key_to_action = {
            "e": 0,
            "a": 1,
            "f": 2,
            "s": 3,
            "d": 4,
        }

        self.step_in_episode = 0
        self.hitl_count_episode = 0
        self.new_states_episode = 0
        self.cumulated_reward = 0.0
        self.episode = 0

        # ===== ROS pubs/subs =====
        self.vel_pub = self.create_publisher(Twist, "/jetbotV21/cmd_vel", 10)
        self.reward_pub = self.create_publisher(Float32, "/jetbotV21/reward", 10)
        self.reset_client = self.create_client(Empty, "reset_world")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.image_fresh = False
        self.debug_img_pub = self.create_publisher(Image, "/aruco/debug_image", 10)
        self.image_sub = self.create_subscription(
            Image,
            "/jetbotV21/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.goal_detector = GoalDetectorAruco(
            GoalArucoConfig(
                goal_id=23,
                dictionary_name="DICT_6X6_250",
                front_threshold_m=0.35,
                min_streak=2,
                use_area_check=False,
                min_area_ratio=0.0,
                use_center_check=True,
                center_tolerance_ratio=0.25,
                debug=False,
            )
        )

        self.sensor_topics = {
            "left_0": "/jetbotV21/ultrasonic_left_0",
            "left_1": "/jetbotV21/ultrasonic_left_1",
            "left_2": "/jetbotV21/ultrasonic_left_2",
            "front": "/jetbotV21/ultrasonic_front",
            "right_0": "/jetbotV21/ultrasonic_right_0",
            "right_1": "/jetbotV21/ultrasonic_right_1",
            "right_2": "/jetbotV21/ultrasonic_right_2",
        }
        self.raw_sonar = {k: 1.5 for k in self.sensor_topics.keys()}
        self.sensor_fresh = {k: False for k in self.sensor_topics.keys()}
        for key, topic in self.sensor_topics.items():
            self.create_subscription(
                Range,
                topic,
                lambda msg, k=key: self.sonar_cb(msg, k),
                qos_profile_sensor_data,
            )

        self.wait_after_reset = True
        self.reset_ready_time = time.time() + 1.0
        self.last_wait_warn_time = 0.0

        # ===== RL objects =====
        self.discretizer = SonarDiscretizer(keys=SENSOR_KEYS)
        self.agent = SARSAAgent(
            n_actions=5,
            alpha=0.12,
            gamma=0.92,
            epsilon=0.10,
            state_dims=7,
            save_path=SAVE_Q_TABLE_PATH,
        )
        self.agent.load_qtable(BASE_Q_TABLE_PATH)

        # Coverage count: initialize from Q-table and optionally resume from file
        self.state_count = np.zeros(self.agent.n_states, dtype=np.int32)
        self._bootstrap_state_count_from_qtable()
        self._try_load_state_count()

        # Internal RL step memory
        self.prev_state_idx = None
        self.prev_action = None

        # HITL wait state
        self.waiting_for_human = False
        self.wait_started_at = 0.0
        self.wait_context: Optional[WaitContext] = None

        # Terminal keyboard setup
        self.stdin_ready = False
        self.stdin_fd = None
        self.old_term_settings = None
        self._setup_keyboard()

        self.timer = self.create_timer(self.control_period, self.control_loop)

        self.get_logger().info("SARSA HITL fine-tune node initialized")
        self.get_logger().info(f"BASE_Q_TABLE_PATH={BASE_Q_TABLE_PATH}")
        self.get_logger().info(f"SAVE_Q_TABLE_PATH={SAVE_Q_TABLE_PATH}")
        self.get_logger().info(
            f"trigger_threshold={self.hitl_trigger_threshold}, "
            f"wait_timeout={self.human_wait_timeout:.1f}s, epsilon={self.agent.epsilon:.3f}"
        )

    # =============================
    # Keyboard helpers
    # =============================
    def _setup_keyboard(self):
        try:
            if sys.stdin.isatty():
                self.stdin_fd = sys.stdin.fileno()
                self.old_term_settings = termios.tcgetattr(self.stdin_fd)
                tty.setcbreak(self.stdin_fd)
                self.stdin_ready = True
                self.get_logger().info("Keyboard ready: E/A/S/D/F for HITL")
            else:
                self.get_logger().warning("stdin is not a TTY, keyboard HITL disabled")
        except Exception as exc:
            self.stdin_ready = False
            self.get_logger().warning(f"Keyboard setup failed: {exc}")

    def restore_keyboard(self):
        try:
            if self.stdin_ready and self.stdin_fd is not None and self.old_term_settings is not None:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)
        except Exception:
            pass

    def read_key_nonblocking(self) -> Optional[str]:
        if not self.stdin_ready:
            return None
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
            if rlist:
                ch = sys.stdin.read(1)
                if ch:
                    return ch.lower()
        except Exception:
            return None
        return None

    # =============================
    # Coverage helpers
    # =============================
    def _bootstrap_state_count_from_qtable(self):
        """
        Approximate which states were explored by baseline training.
        If Q row is not all zero, mark it as 'known'.
        """
        try:
            nonzero_rows = ~np.all(np.isclose(self.agent.Q, 0.0), axis=1)
            self.state_count[nonzero_rows] = self.known_state_initial_count
            known = int(np.sum(nonzero_rows))
            self.get_logger().info(
                f"Bootstrapped coverage from baseline Q-table: {known}/{self.agent.n_states} known states"
            )
        except Exception as exc:
            self.get_logger().warning(f"Failed to bootstrap coverage from Q-table: {exc}")

    def _try_load_state_count(self):
        if not os.path.exists(STATE_COUNT_PATH):
            return
        try:
            arr = np.load(STATE_COUNT_PATH)
            if arr.shape == self.state_count.shape:
                self.state_count = arr.astype(np.int32)
                visited = int(np.sum(self.state_count > 0))
                self.get_logger().warning(
                    f"Loaded previous state_count: {visited}/{self.agent.n_states} visited states"
                )
            else:
                self.get_logger().warning(
                    f"state_count shape mismatch: file={arr.shape}, expected={self.state_count.shape}"
                )
        except Exception as exc:
            self.get_logger().warning(f"Failed to load state_count: {exc}")

    def save_state_count(self):
        try:
            os.makedirs(os.path.dirname(STATE_COUNT_PATH), exist_ok=True)
            np.save(STATE_COUNT_PATH, self.state_count)
        except Exception as exc:
            self.get_logger().warning(f"Failed to save state_count: {exc}")

    def coverage_percent(self) -> float:
        return 100.0 * float(np.sum(self.state_count > 0)) / float(self.agent.n_states)

    def should_request_hitl_from_count(self, count_before: int) -> bool:
        return int(count_before) < int(self.hitl_trigger_threshold)

    # =============================
    # Callbacks
    # =============================
    def sonar_cb(self, msg, key):
        try:
            d = float(msg.range)
        except Exception:
            return
        if d != d or d == float("inf") or d == float("-inf"):
            self.get_logger().warning(f"Invalid sonar {key}: {d}")
            return
        d = float(max(min(d, 1.5), 0.05))
        self.raw_sonar[key] = d
        self.sensor_fresh[key] = True

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_fresh = True
        except Exception as exc:
            self.get_logger().warning(f"Image callback failed: {exc}")
            self.latest_frame = None
            self.image_fresh = False

    # =============================
    # Goal / reward helpers
    # =============================
    def check_goal(self):
        if self.latest_frame is None:
            return False, {}

        front_distance_m = float(self.raw_sonar.get("front", 1.5))
        try:
            reached_goal, dbg = self.goal_detector.update(self.latest_frame, front_distance_m)
        except Exception as exc:
            self.get_logger().warning(f"ArUco goal detection failed: {exc}")
            return False, {}

        if self.publish_debug_image_enabled:
            self.debug_counter += 1
            if self.debug_counter % self.debug_image_every == 0:
                try:
                    vis = self.goal_detector.draw_debug(self.latest_frame, dbg)
                    img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    self.debug_img_pub.publish(img_msg)
                except Exception as exc:
                    self.get_logger().warning(f"Failed to publish ArUco debug image: {exc}")

        return reached_goal, dbg

    def compute_camera_bonus(self, goal_dbg, reached_goal, crash):
        if reached_goal or crash:
            return 0.0

        marker_seen = bool(goal_dbg.get("goal_id_found", False))
        center_ok = bool(goal_dbg.get("center_ok", False))
        valid = bool(goal_dbg.get("valid", False))

        if valid:
            return 5.0
        if marker_seen and center_ok:
            return 1.5
        if marker_seen:
            return 0.2
        return 0.0

    def choose_agent_action(self, state_idx: int) -> int:
        """
        Local action selection with random tie-break.
        This avoids deterministic bias to action 0 when Q row is flat.
        """
        if np.random.rand() < self.agent.epsilon:
            return int(np.random.randint(0, self.agent.n_actions))
        q = self.agent.Q[state_idx, :]
        max_q = np.max(q)
        best = np.flatnonzero(np.isclose(q, max_q))
        return int(np.random.choice(best))

    def safe_fallback_action(self, state_tuple: Tuple[int, ...]) -> int:
        left = state_tuple[0:3]
        front = state_tuple[3]
        right = state_tuple[4:7]

        if front == 0:
            # turn toward safer side
            if sum(left) > sum(right):
                return 1  # left wide
            return 2      # right wide

        if left[0] == 0 and right[0] > 0:
            return 2
        if right[0] == 0 and left[0] > 0:
            return 1
        return 0

    # =============================
    # Reset / wait helpers
    # =============================
    def hard_stop(self, repeat=5, dt=0.05):
        for _ in range(repeat):
            stop(self.vel_pub)
            time.sleep(dt)

    def begin_reset_wait(self):
        self.wait_after_reset = True
        self.reset_ready_time = time.time() + 0.8
        self.sensor_fresh = {k: False for k in self.sensor_topics.keys()}
        self.raw_sonar = {k: 1.5 for k in self.sensor_topics.keys()}
        self.latest_frame = None
        self.image_fresh = False
        self.last_wait_warn_time = 0.0
        self.goal_detector.reset()
        self.waiting_for_human = False
        self.wait_context = None

    def ready_after_reset(self):
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

    def reset_world(self):
        if self.reset_client.wait_for_service(timeout_sec=1.0):
            self.reset_client.call_async(Empty.Request())

    # =============================
    # HITL wait state machine
    # =============================
    def start_human_wait(self, ctx: WaitContext):
        self.waiting_for_human = True
        self.wait_started_at = time.time()
        self.wait_context = ctx
        stop(self.vel_pub)
        self.get_logger().warning(
            f"UNKNOWN/LOW-COVERAGE STATE {ctx.state_tuple} -> waiting HITL input "
            f"[E=FWD, A=L, S=Lbit, D=Rbit, F=R]"
        )

    def handle_human_wait(self):
        if not self.waiting_for_human or self.wait_context is None:
            return

        stop(self.vel_pub)
        key = self.read_key_nonblocking()
        used_human = False
        action = None

        if key is not None:
            if key in self.key_to_action:
                action = int(self.key_to_action[key])
                used_human = True
            else:
                self.get_logger().warning(f"Ignored key: {repr(key)}")

        if action is None:
            elapsed = time.time() - self.wait_started_at
            if elapsed < self.human_wait_timeout:
                return

            # timeout fallback
            if self.fallback_to_safe_policy_after_timeout:
                action = self.safe_fallback_action(self.wait_context.state_tuple)
                self.get_logger().warning(f"HITL timeout -> safe fallback action {action}")
            elif self.fallback_to_agent_after_timeout:
                action = self.choose_agent_action(self.wait_context.state_idx)
                self.get_logger().warning(f"HITL timeout -> agent fallback action {action}")
            else:
                self.get_logger().warning("HITL timeout -> keep waiting")
                self.wait_started_at = time.time()
                return

        ctx = self.wait_context
        self.waiting_for_human = False
        self.wait_context = None

        if used_human:
            self.hitl_count_episode += 1
            self.get_logger().warning(f"HITL action executed: {action}")

        if ctx.is_initial_step:
            do_action(self.vel_pub, action)
            self.prev_state_idx = ctx.state_idx
            self.prev_action = action
            self.step_in_episode = 1
            return

        # Non-initial: resolve SARSA update with chosen next action
        self.agent.update(
            ctx.previous_state_idx,
            ctx.previous_action,
            ctx.pending_reward,
            ctx.state_idx,
            action,
        )
        do_action(self.vel_pub, action)
        self.prev_state_idx = ctx.state_idx
        self.prev_action = action
        self.step_in_episode += 1

    # =============================
    # Coverage log
    # =============================
    def append_coverage_log(self, reached_goal: bool, crash: bool, timeout: bool):
        try:
            os.makedirs(os.path.dirname(COVERAGE_LOG_PATH), exist_ok=True)
            file_exists = os.path.exists(COVERAGE_LOG_PATH)
            with open(COVERAGE_LOG_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "episode",
                        "success",
                        "crash",
                        "timeout",
                        "steps",
                        "total_reward",
                        "hitl_count",
                        "new_states_episode",
                        "visited_states_total",
                        "coverage_pct",
                    ])
                visited_total = int(np.sum(self.state_count > 0))
                writer.writerow([
                    self.episode,
                    int(reached_goal),
                    int(crash),
                    int(timeout),
                    self.step_in_episode,
                    float(self.cumulated_reward),
                    int(self.hitl_count_episode),
                    int(self.new_states_episode),
                    visited_total,
                    float(self.coverage_percent()),
                ])
        except Exception as exc:
            self.get_logger().warning(f"Failed to append coverage log: {exc}")

    # =============================
    # Main control loop
    # =============================
    def control_loop(self):
        try:
            self._control_loop_impl()
        except Exception as exc:
            self.get_logger().error(f"control_loop exception: {exc}")
            stop(self.vel_pub)

    def _control_loop_impl(self):
        if not self.ready_after_reset():
            return

        # If robot is stopped waiting for HITL input, only process that state machine.
        if self.waiting_for_human:
            self.handle_human_wait()
            return

        # 1) Current state from sonar
        state_tuple = self.discretizer.process(self.raw_sonar)
        crash = self.discretizer.is_crash(self.raw_sonar)

        # 2) Goal from ArUco + front sonar
        if crash:
            reached_goal, goal_dbg = False, {}
        else:
            reached_goal, goal_dbg = self.check_goal()

        # 3) Map current state to index
        s_idx = self.agent.state_to_index(state_tuple)

        # 4) Initial step in episode
        if self.prev_state_idx is None:
            if crash:
                self.get_logger().warning("Spawn crash detected -> reset again, not counted as episode")
                self.hard_stop()
                try:
                    self.reset_world()
                except Exception as exc:
                    self.get_logger().warning(f"Reset failed: {exc}")
                self.begin_reset_wait()
                return

            # coverage decision must use count BEFORE increment
            count_before = int(self.state_count[s_idx])
            need_hitl = self.should_request_hitl_from_count(count_before)

            self.state_count[s_idx] += 1
            if count_before == 0:
                self.new_states_episode += 1

            if need_hitl:
                self.get_logger().warning(
                    f"HITL trigger initial: state={state_tuple}, s_idx={s_idx}, count_before={count_before}"
                )
                self.start_human_wait(
                    WaitContext(
                        state_idx=s_idx,
                        state_tuple=state_tuple,
                        is_initial_step=True,
                    )
                )
                return

            action = self.choose_agent_action(s_idx)
            do_action(self.vel_pub, action)
            self.prev_state_idx = s_idx
            self.prev_action = action
            self.step_in_episode = 1
            return

        # 5) Reward + done for transition from previous state/action to current state
        timeout = self.step_in_episode >= self.max_step
        reward_value, done = compute_reward(
            state_tuple,
            crash=crash,
            reached_goal=reached_goal,
            timeout=timeout,
        )

        if self.enable_camera_bonus and not done:
            reward_value += self.compute_camera_bonus(goal_dbg, reached_goal, crash)

        self.cumulated_reward += reward_value

        # 6) Terminal update
        if done:
            self.agent.Q[self.prev_state_idx, self.prev_action] += (
                self.agent.alpha
                * (reward_value - self.agent.Q[self.prev_state_idx, self.prev_action])
            )

            rmsg = Float32()
            rmsg.data = float(reward_value)
            self.reward_pub.publish(rmsg)

            if timeout:
                self.get_logger().warning("Episode terminated cause Max Steps")

            self.finish_episode(reached_goal=reached_goal, crash=crash, timeout=timeout)
            return

        # 7) Non-terminal: coverage decision must use count BEFORE increment
        count_before = int(self.state_count[s_idx])
        need_hitl = self.should_request_hitl_from_count(count_before)

        self.state_count[s_idx] += 1
        if count_before == 0:
            self.new_states_episode += 1

        # 8) Choose next action either from HITL or agent
        if need_hitl:
            self.get_logger().warning(
                f"HITL trigger transition: state={state_tuple}, s_idx={s_idx}, count_before={count_before}"
            )
            self.start_human_wait(
                WaitContext(
                    state_idx=s_idx,
                    state_tuple=state_tuple,
                    is_initial_step=False,
                    pending_reward=float(reward_value),
                    previous_state_idx=int(self.prev_state_idx),
                    previous_action=int(self.prev_action),
                )
            )
        else:
            next_action = self.choose_agent_action(s_idx)
            self.agent.update(self.prev_state_idx, self.prev_action, reward_value, s_idx, next_action)
            do_action(self.vel_pub, next_action)
            self.prev_state_idx = s_idx
            self.prev_action = next_action
            self.step_in_episode += 1

        # 9) Publish step reward
        rmsg = Float32()
        rmsg.data = float(reward_value)
        self.reward_pub.publish(rmsg)

    def finish_episode(self, reached_goal, crash, timeout):
        self.episode += 1

        if reached_goal:
            reason = "goal"
        elif crash:
            reason = "crash"
        elif timeout:
            reason = "timeout"
        else:
            reason = "done"

        visited_total = int(np.sum(self.state_count > 0))
        coverage_pct = self.coverage_percent()
        self.get_logger().info(
            f"Episode {self.episode} finished, reason={reason}, reward={self.cumulated_reward:.2f}, "
            f"steps={self.step_in_episode}, hitl={self.hitl_count_episode}, "
            f"new_states={self.new_states_episode}, coverage={visited_total}/{self.agent.n_states} ({coverage_pct:.2f}%)"
        )

        self.hard_stop()

        total_msg = Float32()
        total_msg.data = float(self.cumulated_reward)
        self.reward_pub.publish(total_msg)

        try:
            self.reset_world()
        except Exception as exc:
            self.get_logger().warning(f"Reset failed: {exc}")

        if self.agent.save_path:
            self.agent.save_qtable(self.agent.save_path)
        self.save_state_count()
        self.append_coverage_log(reached_goal, crash, timeout)

        # Fine-tune phase: keep epsilon small but not zero.
        self.agent.epsilon = max(0.02, self.agent.epsilon * 0.999)

        try:
            self.logger_plot.log_episode(
                episode=self.episode,
                total_reward=self.cumulated_reward,
                steps=self.step_in_episode,
                hitl_count=self.hitl_count_episode,
                success=reached_goal,
                epsilon=self.agent.epsilon,
            )
        except TypeError:
            self.logger_plot.log_episode(
                self.episode,
                self.cumulated_reward,
                self.step_in_episode,
                self.hitl_count_episode,
                reached_goal,
                self.agent.epsilon,
            )

        self.prev_state_idx = None
        self.prev_action = None
        self.cumulated_reward = 0.0
        self.step_in_episode = 0
        self.hitl_count_episode = 0
        self.new_states_episode = 0
        self.goal_detector.reset()
        self.begin_reset_wait()


def main(args=None):
    rclpy.init(args=args)
    node = LearningNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt — shutting down")
    finally:
        node.restore_keyboard()
        node.logger_plot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
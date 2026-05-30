from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from geometry_msgs.msg import Pose, Quaternion, Twist
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty


@dataclass
class SpawnCandidate:
    x: float
    y: float
    z: float
    yaw: float


DEFAULT_SPAWN_CANDIDATES_JSON = json.dumps(
    [
        {"x": -0.062859, "y": 4.488687, "z": -0.045254, "yaw": 0.003262}, ## BLK Tengah
        {"x": -0.050445, "y": 3.951049, "z": -0.045254, "yaw": 0.013988}, ## BLK Kanan
        {"x": -0.066411, "y": 4.952979, "z": -0.045254, "yaw": 0.030282}, ## BLK Kiri
        {"x": 0.988868, "y": 4.997304, "z": -0.045254, "yaw": 0.042971}, ## Tengah Kiri
        {"x": 1.027265, "y": 4.474612, "z": -0.045254, "yaw": 0.055461}, ## Tengah Tengah
        {"x": 1.447848, "y": 4.501671, "z": -0.045255, "yaw": 0.065044}, ## Tengah Depan
    ]
)


def declare_random_spawn_params(node, prefix: str = "random_spawn") -> None:
    p = lambda name: f"{prefix}.{name}"

    node.declare_parameter(p("enabled"), True)
    node.declare_parameter(p("robot_name"), "jetbotV21")
    node.declare_parameter(p("reference_frame"), "world")

    node.declare_parameter(p("pause_physics_service"), "/pause_physics")
    node.declare_parameter(p("unpause_physics_service"), "/unpause_physics")
    node.declare_parameter(p("reset_world_service"), "/reset_world")
    node.declare_parameter(p("set_entity_state_service"), "/set_entity_state")

    node.declare_parameter(p("pause_physics_during_reset"), True)
    node.declare_parameter(p("use_reset_world"), True)
    node.declare_parameter(p("wait_after_reset_sec"), 0.15)
    node.declare_parameter(p("set_twist_zero"), True)

    # 1 = ganti preset setiap episode.
    # 3 = gunakan preset yang sama untuk 3 episode berturut-turut.
    node.declare_parameter(p("change_every_n_episodes"), 20)
    node.declare_parameter(p("force_new_on_pre_episode_reset"), True)

    node.declare_parameter(p("seed"), -1)
    node.declare_parameter(p("spawn_candidates_json"), DEFAULT_SPAWN_CANDIDATES_JSON)

    node.declare_parameter(p("service_timeout_sec"), 5.0)
    node.declare_parameter(p("future_timeout_sec"), 5.0)


class GazeboRandomSpawnManager:
    def __init__(self, node, prefix: str = "random_spawn", callback_group=None) -> None:
        self.node = node
        self.prefix = prefix
        self.callback_group = callback_group
        p = lambda name: f"{prefix}.{name}"

        self.enabled = bool(node.get_parameter(p("enabled")).value)
        self.robot_name = str(node.get_parameter(p("robot_name")).value)
        self.reference_frame = str(node.get_parameter(p("reference_frame")).value)

        self.pause_physics_service = str(node.get_parameter(p("pause_physics_service")).value)
        self.unpause_physics_service = str(node.get_parameter(p("unpause_physics_service")).value)
        self.reset_world_service = str(node.get_parameter(p("reset_world_service")).value)
        self.set_entity_state_service = str(node.get_parameter(p("set_entity_state_service")).value)

        self.pause_physics_during_reset = bool(node.get_parameter(p("pause_physics_during_reset")).value)
        self.use_reset_world = bool(node.get_parameter(p("use_reset_world")).value)
        self.wait_after_reset_sec = float(node.get_parameter(p("wait_after_reset_sec")).value)
        self.set_twist_zero = bool(node.get_parameter(p("set_twist_zero")).value)

        self.change_every_n_episodes = max(1, int(node.get_parameter(p("change_every_n_episodes")).value))
        self.force_new_on_pre_episode_reset = bool(node.get_parameter(p("force_new_on_pre_episode_reset")).value)

        self.service_timeout_sec = float(node.get_parameter(p("service_timeout_sec")).value)
        self.future_timeout_sec = float(node.get_parameter(p("future_timeout_sec")).value)

        seed = int(node.get_parameter(p("seed")).value)
        if seed >= 0:
            random.seed(seed)
            node.get_logger().warning(f"[random_spawn] seed={seed}")

        self.spawn_candidates = self._parse_spawn_candidates(
            str(node.get_parameter(p("spawn_candidates_json")).value)
        )

        self.pause_cli = node.create_client(
            Empty,
            self.pause_physics_service,
            callback_group=self.callback_group,
        )
        self.unpause_cli = node.create_client(
            Empty,
            self.unpause_physics_service,
            callback_group=self.callback_group,
        )
        self.reset_world_cli = node.create_client(
            Empty,
            self.reset_world_service,
            callback_group=self.callback_group,
        )
        self.set_state_cli = node.create_client(
            SetEntityState,
            self.set_entity_state_service,
            callback_group=self.callback_group,
        )

        self._current_candidate_idx: Optional[int] = None
        self._current_candidate: Optional[SpawnCandidate] = None
        self._respawn_calls = 0

        node.get_logger().warning(
            f"[random_spawn] enabled={self.enabled}, robot_name={self.robot_name}, "
            f"change_every_n_episodes={self.change_every_n_episodes}, "
            f"candidates={len(self.spawn_candidates)}"
        )

    def _parse_spawn_candidates(self, raw_json: str) -> List[SpawnCandidate]:
        data = json.loads(raw_json)
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("spawn_candidates_json must be a non-empty JSON list")

        candidates: List[SpawnCandidate] = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                cand = SpawnCandidate(
                    x=float(item["x"]),
                    y=float(item["y"]),
                    z=float(item.get("z", 0.01)),
                    yaw=float(item.get("yaw", 0.0)),
                )
            elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if len(item) < 4:
                    raise ValueError(f"candidate #{i} must contain [x, y, z, yaw]")
                cand = SpawnCandidate(
                    x=float(item[0]),
                    y=float(item[1]),
                    z=float(item[2]),
                    yaw=float(item[3]),
                )
            else:
                raise ValueError(
                    f"candidate #{i} must be dict or list/tuple, got {type(item).__name__}"
                )
            candidates.append(cand)
        return candidates

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def _wait_for_service(self, client, name: str, timeout_sec: Optional[float] = None) -> bool:
        timeout_sec = self.service_timeout_sec if timeout_sec is None else timeout_sec
        start = time.time()
        while not client.wait_for_service(timeout_sec=0.25):
            if time.time() - start >= timeout_sec:
                self.node.get_logger().warning(
                    f"[random_spawn] service {name} not available within {timeout_sec:.1f}s"
                )
                return False
        return True

    def _wait_future(self, future, timeout_sec: Optional[float] = None) -> bool:
        timeout_sec = self.future_timeout_sec if timeout_sec is None else timeout_sec
        start = time.time()
        while not future.done():
            if time.time() - start >= timeout_sec:
                return False
            time.sleep(0.02)
        return True

    def _call_empty(self, client, name: str) -> bool:
        if not self._wait_for_service(client, name):
            return False
        future = client.call_async(Empty.Request())
        if not self._wait_future(future):
            self.node.get_logger().warning(f"[random_spawn] call {name} timed out")
            return False
        return future.result() is not None

    def _choose_candidate(self, force_new_candidate: bool) -> Tuple[SpawnCandidate, int, bool]:
        self._respawn_calls += 1
        choose_new = (
            self._current_candidate is None
            or force_new_candidate
            or ((self._respawn_calls - 1) % self.change_every_n_episodes == 0)
        )

        if choose_new:
            idx_pool = list(range(len(self.spawn_candidates)))
            if self._current_candidate_idx is not None and len(idx_pool) > 1:
                try:
                    idx_pool.remove(self._current_candidate_idx)
                except ValueError:
                    pass
            idx = random.choice(idx_pool)
            self._current_candidate_idx = idx
            self._current_candidate = self.spawn_candidates[idx]
            changed = True
        else:
            idx = int(self._current_candidate_idx)
            changed = False

        return self._current_candidate, idx, changed

    def _build_entity_state(self, candidate: SpawnCandidate) -> EntityState:
        state = EntityState()
        state.name = self.robot_name
        state.reference_frame = self.reference_frame

        pose = Pose()
        pose.position.x = candidate.x
        pose.position.y = candidate.y
        pose.position.z = candidate.z
        pose.orientation = self.yaw_to_quaternion(candidate.yaw)
        state.pose = pose

        twist = Twist()
        if self.set_twist_zero:
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
        state.twist = twist
        return state

    def _set_entity_state(self, candidate: SpawnCandidate) -> bool:
        if not self._wait_for_service(self.set_state_cli, self.set_entity_state_service):
            return False
        req = SetEntityState.Request()
        req.state = self._build_entity_state(candidate)

        future = self.set_state_cli.call_async(req)
        if not self._wait_future(future):
            self.node.get_logger().warning("[random_spawn] /set_entity_state timed out")
            return False

        result = future.result()
        if result is None:
            self.node.get_logger().warning("[random_spawn] /set_entity_state returned None")
            return False
        if not bool(result.success):
            self.node.get_logger().warning(
                f"[random_spawn] /set_entity_state success=False status='{result.status_message}'"
            )
            return False
        return True

    def reset_and_respawn(self, force_new_candidate: bool = False) -> Tuple[bool, Optional[SpawnCandidate], bool]:
        if not self.enabled:
            ok = True
            if self.use_reset_world:
                ok = self._call_empty(self.reset_world_cli, self.reset_world_service)
            return ok, None, False

        candidate, idx, changed = self._choose_candidate(force_new_candidate)

        try:
            if self.pause_physics_during_reset:
                ok = self._call_empty(self.pause_cli, self.pause_physics_service)
                if not ok:
                    return False, candidate, changed

            if self.use_reset_world:
                ok = self._call_empty(self.reset_world_cli, self.reset_world_service)
                if not ok:
                    return False, candidate, changed
                if self.wait_after_reset_sec > 0.0:
                    time.sleep(self.wait_after_reset_sec)

            ok = self._set_entity_state(candidate)
            if not ok:
                return False, candidate, changed

            self.node.get_logger().warning(
                f"[random_spawn] respawn -> idx={idx}, changed={changed}, "
                f"x={candidate.x:.3f}, y={candidate.y:.3f}, z={candidate.z:.3f}, yaw={candidate.yaw:.3f}"
            )
            return True, candidate, changed

        finally:
            if self.pause_physics_during_reset:
                self._call_empty(self.unpause_cli, self.unpause_physics_service)


def create_random_spawn_manager(node, prefix: str = "random_spawn", callback_group=None):
    return GazeboRandomSpawnManager(node=node, prefix=prefix, callback_group=callback_group)
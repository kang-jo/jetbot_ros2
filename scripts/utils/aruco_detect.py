#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GoalArucoConfig:
    goal_id: int = 23
    dictionary_name: str = "DICT_6X6_250"

    # All distances are in meters.
    front_threshold_m: float = 0.25
    min_streak: int = 5

    use_area_check: bool = False
    min_area_ratio: float = 0.02

    use_center_check: bool = False
    center_tolerance_ratio: float = 0.25

    debug: bool = False


class GoalDetectorAruco:
    """Goal detector based on ArUco marker + distance validation."""

    def __init__(self, config: GoalArucoConfig):
        self.cfg = config
        self.detect_streak = 0

        self.aruco = cv2.aruco
        self.dictionary = self._get_dictionary(config.dictionary_name)
        self.parameters = self._create_detector_parameters()

    def reset(self) -> None:
        self.detect_streak = 0

    def _get_dictionary(self, dictionary_name: str):
        if not hasattr(self.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dict_id = getattr(self.aruco, dictionary_name)
        return self.aruco.getPredefinedDictionary(dict_id)

    def _create_detector_parameters(self):
        if hasattr(self.aruco, "DetectorParameters"):
            return self.aruco.DetectorParameters()
        return self.aruco.DetectorParameters_create()

    def _detect_markers(self, gray: np.ndarray):
        if hasattr(self.aruco, "ArucoDetector"):
            detector = self.aruco.ArucoDetector(self.dictionary, self.parameters)
            return detector.detectMarkers(gray)
        return self.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

    def detect_goal_marker(self, frame: Optional[np.ndarray]) -> Tuple[bool, Dict[str, Any]]:
        if frame is None:
            return False, {"goal_id_found": False, "ids": []}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detect_markers(gray)

        base_info: Dict[str, Any] = {
            "found_any": ids is not None and len(ids) > 0,
            "goal_id_found": False,
            "ids": [] if ids is None else ids.flatten().tolist(),
            "rejected_count": 0 if rejected is None else len(rejected),
        }

        if ids is None or len(ids) == 0:
            return False, base_info

        ids_flat = ids.flatten().tolist()
        if self.cfg.goal_id not in ids_flat:
            return False, base_info

        idx = ids_flat.index(self.cfg.goal_id)
        pts = corners[idx][0].astype(np.float32)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        marker_area = float(cv2.contourArea(pts))
        area_ratio = marker_area / frame_area if frame_area > 0 else 0.0

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        center_offset_ratio = abs(cx - (w / 2.0)) / float(w) if w > 0 else 1.0

        base_info.update(
            {
                "goal_id_found": True,
                "target_id": self.cfg.goal_id,
                "corners": pts,
                "center_x": cx,
                "center_y": cy,
                "center_offset_ratio": center_offset_ratio,
                "bbox_x_min": float(np.min(pts[:, 0])),
                "bbox_y_min": float(np.min(pts[:, 1])),
                "bbox_x_max": float(np.max(pts[:, 0])),
                "bbox_y_max": float(np.max(pts[:, 1])),
                "marker_area_px": marker_area,
                "area_ratio": area_ratio,
                "frame_width": w,
                "frame_height": h,
            }
        )
        return True, base_info

    def _check_front_distance(self, front_distance_m: Optional[float]) -> bool:
        if front_distance_m is None:
            return False
        if front_distance_m <= 0.0:
            return False
        return front_distance_m <= self.cfg.front_threshold_m

    def _check_area(self, info: Dict[str, Any]) -> bool:
        if not self.cfg.use_area_check:
            return True
        return float(info.get("area_ratio", 0.0)) >= self.cfg.min_area_ratio

    def _check_center(self, info: Dict[str, Any]) -> bool:
        if not self.cfg.use_center_check:
            return True
        return float(info.get("center_offset_ratio", 1.0)) <= self.cfg.center_tolerance_ratio

    def update(self, frame: Optional[np.ndarray], front_distance_m: Optional[float]) -> Tuple[bool, Dict[str, Any]]:
        found, info = self.detect_goal_marker(frame)

        info["front_distance_m"] = None if front_distance_m is None else float(front_distance_m)
        info["front_ok"] = False
        info["area_ok"] = False
        info["center_ok"] = False
        info["valid"] = False
        info["streak"] = self.detect_streak
        info["goal_reached"] = False

        if not found:
            self.detect_streak = 0
            info["streak"] = 0
            return False, info

        front_ok = self._check_front_distance(front_distance_m)
        area_ok = self._check_area(info)
        center_ok = self._check_center(info)
        valid = bool(info.get("goal_id_found", False) and front_ok and area_ok and center_ok)

        if valid:
            self.detect_streak += 1
        else:
            self.detect_streak = 0

        goal_reached = self.detect_streak >= self.cfg.min_streak

        info["front_ok"] = front_ok
        info["area_ok"] = area_ok
        info["center_ok"] = center_ok
        info["valid"] = valid
        info["streak"] = self.detect_streak
        info["goal_reached"] = goal_reached
        return goal_reached, info

    def draw_debug(self, frame: np.ndarray, info: Optional[Dict[str, Any]]) -> np.ndarray:
        vis = frame.copy()
        if info is None:
            return vis

        line1 = (
            f"front={info.get('front_distance_m', None)}m "
            f"front_ok={info.get('front_ok', False)} "
            f"streak={info.get('streak', 0)} "
            f"goal={info.get('goal_reached', False)}"
        )
        cv2.putText(vis, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if not info.get("goal_id_found", False):
            txt = f"goal_id={self.cfg.goal_id} not found, seen={info.get('ids', [])}"
            cv2.putText(vis, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            return vis

        pts = info["corners"].astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
        cx = int(info["center_x"])
        cy = int(info["center_y"])
        cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)

        line2 = (
            f"id={self.cfg.goal_id} area={info.get('area_ratio', 0.0):.4f} "
            f"center_off={info.get('center_offset_ratio', 0.0):.3f} "
            f"valid={info.get('valid', False)}"
        )
        cv2.putText(vis, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if self.cfg.use_center_check:
            w = int(info["frame_width"])
            h = int(info["frame_height"])
            mid = w / 2.0
            tol = self.cfg.center_tolerance_ratio * w
            x1 = int(mid - tol)
            x2 = int(mid + tol)
            cv2.line(vis, (x1, 0), (x1, h), (255, 0, 0), 1)
            cv2.line(vis, (x2, 0), (x2, h), (255, 0, 0), 1)
            cv2.line(vis, (int(mid), 0), (int(mid), h), (255, 255, 0), 1)

        return vis
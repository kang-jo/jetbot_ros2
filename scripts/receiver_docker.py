#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone ArUco area-ratio calibrator from chunked UDP JPEG frames.

This script is intended to run INSIDE Docker without cv_bridge and without ROS Image.
It follows the same chunked UDP frame format used by the previous Docker receiver:

    MAGIC = b"JBF1"
    HEADER_STRUCT = struct.Struct("!4sIHHH")

Typical flow:
    Host camera sender  ->  chunked JPEG UDP packets  ->  this calibrator in Docker

Features:
- Reassembles JPEG frame chunks from UDP.
- Decodes frames with cv2.imdecode.
- Detects ArUco marker.
- Logs area_ratio, zone, FAR/NEAR state, cx ratio, min/avg/max area.
- Draws LEFT/CENTER/RIGHT zones and marker overlay.
- Optional cv2.imshow if X11 works.
- Optional periodic save of debug JPGs if X11 does not work.
- Optional status JSON sender for compatibility/debugging.

Example:
    python3 aruco_udp_area_calibrator_chunked.py \
      --frame-bind-host 0.0.0.0 \
      --frame-port 5020 \
      --target-id 23 \
      --near-area-ratio 0.002 \
      --left-boundary-ratio 0.35 \
      --right-boundary-ratio 0.65 \
      --log-every 0.5 \
      --save-debug-dir /tmp/aruco_calib_debug \
      --save-every 30

If X11 works:
    python3 aruco_udp_area_calibrator_chunked.py --show
"""

import argparse
import json
import os
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


MAGIC = b"JBF1"
HEADER_STRUCT = struct.Struct("!4sIHHH")
HEADER_SIZE = HEADER_STRUCT.size
MAX_FRAME_AGE_S = 1.0

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


@dataclass
class ArucoCalibConfig:
    target_id: int = 23
    dictionary_name: str = "DICT_6X6_250"
    near_area_ratio: float = 0.002
    left_boundary_ratio: float = 0.35
    right_boundary_ratio: float = 0.65


class FrameReassembler:
    def __init__(self, max_frame_age_s: float = MAX_FRAME_AGE_S):
        self.frames: Dict[int, Dict[str, Any]] = {}
        self.max_frame_age_s = max_frame_age_s
        self.bad_packets = 0
        self.completed_frames = 0

    def add_packet(self, data: bytes) -> Optional[Tuple[int, bytes]]:
        if len(data) < HEADER_SIZE:
            self.bad_packets += 1
            return None

        try:
            magic, frame_id, chunk_idx, chunk_total, payload_len = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        except struct.error:
            self.bad_packets += 1
            return None

        if magic != MAGIC:
            self.bad_packets += 1
            return None

        if chunk_total <= 0 or chunk_idx >= chunk_total:
            self.bad_packets += 1
            return None

        payload = data[HEADER_SIZE:]
        if len(payload) != payload_len:
            self.bad_packets += 1
            return None

        now = time.time()
        old_ids = [
            fid for fid, rec in self.frames.items()
            if now - float(rec["t"]) > self.max_frame_age_s
        ]
        for fid in old_ids:
            self.frames.pop(fid, None)

        rec = self.frames.get(frame_id)
        if rec is None:
            rec = {"t": now, "total": int(chunk_total), "chunks": {}}
            self.frames[frame_id] = rec

        # If a stale/reused frame_id somehow has different total, reset it.
        if int(rec["total"]) != int(chunk_total):
            rec = {"t": now, "total": int(chunk_total), "chunks": {}}
            self.frames[frame_id] = rec

        rec["chunks"][int(chunk_idx)] = payload

        if len(rec["chunks"]) == int(rec["total"]):
            try:
                jpg = b"".join(rec["chunks"][i] for i in range(int(rec["total"])))
            except KeyError:
                return None
            self.frames.pop(frame_id, None)
            self.completed_frames += 1
            return int(frame_id), jpg

        return None


class ArucoAreaCalibrator:
    def __init__(self, cfg: ArucoCalibConfig):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco tidak tersedia di environment ini. Cek dengan: "
                "python3 -c \"import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))\""
            )

        if not (0.0 < cfg.left_boundary_ratio < cfg.right_boundary_ratio < 1.0):
            raise ValueError(
                "Boundary kamera harus 0.0 < left_boundary_ratio < right_boundary_ratio < 1.0"
            )

        self.cfg = cfg
        self.aruco = cv2.aruco
        self.dictionary = self._get_dictionary(cfg.dictionary_name)
        self.parameters = self._create_detector_parameters()
        self.area_history: List[float] = []
        self.last_info: Dict[str, Any] = {}

    def _get_dictionary(self, dictionary_name: str):
        if not hasattr(self.aruco, dictionary_name):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        return self.aruco.getPredefinedDictionary(getattr(self.aruco, dictionary_name))

    def _create_detector_parameters(self):
        if hasattr(self.aruco, "DetectorParameters"):
            return self.aruco.DetectorParameters()
        return self.aruco.DetectorParameters_create()

    def _detect_markers(self, gray: np.ndarray):
        if hasattr(self.aruco, "ArucoDetector"):
            detector = self.aruco.ArucoDetector(self.dictionary, self.parameters)
            return detector.detectMarkers(gray)
        return self.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

    def classify_camera_state(self, cx_ratio: float, area_ratio: float) -> Tuple[int, str, str]:
        if cx_ratio < self.cfg.left_boundary_ratio:
            zone = "LEFT"
        elif cx_ratio > self.cfg.right_boundary_ratio:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        dist = "NEAR" if area_ratio >= self.cfg.near_area_ratio else "FAR"

        if zone == "LEFT" and dist == "FAR":
            return CAM_LEFT_FAR, zone, dist
        if zone == "CENTER" and dist == "FAR":
            return CAM_CENTER_FAR, zone, dist
        if zone == "RIGHT" and dist == "FAR":
            return CAM_RIGHT_FAR, zone, dist
        if zone == "LEFT" and dist == "NEAR":
            return CAM_LEFT_NEAR, zone, dist
        if zone == "CENTER" and dist == "NEAR":
            return CAM_CENTER_NEAR, zone, dist
        if zone == "RIGHT" and dist == "NEAR":
            return CAM_RIGHT_NEAR, zone, dist
        return CAM_NONE, "NONE", "FAR"

    def detect_and_draw(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        h, w = frame.shape[:2]
        vis = frame.copy()
        self._draw_zones(vis)

        info: Dict[str, Any] = {
            "timestamp": time.time(),
            "target_id": int(self.cfg.target_id),
            "found_any": False,
            "goal_id_found": False,
            "ids": [],
            "rejected_count": 0,
            "frame_width": int(w),
            "frame_height": int(h),
            "area_ratio": 0.0,
            "marker_area_px": 0.0,
            "cx_ratio": None,
            "center_x": None,
            "center_y": None,
            "zone": "NONE",
            "distance_label": "FAR",
            "camera_state": CAM_NONE,
            "camera_state_name": CAMERA_STATE_NAMES[CAM_NONE],
            "near_threshold": float(self.cfg.near_area_ratio),
            "left_boundary_ratio": float(self.cfg.left_boundary_ratio),
            "right_boundary_ratio": float(self.cfg.right_boundary_ratio),
        }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detect_markers(gray)
        info["found_any"] = bool(ids is not None and len(ids) > 0)
        info["ids"] = [] if ids is None else [int(x) for x in ids.flatten().tolist()]
        info["rejected_count"] = 0 if rejected is None else int(len(rejected))

        try:
            if ids is not None and len(ids) > 0:
                self.aruco.drawDetectedMarkers(vis, corners, ids)
        except Exception:
            pass

        if ids is None or len(ids) == 0:
            self._draw_text(vis, info)
            self.last_info = info
            return info, vis

        ids_flat = ids.flatten().tolist()
        if self.cfg.target_id not in ids_flat:
            self._draw_text(vis, info)
            self.last_info = info
            return info, vis

        idx = ids_flat.index(self.cfg.target_id)
        pts = corners[idx][0].astype(np.float32)
        frame_area = float(h * w)
        marker_area = float(cv2.contourArea(pts))
        area_ratio = marker_area / frame_area if frame_area > 0 else 0.0
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        cx_ratio = cx / float(w) if w > 0 else 0.0

        cam_state, zone, dist = self.classify_camera_state(cx_ratio, area_ratio)
        self.area_history.append(area_ratio)

        pts_i = pts.astype(np.int32)
        cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 5, (0, 255, 0), -1)

        info.update(
            {
                "goal_id_found": True,
                "center_x": cx,
                "center_y": cy,
                "cx_ratio": cx_ratio,
                "bbox_x_min": float(np.min(pts[:, 0])),
                "bbox_y_min": float(np.min(pts[:, 1])),
                "bbox_x_max": float(np.max(pts[:, 0])),
                "bbox_y_max": float(np.max(pts[:, 1])),
                "marker_area_px": marker_area,
                "area_ratio": area_ratio,
                "zone": zone,
                "distance_label": dist,
                "camera_state": int(cam_state),
                "camera_state_name": CAMERA_STATE_NAMES[int(cam_state)],
                "area_min": float(np.min(self.area_history)),
                "area_avg": float(np.mean(self.area_history)),
                "area_max": float(np.max(self.area_history)),
                "sample_count": int(len(self.area_history)),
            }
        )

        self._draw_text(vis, info)
        self.last_info = info
        return info, vis

    def _draw_zones(self, vis: np.ndarray) -> None:
        h, w = vis.shape[:2]
        x_left = int(self.cfg.left_boundary_ratio * w)
        x_right = int(self.cfg.right_boundary_ratio * w)
        x_mid = int(0.5 * w)

        cv2.line(vis, (x_left, 0), (x_left, h), (255, 0, 0), 2)
        cv2.line(vis, (x_right, 0), (x_right, h), (255, 0, 0), 2)
        cv2.line(vis, (x_mid, 0), (x_mid, h), (255, 255, 0), 1)

        cv2.putText(vis, "LEFT", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(vis, "CENTER", (max(x_left + 10, 10), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(vis, "RIGHT", (max(x_right + 10, 10), h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    def _draw_text(self, vis: np.ndarray, info: Dict[str, Any]) -> None:
        line1 = f"id={self.cfg.target_id} found={info.get('goal_id_found')} seen={info.get('ids')}"
        line2 = (
            f"state={info.get('camera_state')} {info.get('camera_state_name')} "
            f"zone={info.get('zone')} {info.get('distance_label')} "
            f"area={float(info.get('area_ratio', 0.0)):.6f} near_th={self.cfg.near_area_ratio:.6f}"
        )
        cx_ratio = info.get("cx_ratio")
        if cx_ratio is None:
            line3 = (
                f"bounds L<{self.cfg.left_boundary_ratio:.2f} "
                f"C {self.cfg.left_boundary_ratio:.2f}-{self.cfg.right_boundary_ratio:.2f} "
                f"R>{self.cfg.right_boundary_ratio:.2f}"
            )
        else:
            line3 = (
                f"cx_ratio={float(cx_ratio):.3f} "
                f"min/avg/max={float(info.get('area_min', 0.0)):.6f}/"
                f"{float(info.get('area_avg', 0.0)):.6f}/"
                f"{float(info.get('area_max', 0.0)):.6f} "
                f"n={int(info.get('sample_count', 0))}"
            )

        cv2.putText(vis, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(vis, line3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

    def reset_stats(self) -> None:
        self.area_history.clear()


class OptionalStatusSender:
    def __init__(self, enabled: bool, host: str, port: int):
        self.enabled = bool(enabled)
        self.addr = (host, int(port))
        self.sock: Optional[socket.socket] = None
        if self.enabled:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, info: Dict[str, Any]) -> None:
        if not self.enabled or self.sock is None:
            return
        try:
            payload = json.dumps(info, separators=(",", ":")).encode("utf-8")
            self.sock.sendto(payload, self.addr)
        except Exception as exc:
            print(f"[WARN] Failed to send status JSON: {exc}")

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone chunked-UDP ArUco area-ratio calibrator without cv_bridge."
    )
    p.add_argument("--frame-bind-host", default="0.0.0.0")
    p.add_argument("--frame-port", type=int, default=5020)

    p.add_argument("--target-id", type=int, default=23)
    p.add_argument("--dictionary", default="DICT_6X6_250")
    p.add_argument("--near-area-ratio", type=float, default=0.002)
    p.add_argument("--left-boundary-ratio", type=float, default=0.35)
    p.add_argument("--right-boundary-ratio", type=float, default=0.65)

    p.add_argument("--show", action="store_true", help="Show debug window if X11 works.")
    p.add_argument("--window-name", default="ArUco UDP Area Calibrator")
    p.add_argument("--log-every", type=float, default=0.5, help="Seconds between logs.")

    p.add_argument("--save-debug-dir", default="", help="Directory to save debug JPEGs. Empty disables saving.")
    p.add_argument("--save-every", type=int, default=30, help="Save one debug image every N decoded frames.")

    p.add_argument("--send-status", action="store_true", help="Optionally send status JSON by UDP.")
    p.add_argument("--status-host", default="127.0.0.1")
    p.add_argument("--status-port", type=int, default=5010)
    return p.parse_args()


def main():
    args = parse_args()

    print("[INFO] Standalone ArUco UDP area calibrator")
    print(f"[INFO] cv2 version: {cv2.__version__}")
    print(f"[INFO] has cv2.aruco: {hasattr(cv2, 'aruco')}")
    print("[INFO] No ROS Image and no cv_bridge are used.")

    calibrator = ArucoAreaCalibrator(
        ArucoCalibConfig(
            target_id=args.target_id,
            dictionary_name=args.dictionary,
            near_area_ratio=args.near_area_ratio,
            left_boundary_ratio=args.left_boundary_ratio,
            right_boundary_ratio=args.right_boundary_ratio,
        )
    )

    if args.save_debug_dir:
        os.makedirs(args.save_debug_dir, exist_ok=True)
        print(f"[INFO] Saving debug images to: {args.save_debug_dir}")

    status_sender = OptionalStatusSender(args.send_status, args.status_host, args.status_port)

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.frame_bind_host, args.frame_port))
    rx.settimeout(1.0)

    reasm = FrameReassembler()
    decoded_count = 0
    start = time.time()
    last_log = 0.0
    last_packet_time = time.time()

    print(f"[INFO] Listening chunked JPEG frames on udp://{args.frame_bind_host}:{args.frame_port}")
    print(
        f"[INFO] target_id={args.target_id}, near_th={args.near_area_ratio}, "
        f"CENTER={args.left_boundary_ratio:.2f}-{args.right_boundary_ratio:.2f}"
    )

    try:
        while True:
            try:
                data, _ = rx.recvfrom(65535)
                last_packet_time = time.time()
            except socket.timeout:
                dt = time.time() - last_packet_time
                print(f"[WARN] No UDP frame packet for {dt:.1f}s on port {args.frame_port}")
                continue

            ready = reasm.add_packet(data)
            if ready is None:
                continue

            frame_id, jpg = ready
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Failed to decode frame_id={frame_id}")
                continue

            decoded_count += 1
            info, vis = calibrator.detect_and_draw(frame)
            info["frame_id"] = int(frame_id)
            info["rx_time"] = time.time()
            info["decoded_count"] = int(decoded_count)
            info["fps_avg"] = float(decoded_count / max(time.time() - start, 1e-6))
            status_sender.send(info)

            now = time.time()
            if now - last_log >= args.log_every:
                print(
                    f"[ARUCO-CALIB] frame={frame_id} fps_avg={info['fps_avg']:.1f} "
                    f"found={info.get('goal_id_found')} ids={info.get('ids')} "
                    f"state={info.get('camera_state')} {info.get('camera_state_name')} "
                    f"zone={info.get('zone')} {info.get('distance_label')} "
                    f"area={float(info.get('area_ratio', 0.0)):.6f} "
                    f"near_th={args.near_area_ratio:.6f} "
                    f"cx_ratio={info.get('cx_ratio')} "
                    f"min/avg/max="
                    f"{float(info.get('area_min', 0.0)):.6f}/"
                    f"{float(info.get('area_avg', 0.0)):.6f}/"
                    f"{float(info.get('area_max', 0.0)):.6f} "
                    f"bad_packets={reasm.bad_packets} partial={len(reasm.frames)}"
                )
                last_log = now

            if args.save_debug_dir and args.save_every > 0 and decoded_count % args.save_every == 0:
                out_path = os.path.join(args.save_debug_dir, f"aruco_calib_{decoded_count:06d}_fid{frame_id}.jpg")
                cv2.imwrite(out_path, vis)

            if args.show:
                cv2.imshow(args.window_name, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    calibrator.reset_stats()
                    print("[INFO] Area statistics reset")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        rx.close()
        status_sender.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

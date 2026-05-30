#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker-side ArUco detector from UDP JPEG frames.

Run this INSIDE Docker, assuming:
- Docker has cv2.aruco available.
- Docker uses --network host.
- Host sends JPEG frames with camera_udp_frame_sender_host.py to UDP port 5020.

This script:
1) receives JPEG frame chunks over UDP,
2) reconstructs + decodes frame,
3) detects ArUco in Docker,
4) sends compact JSON status to control node UDP port 5010.

It intentionally does NOT use GStreamer/camera in Docker.
"""

import argparse
import json
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


MAGIC = b"JBF1"
HEADER_STRUCT = struct.Struct("!4sIHHH")
HEADER_SIZE = HEADER_STRUCT.size
MAX_FRAME_AGE_S = 1.0


@dataclass
class ArucoVisionConfig:
    goal_id: int = 23
    dictionary_name: str = "DICT_6X6_250"
    use_area_check: bool = True
    min_area_ratio: float = 0.01
    use_center_check: bool = True
    center_tolerance_ratio: float = 0.18


class VisionArucoDetector:
    def __init__(self, cfg: ArucoVisionConfig):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco tidak tersedia di Docker. Cek dengan: "
                "python3 -c \"import cv2; print(hasattr(cv2, 'aruco'))\""
            )
        self.cfg = cfg
        self.aruco = cv2.aruco
        self.dictionary = self._get_dictionary(cfg.dictionary_name)
        self.parameters = self._create_detector_parameters()

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

    def detect(self, frame: Optional[np.ndarray]) -> Tuple[Dict[str, Any], np.ndarray]:
        now = time.time()
        if frame is None:
            return {
                "timestamp": now,
                "found_any": False,
                "goal_id_found": False,
                "ids": [],
                "valid_vision": False,
                "area_ok": False,
                "center_ok": False,
            }, frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detect_markers(gray)

        info: Dict[str, Any] = {
            "timestamp": now,
            "found_any": ids is not None and len(ids) > 0,
            "goal_id_found": False,
            "ids": [] if ids is None else ids.flatten().tolist(),
            "rejected_count": 0 if rejected is None else len(rejected),
            "target_id": self.cfg.goal_id,
            "valid_vision": False,
            "area_ok": False,
            "center_ok": False,
            "area_ratio": 0.0,
            "center_offset_ratio": 1.0,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
        }

        vis = frame.copy()

        if ids is None or len(ids) == 0:
            self._draw_info(vis, info)
            return info, vis

        ids_flat = ids.flatten().tolist()
        # Draw all detected markers if API exists.
        try:
            self.aruco.drawDetectedMarkers(vis, corners, ids)
        except Exception:
            pass

        if self.cfg.goal_id not in ids_flat:
            self._draw_info(vis, info)
            return info, vis

        idx = ids_flat.index(self.cfg.goal_id)
        pts = corners[idx][0].astype(np.float32)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        marker_area = float(cv2.contourArea(pts))
        area_ratio = marker_area / frame_area if frame_area > 0 else 0.0

        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        center_offset_ratio = abs(cx - (w / 2.0)) / float(w) if w > 0 else 1.0

        area_ok = True
        if self.cfg.use_area_check:
            area_ok = area_ratio >= self.cfg.min_area_ratio

        center_ok = True
        if self.cfg.use_center_check:
            center_ok = center_offset_ratio <= self.cfg.center_tolerance_ratio

        valid_vision = bool(area_ok and center_ok)

        info.update(
            {
                "goal_id_found": True,
                "center_x": cx,
                "center_y": cy,
                "center_offset_ratio": center_offset_ratio,
                "bbox_x_min": float(np.min(pts[:, 0])),
                "bbox_y_min": float(np.min(pts[:, 1])),
                "bbox_x_max": float(np.max(pts[:, 0])),
                "bbox_y_max": float(np.max(pts[:, 1])),
                "marker_area_px": marker_area,
                "area_ratio": area_ratio,
                "area_ok": bool(area_ok),
                "center_ok": bool(center_ok),
                "valid_vision": bool(valid_vision),
            }
        )

        # Debug overlay.
        pts_i = pts.astype(np.int32)
        cv2.polylines(vis, [pts_i], True, (0, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)

        if self.cfg.use_center_check:
            mid = w / 2.0
            tol = self.cfg.center_tolerance_ratio * w
            cv2.line(vis, (int(mid - tol), 0), (int(mid - tol), h), (255, 0, 0), 1)
            cv2.line(vis, (int(mid + tol), 0), (int(mid + tol), h), (255, 0, 0), 1)
            cv2.line(vis, (int(mid), 0), (int(mid), h), (255, 255, 0), 1)

        self._draw_info(vis, info)
        return info, vis

    def _draw_info(self, vis: np.ndarray, info: Dict[str, Any]) -> None:
        txt1 = (
            f"id={self.cfg.goal_id} found={info.get('goal_id_found', False)} "
            f"seen={info.get('ids', [])}"
        )
        txt2 = (
            f"area={info.get('area_ratio', 0.0):.4f} area_ok={info.get('area_ok', False)} "
            f"center_off={info.get('center_offset_ratio', 1.0):.3f} "
            f"center_ok={info.get('center_ok', False)} valid={info.get('valid_vision', False)}"
        )
        cv2.putText(vis, txt1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, txt2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)


class FrameReassembler:
    def __init__(self):
        self.frames: Dict[int, Dict[str, Any]] = {}

    def add_packet(self, data: bytes) -> Optional[Tuple[int, bytes]]:
        if len(data) < HEADER_SIZE:
            return None

        magic, frame_id, chunk_idx, chunk_total, payload_len = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        if magic != MAGIC:
            return None

        payload = data[HEADER_SIZE:]
        if len(payload) != payload_len:
            return None

        now = time.time()
        # Drop old partial frames.
        old_ids = [
            fid for fid, rec in self.frames.items()
            if now - rec["t"] > MAX_FRAME_AGE_S
        ]
        for fid in old_ids:
            self.frames.pop(fid, None)

        rec = self.frames.get(frame_id)
        if rec is None:
            rec = {
                "t": now,
                "total": chunk_total,
                "chunks": {},
            }
            self.frames[frame_id] = rec

        rec["chunks"][chunk_idx] = payload

        if len(rec["chunks"]) == rec["total"]:
            jpg = b"".join(rec["chunks"][i] for i in range(rec["total"]))
            self.frames.pop(frame_id, None)
            return frame_id, jpg

        return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frame-bind-host", default="0.0.0.0")
    p.add_argument("--frame-port", type=int, default=5020)

    p.add_argument("--status-host", default="127.0.0.1", help="Control node UDP host.")
    p.add_argument("--status-port", type=int, default=5010, help="Control node UDP port.")

    p.add_argument("--target-id", type=int, default=23)
    p.add_argument("--dictionary", default="DICT_6X6_250")
    p.add_argument("--min-area-ratio", type=float, default=0.01)
    p.add_argument("--center-tolerance-ratio", type=float, default=0.18)

    p.add_argument("--show", action="store_true", help="Show debug image inside Docker if X11 works.")
    p.add_argument("--log-every", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()

    print("[INFO] cv2:", cv2.__version__)
    print("[INFO] has cv2.aruco:", hasattr(cv2, "aruco"))

    detector = VisionArucoDetector(
        ArucoVisionConfig(
            goal_id=args.target_id,
            dictionary_name=args.dictionary,
            use_area_check=True,
            min_area_ratio=args.min_area_ratio,
            use_center_check=True,
            center_tolerance_ratio=args.center_tolerance_ratio,
        )
    )

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rx.bind((args.frame_bind_host, args.frame_port))

    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    status_addr = (args.status_host, args.status_port)

    reasm = FrameReassembler()
    last_log = 0.0
    decoded_count = 0
    start = time.time()

    print(f"[INFO] Listening JPEG frames on udp://{args.frame_bind_host}:{args.frame_port}")
    print(f"[INFO] Sending ArUco status to udp://{args.status_host}:{args.status_port}")

    try:
        while True:
            data, _ = rx.recvfrom(65535)
            ready = reasm.add_packet(data)
            if ready is None:
                continue

            frame_id, jpg = ready
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            decoded_count += 1
            info, vis = detector.detect(frame)
            info["frame_id"] = int(frame_id)
            info["rx_time"] = time.time()

            # Keep payload JSON-friendly and small.
            payload = json.dumps(info, separators=(",", ":")).encode("utf-8")
            tx.sendto(payload, status_addr)

            now = time.time()
            if now - last_log >= args.log_every:
                avg_fps = decoded_count / max(now - start, 1e-6)
                print(
                    f"[ARUCO-UDP] frame={frame_id} fps_avg={avg_fps:.1f} "
                    f"found={info.get('goal_id_found')} "
                    f"ids={info.get('ids')} "
                    f"area={info.get('area_ratio', 0.0):.4f} "
                    f"area_ok={info.get('area_ok')} "
                    f"center_ok={info.get('center_ok')} "
                    f"valid_vision={info.get('valid_vision')}"
                )
                last_log = now

            if args.show:
                cv2.imshow("DOCKER ArUco UDP Receiver", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        rx.close()
        tx.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

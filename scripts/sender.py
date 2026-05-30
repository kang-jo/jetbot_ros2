#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import socket
import struct
import time
from typing import Optional, Tuple

import cv2

MAGIC = b"JBF1"
HEADER_STRUCT = struct.Struct("!4sIHHH")
MAX_CHUNK_PAYLOAD = 60000


def build_gst_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jetson camera JPEG chunked UDP sender compatible with JBF1 receiver."
    )

    p.add_argument("--udp-host", "--host", dest="udp_host", default="127.0.0.1",
                   help="Receiver IP. Use 127.0.0.1 when Docker uses --network host.")
    p.add_argument("--udp-port", "--port", dest="udp_port", type=int, default=5020,
                   help="Receiver UDP port for JPEG frames.")

    p.add_argument("--sensor-id", "--camera", dest="sensor_id", type=int, default=0,
                   help="CSI sensor-id, or USB camera index when --source usb is used.")
    p.add_argument("--source", choices=["csi", "usb"], default="csi",
                   help="Camera source type. Default csi uses nvarguscamerasrc.")
    p.add_argument("--width", type=int, default=1280, help="Capture width.")
    p.add_argument("--height", type=int, default=720, help="Capture height.")
    p.add_argument("--fps", type=int, default=30, help="Capture FPS.")
    p.add_argument("--flip-method", type=int, default=0, help="Jetson nvvidconv flip-method.")

    p.add_argument("--send-width", type=int, default=640,
                   help="Resize width before JPEG. 0 = no resize.")
    p.add_argument("--send-height", type=int, default=360,
                   help="Resize height before JPEG. 0 = no resize.")
    p.add_argument("--jpeg-quality", "--quality", dest="jpeg_quality", type=int, default=55,
                   help="JPEG quality 1-100.")
    p.add_argument("--send-fps", type=float, default=15.0, help="Limit UDP send FPS.")

    p.add_argument("--show", action="store_true", help="Show local camera preview on host.")
    p.add_argument("--save-on-s", action="store_true", help="Press s to save current frame if --show is active.")
    p.add_argument("--max-chunk-payload", type=int, default=MAX_CHUNK_PAYLOAD,
                   help="UDP payload bytes per chunk. Default 60000.")
    return p.parse_args()


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.source == "csi":
        gst = build_gst_pipeline(
            sensor_id=args.sensor_id,
            width=args.width,
            height=args.height,
            fps=args.fps,
            flip_method=args.flip_method,
        )
        print("[INFO] Opening CSI camera with GStreamer pipeline:")
        print(gst)
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        print(f"[INFO] Opening USB/V4L2 camera index {args.sensor_id}")
        cap = cv2.VideoCapture(args.sensor_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    return cap


def send_jpeg_chunked(
    sock: socket.socket,
    addr: Tuple[str, int],
    frame_id: int,
    jpg_bytes: bytes,
    max_chunk_payload: int,
) -> int:
    if max_chunk_payload <= 0 or max_chunk_payload > 65535:
        raise ValueError("max_chunk_payload must be between 1 and 65535")

    total = (len(jpg_bytes) + max_chunk_payload - 1) // max_chunk_payload
    if total > 65535:
        raise ValueError(f"JPEG too large: {len(jpg_bytes)} bytes -> {total} chunks")

    for idx in range(total):
        start = idx * max_chunk_payload
        payload = jpg_bytes[start:start + max_chunk_payload]
        header = HEADER_STRUCT.pack(MAGIC, frame_id, idx, total, len(payload))
        sock.sendto(header + payload, addr)
    return total


def main() -> None:
    args = parse_args()

    cap = open_camera(args)
    if not cap.isOpened():
        raise RuntimeError(
            "Kamera gagal dibuka. Jika memakai CSI Jetson, coba: "
            "sudo systemctl restart nvargus-daemon. "
            "Jika memakai USB camera, jalankan dengan --source usb."
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.udp_host, args.udp_port)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    frame_id = 0
    last_send = 0.0
    send_period = 1.0 / max(args.send_fps, 0.1)
    last_log = 0.0
    sent_count = 0
    start_time = time.time()
    last_jpeg_len: Optional[int] = None
    last_chunks: Optional[int] = None

    print(f"[INFO] Sending chunked JPEG frames to udp://{args.udp_host}:{args.udp_port}")
    print(
        f"[INFO] capture={args.width}x{args.height}@{args.fps} source={args.source} "
        f"send={args.send_width}x{args.send_height} send_fps={args.send_fps} "
        f"jpeg_quality={args.jpeg_quality}"
    )
    print("[INFO] Receiver/calibrator should listen on the same frame port, usually 5020.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Gagal ambil frame dari kamera")
                time.sleep(0.05)
                continue

            now = time.time()

            if args.show:
                cv2.imshow("HOST Camera UDP Sender", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if args.save_on_s and key == ord("s"):
                    name = f"host_frame_{int(now)}.jpg"
                    cv2.imwrite(name, frame)
                    print(f"[INFO] Saved {name}")

            if now - last_send < send_period:
                continue
            last_send = now

            send_frame = frame
            if args.send_width > 0 and args.send_height > 0:
                send_frame = cv2.resize(
                    frame,
                    (args.send_width, args.send_height),
                    interpolation=cv2.INTER_AREA,
                )

            ok, enc = cv2.imencode(".jpg", send_frame, encode_param)
            if not ok:
                print("[WARN] JPEG encode gagal")
                continue

            jpg = enc.tobytes()
            frame_id = (frame_id + 1) & 0xFFFFFFFF
            chunks = send_jpeg_chunked(sock, addr, frame_id, jpg, args.max_chunk_payload)
            sent_count += 1
            last_jpeg_len = len(jpg)
            last_chunks = chunks

            if now - last_log >= 1.0:
                fps_actual = sent_count / max(now - start_time, 1e-6)
                print(
                    f"[UDP] frame_id={frame_id} sent={sent_count} "
                    f"jpeg={last_jpeg_len} bytes chunks={last_chunks} "
                    f"sent_fps_avg={fps_actual:.1f}"
                )
                last_log = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    finally:
        cap.release()
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

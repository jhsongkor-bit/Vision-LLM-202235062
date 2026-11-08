from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src" / "models" / "YOLO" / "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO11n on a webcam (Windows or Linux/Jetson).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt model path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else REPO_ROOT / args.model

    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(model_path))
    model.to(device)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print("Press q or Esc in the video window to stop.")

    backend = cv2.CAP_DSHOW if __import__("platform").system() == "Windows" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check the camera connection and permissions."
        )

    shown_fps = 0.0
    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("Camera frame could not be read; stopping.")
                break

            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )[0]
            output = result.plot()

            elapsed = max(time.perf_counter() - started, 1e-9)
            current_fps = 1.0 / elapsed
            shown_fps = current_fps if shown_fps == 0.0 else 0.9 * shown_fps + 0.1 * current_fps
            cv2.putText(
                output,
                f"FPS: {shown_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO11n Webcam Detection", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

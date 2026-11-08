from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
from ultralytics import YOLO


# [Windows 포팅] 카메라 프레임의 YOLO 탐지 결과를 Dictionary → JSON 순서로 저장
# Jetson 전용 CSI/GStreamer와 TensorRT 엔진 대신 Windows 카메라와 YOLO .pt 모델을 사용합니다.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src" / "models" / "YOLO" / "yolo11n.pt"
DEFAULT_JSON = REPO_ROOT / "src" / "json" / "vision_data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write YOLO camera detections to JSON.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO .pt model path.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Detection JSON output path.")
    return parser.parse_args()


def to_detection_json(result, frame_width: int, frame_height: int) -> dict:
    # 3) YOLO 탐지 결과를 Dictionary 형태로 변환
    objects = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        objects.append(
            {
                "class": result.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            }
        )
    return {"image_width": frame_width, "image_height": frame_height, "objects": objects}


def save_json_atomic(data: dict, json_path: Path) -> None:
    # 4) Dictionary → JSON 파일 저장 (임시 파일 작성 후 실제 파일로 교체)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = json_path.with_name(json_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, json_path)


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    json_path = args.json if args.json.is_absolute() else REPO_ROOT / args.json
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model file not found: {model_path}")

    # 1) YOLO11n 모델 불러오기
    model = YOLO(str(model_path))
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check the camera connection and permissions."
        )

    print("YOLO JSON capture ready. Press l to save the current frame's detections; q or Esc to stop.")
    try:
        while True:
            # 2) 현재 프레임 YOLO 탐지
            ok, frame = camera.read()
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
            # 3) YOLO 객체 탐지 결과 화면 출력
            cv2.putText(
                output,
                "l: save detections to JSON | q/Esc: quit",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("YOLO -> JSON", output)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("l"):
                # 5) 현재 프레임 탐지 결과를 JSON 구조로 저장
                height, width = frame.shape[:2]
                data = to_detection_json(result, width, height)
                save_json_atomic(data, json_path)
                print(f"Saved {len(data['objects'])} detections: {json_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

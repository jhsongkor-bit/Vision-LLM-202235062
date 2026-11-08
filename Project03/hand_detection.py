from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src" / "models" / "MediaPipe" / "hand_landmarker.task"
FINGER_TIPS = (4, 8, 12, 16, 20)
ANGLE_THRESHOLD = 160
FINGER_ANGLE_POINTS = (
    (1, 2, 3),       # 엄지
    (5, 6, 7),       # 검지
    (9, 10, 11),     # 중지
    (13, 14, 15),    # 약지
    (17, 18, 19),    # 새끼
)


def calculate_angle(p1, p2, p3) -> float:
    """중간 점을 기준으로 두 관절 벡터 사이의 각도를 계산합니다."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z], dtype=np.float32)
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z], dtype=np.float32)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect hands and count extended fingers with MediaPipe.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0).")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MediaPipe hand_landmarker.task path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe hand model not found: {model_path}")
    if args.max_hands < 1:
        raise ValueError("--max-hands must be at least 1")

    # MediaPipe Hand Landmarker 모델 설정 (Project01과 같은 Tasks API)
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=args.max_hands,
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)
    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

    # Use DirectShow on Windows and V4L2 on Linux/Jetson.
    import platform
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    camera = cv2.VideoCapture(args.camera, backend)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        hand_detector.close()
        raise RuntimeError(
            f"Could not open camera {args.camera}. Check the camera connection and permissions."
        )

    print("MediaPipe hand detection ready. Press q or Esc in the video window to stop.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Camera frame could not be read; stopping.")
                break

            # Project01처럼 좌우 반전 후 BGR → RGB 변환
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 프레임 내 손 탐지
            result = hand_detector.detect(mp_image)
            height, width = frame.shape[:2]
            total_fingers = 0

            # 각 손의 펼친 손가락 수 계산
            for hand in result.hand_landmarks:
                for point1_idx, point2_idx, point3_idx in FINGER_ANGLE_POINTS:
                    angle = calculate_angle(
                        hand[point1_idx],
                        hand[point2_idx],
                        hand[point3_idx],
                    )
                    if angle >= ANGLE_THRESHOLD:
                        total_fingers += 1

            # 화면에 손 개수와 손가락 수 표시
            cv2.putText(
                frame,
                f"Hands: {len(result.hand_landmarks)}  Fingers: {total_fingers}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 좌우 반전 화면 기준으로 왼손/오른손 이름을 표시
            labels = [
                "Left" if handedness[0].category_name == "Right" else "Right"
                for handedness in result.handedness
            ]
            if labels:
                cv2.putText(
                    frame,
                    " / ".join(labels),
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # 탐지된 각 손의 landmark 연결선(skeleton)과 점을 그림
            for hand in result.hand_landmarks:
                points = [(int(point.x * width), int(point.y * height)) for point in hand]
                for connection in connections:
                    cv2.line(
                        frame,
                        points[connection.start],
                        points[connection.end],
                        (0, 255, 0),
                        2,
                    )
                for index, point in enumerate(points):
                    color = (0, 0, 255) if index in FINGER_TIPS else (255, 0, 0)
                    radius = 6 if index in FINGER_TIPS else 4
                    cv2.circle(frame, point, radius, color, -1)

            cv2.imshow("MediaPipe Hand Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        hand_detector.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

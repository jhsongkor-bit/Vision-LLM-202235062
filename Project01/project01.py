import cv2
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 손가락 각도 계산

def calculate_angle(p1, p2, p3):
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# 점과 선분 사이의 거리 계산

def point_to_segment_distance(point, start, end):
    point = np.array(point, dtype=np.float32)
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)

    segment = end - start
    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / length_squared
    t = np.clip(t, 0.0, 1.0)
    closest_point = start + t * segment

    return np.linalg.norm(point - closest_point)


# MediaPipe 손 인식 모델 설정
base_option = python.BaseOptions(
    model_asset_path="src/models/MediaPipe/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=1
)
hand_detector = vision.HandLandmarker.create_from_options(options)
connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

finger_tips = (4, 8, 12, 16, 20)
angle_threshold = 160

# 각 손가락의 각도를 계산할 landmark index
finger_angle_points = (
    (1, 2, 3),
    (5, 6, 7),
    (9, 10, 11),
    (13, 14, 15),
    (17, 18, 19),
)


# Kalman Filter 설정
kalman = cv2.KalmanFilter(4, 2)

kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32
)

kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]],
    dtype=np.float32
)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
kalman.errorCovPost = np.eye(4, dtype=np.float32)

found = False
radius = 10
x_bel = 0
y_bel = 0


# 가상 탄환 설정
bullets = []              # 생성된 탄환들의 위치와 이동 속도를 저장
bullet_speed = 120        # 탄환 이동 속도
bullet_radius = 10        # 탄환 반지름

gun_was_detected = False  # 이전 프레임에서 총 자세가 감지되었는지 저장

shot_text = ""            # 화면에 표시할 발사/명중 결과
shot_text_frames = 0      # 결과 문구를 표시할 남은 프레임 수

hit_count = 0             # 표적을 맞힌 횟수
hit_effects = []          # 명중 효과 정보 저장

# Jetson 카메라 설정
pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # 화면 좌우 반전
    frame = cv2.flip(frame, 1)

    # 프레임 높이와 너비
    h, w = frame.shape[:2]
    result = frame.copy()

    # ============== 1. 공 검출 ==============

    # Gaussian Blur로 노이즈 제거
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)

    # BGR -> LAB 색공간 변환
    frame_lab = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2LAB)

    # CLAHE를 L 채널에 적용하여 이미지 강화
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l, a, b = cv2.split(frame_lab)
    l_clahe = clahe.apply(l)
    frame_lab = cv2.merge((l_clahe, a, b))

    # 공 색상에 맞는 색상 분할
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)

    final_ball_mask = cv2.inRange(frame_lab, green_lower, green_upper)

    # Morphology 연산으로 작은 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    final_ball_mask = cv2.morphologyEx(
        final_ball_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=2
    )

    # 윤곽선 검출
    contours, _ = cv2.findContours(
        final_ball_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # 가장 큰 윤곽선을 공으로 판단
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )

            (_, _), r = cv2.minEnclosingCircle(largest_contour)
            radius = max(1, int(r))

            # 최초 검출 시 Kalman filter 초기화
            if not found:
                kalman.statePre = np.array(
                    [[center[0]], [center[1]], [0.0], [0.0]],
                    dtype=np.float32
                )
                found = True

            # 실제 측정 위치
            z = np.array(
                [[np.float32(center[0])], [np.float32(center[1])]],
                dtype=np.float32
            )

            # 칼만 필터 예측 및 보정
            kalman.predict()
            corrected_state = kalman.correct(z)
            x_bel = int(corrected_state[0, 0])
            y_bel = int(corrected_state[1, 0])

            # 실제 검출 위치 표시
            cv2.circle(result, center, radius, (255, 0, 0), 2)

    # 공이 검출되지 않아도 이전 상태로 위치 예측
    elif found:
        predicted_state = kalman.predict()
        x_bel = int(predicted_state[0, 0])
        y_bel = int(predicted_state[1, 0])

    # 칼만 필터가 추정한 공 위치 표시
    if found:
        cv2.circle(result, (x_bel, y_bel), radius, (0, 255, 255), 2)
        cv2.circle(result, (x_bel, y_bel), 5, (0, 255, 255), -1)


    # ============== 2. 손 검출 ==============

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_result = hand_detector.detect(mp_image)

    # 좌우 반전된 화면을 기준으로 손 정보 표시
    labels = [
        "Left" if handedness[0].category_name == "Right" else "Right"
        for handedness in hand_result.handedness
    ]

    gun_detected = False
    shoot_info = None

    for hand in hand_result.hand_landmarks:

        # 손가락 관절 각도 계산
        finger_angles = []

        for point1_idx, point2_idx, point3_idx in finger_angle_points:
            angle = calculate_angle(
                hand[point1_idx],
                hand[point2_idx],
                hand[point3_idx]
            )
            finger_angles.append(angle)

        # 검지 두 번째 관절 각도
        index_angle2 = calculate_angle(hand[6], hand[7], hand[8])

        # 검지가 펴져 있는지 확인
        index_extended = (
            finger_angles[1] >= angle_threshold and
            index_angle2 >= angle_threshold
        )

        # 중지, 약지, 새끼가 접혀 있는지 확인
        other_fingers_folded = all(
            finger_angles[i] <= 90 for i in (2, 3, 4)
        )

        # 총 자세 판정
        if index_extended and other_fingers_folded:
            gun_detected = True

            # 검지 시작점과 끝점
            index_base = np.array(
                [hand[5].x * w, hand[5].y * h],
                dtype=np.float32
            )
            index_tip = np.array(
                [hand[8].x * w, hand[8].y * h],
                dtype=np.float32
            )

            # 검지가 가리키는 방향
            direction = index_tip - index_base
            direction_length = np.linalg.norm(direction)

            if direction_length > 0 and shoot_info is None:
                direction = direction / direction_length
                shoot_info = (
                    float(index_tip[0]),
                    float(index_tip[1]),
                    float(direction[0]),
                    float(direction[1])
                )

    # 총 모양이 새롭게 감지된 순간 한 발 발사
    if gun_detected and not gun_was_detected and shoot_info is not None:
        origin_x, origin_y, dir_x, dir_y = shoot_info

        bullets.append({
            "x": origin_x,
            "y": origin_y,
            "vx": dir_x * bullet_speed,
            "vy": dir_y * bullet_speed
        })

        shot_text = "SHOT!"
        shot_text_frames = 15

    # 이전 프레임의 총 자세 상태 저장
    # 총 자세를 유지하는 동안에는 다시 발사하지 않음
    gun_was_detected = gun_detected

    # 명중 횟수 표시
    cv2.putText(
        result,
        f"HITS: {hit_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # 총 자세 감지 시 상태 표시
    if gun_detected:
        gun_text = "GUN POSE DETECTED"
        text_size = cv2.getTextSize(
            gun_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]
        text_x = (w - text_size[0]) // 2

        cv2.putText(
            result,
            gun_text,
            (text_x, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # 손 방향 표시
    handedness_text = " / ".join(labels)
    if handedness_text:
        text_size = cv2.getTextSize(
            handedness_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )[0]
        text_x = max(10, w - text_size[0] - 20)

        cv2.putText(
            result,
            handedness_text,
            (text_x, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    # 총 모드 활성화 시 초록색 테두리
    if gun_detected:
        cv2.rectangle(result, (5, 5), (w - 5, h - 5), (0, 255, 0), 4)

    # 손 관절과 연결선 표시
    for hand in hand_result.hand_landmarks:
        points = [(int(p.x * w), int(p.y * h)) for p in hand]

        for connection in connections:
            cv2.line(
                result,
                points[connection.start],
                points[connection.end],
                (0, 255, 0),
                2
            )

        for i, point in enumerate(points):
            color = (0, 0, 255) if i in finger_tips else (255, 0, 0)
            point_radius = 6 if i in finger_tips else 4
            cv2.circle(result, point, point_radius, color, -1)


    # ============== 3. 가상 탄환 이동 및 충돌 판정 ==============

    active_bullets = []

    for bullet in bullets:
        previous_position = (bullet["x"], bullet["y"])

        # 탄환 이동
        bullet["x"] += bullet["vx"]
        bullet["y"] += bullet["vy"]
        current_position = (bullet["x"], bullet["y"])

        # 탄환 표시
        cv2.circle(
            result,
            (int(bullet["x"]), int(bullet["y"])),
            bullet_radius,
            (0, 0, 255),
            -1
        )

        hit = False

        # 목표물과 충돌 검사
        if found:
            distance = point_to_segment_distance(
                (x_bel, y_bel),
                previous_position,
                current_position
            )

            if distance < radius + bullet_radius:
                hit = True
                shot_text = "HIT!"
                shot_text_frames = 45
                hit_count += 1

                # 명중 효과 저장
                hit_effects.append({
                    "center": (int(x_bel), int(y_bel)),
                    "radius": max(5, radius),
                    "until": time.monotonic() + 1.0
                })

        # 명중한 탄환은 제거
        if hit:
            continue

        # 화면 밖으로 나가면 빗나간 것으로 판정
        if (
            bullet["x"] < 0 or bullet["x"] >= w or
            bullet["y"] < 0 or bullet["y"] >= h
        ):
            shot_text = "MISS!"
            shot_text_frames = 45
            continue

        # 아직 유효한 탄환만 유지
        active_bullets.append(bullet)

    bullets = active_bullets

    # 발사 및 명중 결과 표시
    if shot_text_frames > 0:
        text_color = (
            (0, 255, 0) if shot_text == "HIT!"
            else (0, 0, 255) if shot_text == "MISS!"
            else (255, 255, 0)
        )

        cv2.putText(
            result,
            shot_text,
            (20, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            text_color,
            3
        )
        shot_text_frames -= 1

    # 명중 효과 표시
    current_time = time.monotonic()
    hit_effects = [
        effect for effect in hit_effects
        if current_time < effect["until"]
    ]

    if hit_effects:
        overlay = result.copy()

        for effect in hit_effects:
            cv2.circle(
                overlay,
                effect["center"],
                effect["radius"],
                (0, 255, 255),
                -1
            )

        # 명중 효과를 반투명하게 합성
        result = cv2.addWeighted(overlay, 0.45, result, 0.55, 0)


    # ============== 4. 최종 결과 출력 ==============

    cv2.imshow("ball + Hand Tracking", result)
    # cv2.imshow("ball Binary Mask", final_ball_mask)

cap.release()
hand_detector.close()
cv2.destroyAllWindows()
import cv2
import numpy as np
import pandas as plt
from numpy.linalg import inv

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# pipeline = (
#     "nvarguscamerasrc sensor-id=0 ! "
#     "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
#     "nvvidconv ! "
#     "video/x-raw, format=BGRx ! "
#     "videoconvert ! "
#     "video/x-raw, format=BGR ! "
#     "queue leaky=downstream max-size-buffers=1 ! "
#     "appsink drop=true max-buffers=1 sync=false"
# )

def KalmanFilter(mu_prev, sigma_prev, z):
    mu_bar = A_t.dot(mu_prev)
    sigma_bar = A_t.dot(sigma_prev).dot(A_t.transpose()) + R_t
    if z is None:
        return mu_bar, sigma_bar
    else:
        K_t = sigma_bar.dot(C_t.transpose()).dot(inv(C_t.dot(sigma_bar).dot(C_t.transpose()) + Q_t))
        mu = mu_bar + K_t.dot(z - C_t.dot(mu_bar))
        sigma = (np.identity(2) - K_t.dot(C_t)).dot(sigma_bar)
        return mu, sigma


# Kalman filter 변수 정의
A_t = np.array([[1, 1], [0, 1]])
G = np.array([[0.5], [1]])
R_t = G.dot(G.transpose())
C_t = np.array([[1, 0]])
Q_t = np.array([[1]])
mu_t = np.array([[0, 0], [0, 0]])
sigma_t = np.array([[0, 0], [0, 0]])

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

found = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    green = frame.copy()
    # 초록 LAB lower/upper range
    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)

    green_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(green_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    green_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    mask_enhanced = cv2.inRange(green_enhanced_lab, green_lower, green_upper)
    green_enhanced_rgb = cv2.cvtColor(green_enhanced_lab, cv2.COLOR_LAB2RGB)

    # 타원형 커널 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # opening 및 iteration 2 방식 최종 사용
    final_green_mask = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contour를 그릴 원본 이미지 copy
    final_img = green.copy()

    # 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
    contours, _ = cv2.findContours(
        final_green_mask,        # binary mask
        cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
        final_mask = np.zeros_like(final_green_mask)
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(final_img)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

        #cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2)

    final_ball_mask = final_img

    # Contour 및 외접원을 그릴 원본 이미지 copy
    final_img_enhanced = green_enhanced_rgb.copy()

    if len(contours) > 0:
    # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
        mask_color = np.zeros_like(final_img_enhanced)
        mask_color[:, :] = largest_mask
        alpha = 0.15
        final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)


    # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(contours)
        center_circle = (int(x), int(y))
        radius = int(radius)

        cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
        cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)

        if not found:
            found = True

    # 최초 검출 이후 Kalman Filter 적용
    # 측정값 사용 (visible) -> Prediction & Update
    if found and (len(contours) > 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, np.array([list(center_circle)]))
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 측정값 미사용 (occluded) -> Prediction
    elif found and (len(contours) <= 0):
        mu_t, sigma_t = KalmanFilter(mu_t, sigma_t, None)
        x_bel, y_bel = mu_t[0][0], mu_t[0][1]

    # 예측한 객체 위치에 노란 원 overlay
    cv2.circle(frame, (int(x_bel), int(y_bel)), int(radius), (0,0,255), 3)
    cv2.circle(frame, (int(x_bel), int(y_bel)), (0,0,255), -1)

    cv2.imshow("VideoCapture with GStreamer", final_img_enhanced) #frame

cap.release()
cv2.destroyAllWindows()

# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay
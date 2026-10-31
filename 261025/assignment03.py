import cv2
import numpy as np
import cupy as cp
import time
import matplotlib.pyplot as plt

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0


# TODO: NumPy를 사용하여 잔상 효과 함수 정의
def motion_trail_np(image, weights):

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    output = np.zeros(image.shape, dtype=np.float32)

    for output_y in range(image.shape[0]):
        for output_x in range(image.shape[1]):
            total_pixel = 0.0

            for shift in range(TRAIL_LENGTH):
                # TODO: x값 shift
                input_x = output_x - shift

                if input_x < 0:
                    break

                # TODO: shift한 픽셀을 누적
                total_pixel += image[output_y, input_x] * weights[shift]

            # TODO: 출력 배열에 누적 픽셀 저장
            output[output_y, output_x] = total_pixel

    return output


# TODO: CuPy를 사용하여 잔상 효과 함수 정의
def motion_trail_cp(image, weights):

    # TODO: 필요한 변수 정의 및 출력 배열 생성
    output = cp.zeros(image.shape, dtype=cp.float32)

    for shift in range(TRAIL_LENGTH):
        # TODO: 배열 slicing
        input_region = image[:, :image.shape[1] - shift]
        output_region = output[:, shift:]

        # TODO: GPU가 병렬 연산할 output_region 계산 코드
        output_region += input_region * weights[shift]

    return output


# 이동 거리가 멀어질수록 가중치를 작게 설정
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

weights_np /= np.sum(weights_np)


# CPU 처리 시간 측정
cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start


# TODO: NumPy 배열을 CuPy 배열로 변환
upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start


# GPU 처리 시간 측정
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start


# TODO: output_cp를 CPU 메모리로 복사
download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start


# 속도 비교
compute_speedup = cpu_time / gpu_time

gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time

print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")


# 잔상 적용 결과
plt.figure(figsize=(12, 8))
plt.imshow(output_cp_np, cmap="gray")
plt.axis("off")
plt.show()
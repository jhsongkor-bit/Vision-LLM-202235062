# Assignment 02 (261017)

## 파일 구조

총 4개의 python 과제 파일이 있습니다.

<br>

### 03_DL-and-GPU.ipynb
- **Assignment03_cupy.py**  : GPU 가속 (잔상효과, Convolution2D)
- **Assignment03_mnist.py** : MNIST 데이터셋 main_gpu() 함수
- **Assignment03_cifar.py** : CIFAR-10 데이터셋 모델 학습 파트 + 저장한 모델 파일 (.pth)

### 04_DL-Object-Detection.ipynb
- **Assignment03_iou_nms.py** : IoU, NMS 구현

<br>

## 실행 방식

python -m pip install numpy cupy-cuda12x matplotlib - Assignment03.py의 필요 라이브러리 설치
python -m pip install torch torchvision - Assignment03_cifar.py, Assignment03_mnist.py의 필요 라이브러리 설치

<br>

### Assignment03_cupy.py

다음 커맨드를 입력합니다
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
cd Documents/git/Vision-LLM-202235062

Python 환경의 VSCode로 Assignment03.py를 불러온 뒤 우측 상단의 Run Python File 버튼을 눌러 실행합니다.

실행 시 CPU 연산 시간, CPU와 GPU간 정보 전달 시간, GPU 연산 시간, 비교 속도 배율과 잔상이 계산된 최종 이미지가 출력되게 됩니다.

### Assignment03_iou_nms.py

Python 환경의 VSCode로 Assignment03.py를 불러온 뒤 우측 상단의 Run Python File 버튼을 눌러 실행합니다.

실행 시 구현된 IoU를 이용한 NMS가 적용된 허블 이미지가 실행됩니다.

<br>

## 참고 사항

### Assignment03_cifar.py

데이터셋 불러오기가 실패하여 구현하지 못했습니다. 추후에 구현하도록 하겠습니다.

### Assignment03_cupy.py 구현 로직

이미지의 각 행 픽셀을 왼쪽으로 이동시키면서 가중치를 계산합니다. 처음엔 1번만 계산하지만 반복이 진행될수록 왼쪽 위치의 픽셀을 누적하는 방식으로 잔상을 구현합니다.

### Assignment03_iou_nms.py 구현 로직

우선 confidence 임계값을 설정하여 confidence score가 임계값을 넘은 bbox만 남깁니다. 그 후 남긴 bbox들을 확신도가 낮은 순서대로 저장하며 맨 뒤 bbox를 하나씩 꺼내 나머지와 IoU 계산을 하고 IoU가 너무 높으면 나머지 비교 대상을 제거하고 남은 bbox가 없을 때 까지 계속 반복하게 됩니다.  
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
python -m pip install torch torchvision - Assignment03-2.py의 필요 라이브러리 설치

<br>

### Assignment03_cupy.py

Python 환경의 VSCode로 Assignment03.py를 불러온 뒤 우측 상단의 Run Python File 버튼을 눌러 실행합니다.

실행 시 CPU 연산 시간, CPU와 GPU간 정보 전달 시간, GPU 연산 시간, 비교 속도 배율과 잔상이 계산된 최종 이미지가 출력되게 됩니다.

### Assignment03_iou_nms.py

Python 환경의 VSCode로 Assignment03.py를 불러온 뒤 우측 상단의 Run Python File 버튼을 눌러 실행합니다.

<br>

## 참고 사항


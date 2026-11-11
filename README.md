# Vision-LLM

**202235062 송중현**

## Repository 소개

Vision-LLM 수업에서 진행한 실습과 과제를 정리하는 Repository입니다.

## 저장소 구조 설명

```text
.
├── 261010/
│   ├── README.md
│   ├── assignment01.py
├── 261017/
│   ├── README.md
│   ├── assignment02.py
├── 261025/
│   ├── README.md
│   ├── assignment03_cifar.py
│   ├── assignment03_cupy.py
│   ├── assignment03_iou_nms.py
│   ├── assignment03_mnist.py
├── Project01/
│   ├── README.md
│   ├── project01.py
├── Project03/
│   ├── README.md
│   ├── main.py
├── src/
│   ├── dataset/
│   ├── images/
│   ├── models/
├── .gitignore
├── 00_Initial-Setup.ipynb
├── 01_Linux-and-Python.ipynb
├── 02_Computer-Vision.ipynb
├── 03_DL-and-GPU.ipynb
├── 04_DL-Object-Detection.ipynb
├── 05-LLM_and_Gemma.ipynb
├── 06-Vision_LLM_Multimodal_Systems.ipynb
├── kalman_filter.py
├── hand.py
├── hand_and_finger.py
├── yolo11n_camera.py
├── yolo_fps.py
├── yolo_calibration.py
├── mic.py
├── present.pdf
└── README.md
```

## 파일 및 폴더 설명

| 파일 및 폴더 | 설명 |
|---|---|
| `26xxxx/` | 26년 xx월 xx일 실습 및 과제 폴더 |
| `Projectxx` | 프로젝트 폴더 코드 |
| `src/` | 소스 폴더 |
| `.gitignore` | 불필요 파일/폴더 무시 |
| `00_Initial-Setup.ipynb` | 환경 설정 실습 |
| `01_Linux-and-Python.ipynb` | 1회차 실습 |
| `02_Computer-Vision.ipynb` | 2~4회차 실습 |
| `03_DL-and-GPU.ipynb` | 6~7회차 실습 |
| `04_DL-Object-Detection.ipynb` | 7~8회차 실습 |
| `05-LLM_and_Gemma.ipynb` | 8~9회차 실습 |
| `06-Vision_LLM_Multimodal_Systems.ipynb` | 9회차 실습 |
| `README.md` | 저장소 소개 |
| `kalman_filter.py`, `hand.py`, `hand_and_finger.py` | Kalman filter를 이용한 움직임 추적 및 손·손가락 움직임 검출 실습 코드|
| `yolo11n_camera.py`, `yolo_fps.py`, `yolo_calibration.py` | YOLO11n 모델 객체 검출 및 성능 실습 코드 |
| `mic.py` | 마이크 테스트 코드 |
| `present.pdf` | project03 발표자료 |

## Project03 관련

- `Project03/main.py`: 책 감지, 책 선택, STT, Gemma 답변, Piper 음성 출력 통합
- `Project03/audio_test.py`: 마이크 녹음 및 STT/TTS 확인
- `Project03/gemma_voice.py`: YOLO JSON 정보와 음성 질문을 Gemma에 전달하는 실험 코드
- `Project03/README.md`: Project03 상세 설정 및 실행 안내


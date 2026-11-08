# Project 03

## 파일 구조

- **yolo_camera.py**: YOLO11n 웹캠 객체 탐지 및 FPS 표시
- **hand_detection.py**: MediaPipe 손 랜드마크 탐지, 좌우 손 표시, 손가락 개수 계산
- **vision_json.py**: 카메라 프레임의 YOLO 탐지 결과를 JSON으로 저장
- **gemma_chat.py**: 로컬 Gemma 텍스트 챗봇
- **gemma_voice.py**: 음성 STT → YOLO JSON Context → Gemma → Piper TTS 통합 실행 코드
- **audio_test.py**: 마이크 녹음 → 한국어 STT → 인식 문장을 Piper TTS로 재생하는 입출력 테스트
- **main.py**: YOLO 책 감지 + 검지 끝점 선택 + R 키 녹음 STT + Gemma Vision 답변 + Piper 낭독
- **src/json/book_notes.json**: 현재 책 제목만 저장하는 메모리
- **hand_gemma_voice_test.py**: MediaPipe 손가락 수 + Whisper STT + Gemma 응답 + Piper TTS 실험
- **src/models/MediaPipe/hand_landmarker.task**: MediaPipe 손 랜드마크 인식 모델
- **src/models/YOLO/yolo11n.pt**: YOLO11n 사전 학습 모델
- **src/models/Gemma4/**: 로컬 Gemma GGUF 모델 및 이미지 입력용 mmproj
- **src/models/Whisper/base.pt**: OpenAI Whisper base 모델 (첫 실행 때 자동 다운로드)
- **src/models/Piper/**: 한국어 음성 합성용 Piper 모델
<br>

## 실행 방식

1. 저장소 루트에서 가상환경 활성화

cd C:\Users\Song\Desktop\Vision-LLM-202235062
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Jetson

2. YOLO 웹캠 객체 탐지 실행

python .\Project03\yolo_camera.py

기본 카메라는 `0`번입니다. 다른 카메라를 사용하려면 다음과 같이 실행합니다.

python .\Project03\yolo_camera.py --camera 1 --conf 0.25 --iou 0.5

3. MediaPipe 손 감지 실행

python .\Project03\hand_detection.py

최대 손 개수를 바꾸려면 다음과 같이 실행합니다.

python .\Project03\hand_detection.py --camera 0 --max-hands 2

4. 로컬 Gemma 텍스트 챗봇 실행

python .\Project03\gemma_chat.py

5. 책을 가리키고 음성으로 질문하기

python .\Project03\main.py --camera 2  # Windows
python3 ./Project03/main.py --camera 0  # Linux/Jetson

Jetson/Linux에서는 Project01과 같은 GStreamer 카메라 파이프라인(`/dev/videoN`, YUY2 640×480, 30 FPS, leaky queue/appsink)을 우선 사용하고, 실패하면 V4L2로 대체합니다. Windows 기본 카메라는 2번, Linux/Jetson은 0번이며 `--camera`로 지정할 수 있습니다. YOLO가 책을 감지하고 예측 시 NMS IoU 임계값을 기본 `0.5`로 적용합니다. `--iou 0.4`처럼 값을 바꿀 수 있습니다. 화면에는 감지된 책 바운딩 박스 수를 `Books`로 표시하며, “지금 책이 몇 개야?”라고 물으면 NMS 이후 남은 책 박스 수로 답합니다. Project01의 검지 관절 각도 계산(검지 관절 160도 이상)을 적용하며, 펴진 검지 끝점(landmark 8)만 노란 점으로 표시합니다. 검지 끝점이 책 바운딩 박스 안에 들어오면 해당 책의 박스도 노란색으로 표시됩니다. 감지 결과는 기본 8프레임 동안 유지하며 `--hold-frames 12`처럼 조정할 수 있습니다. 책을 가리킨 뒤 R 키를 누르면 오디오 테스트 실습처럼 마이크에서 5초간 녹음하고 Whisper STT가 질문을 인식합니다. 녹음·STT·Gemma·TTS는 백그라운드에서 처리되어 카메라 화면은 계속 갱신됩니다. Gemma가 선택한 책 이미지와 질문을 바탕으로 답하고 Piper가 읽어줍니다. 인식한 질문과 답변은 화면 오른쪽 아래에 표시되며, 10초 동안 새 질문이 없으면 팝업을 숨깁니다. 실행 중 선택한 책 제목만 `src/json/book_notes.json`에 저장합니다. 책을 가리키지 않고 줄거리나 책 내용을 물으면 제목만 Gemma에 전달해 자체 지식으로 답하며, 정보가 부족해도 Gemma가 가장 그럴듯한 답을 단정적으로 말합니다. 저자·장르 질문도 해당 정보를 메모에서 읽지 않고 저장된 제목만 Gemma에 전달해 자체 지식으로 답합니다. 저자·장르 답변은 메모에 저장하지 않으며, 정보가 불확실해도 추정이라고 밝히지 않고 가장 그럴듯한 답을 말합니다. 저자·장르·줄거리·본문 정보는 메모에 저장하지 않습니다. 책 제목을 물어도 기존 제목을 먼저 지우지 않습니다. 다른 책을 가리켜 제목을 새로 인식하면 저장된 제목을 갱신합니다. 책 제목은 프로그램 종료 후에도 저장되어 다음 실행에서 불러옵니다. 책 제목을 다시 묻거나 다른 책을 선택하면 저장된 제목을 갱신합니다. Gemma는 표지에서 확인할 수 있는 제목·저자·장르·간단한 줄거리를 답하고, 펼친 페이지가 보이면 이미지에 보이는 내용에 관해 답합니다. 마이크 장치는 `--device 1`처럼 바꿀 수 있습니다. 음성 처리 중에도 카메라가 계속 갱신됩니다. `Q` 또는 `Esc`를 눌러 종료합니다.

6. 마이크·STT·TTS 단독 테스트

python .\Project03\audio_test.py

5초간 녹음한 뒤 한국어 STT 결과를 표시하고, 인식된 문장을 Piper가 음성으로 재생합니다. 마이크 장치를 확인하려면 `python .\Project03\audio_test.py --list-devices`, 다른 입력 장치를 지정하려면 `python .\Project03\audio_test.py --device 2`처럼 실행합니다. 녹음과 TTS 파일은 `src/audio/input.wav`, `src/audio/response.wav`에 저장됩니다.

7. 손가락 감지 + STT + Gemma + TTS 실험

python .\Project03\hand_gemma_voice_test.py --camera 2

화면에 손을 보여 주면 Project01과 같은 관절 각도 기준으로 손과 펼친 손가락 수를 표시합니다. 마이크가 상시 대기하며 음성이 끝난 구간을 자동으로 STT 처리하고, 현재 손 감지 결과와 함께 Gemma에 전달합니다. Gemma의 답변을 Piper가 읽습니다. 음성·STT·Gemma·TTS 처리는 백그라운드에서 진행되어 카메라 프레임은 계속 갱신됩니다. 마이크를 바꾸려면 `--device 1`을 추가하고, 음성 감지 민감도를 조절하려면 `--energy-threshold 0.015`처럼 지정합니다. `Q` 또는 `Esc`를 눌러 종료합니다.

가상환경을 새로 만들 때에는 저장소 루트에서 아래 패키지를 설치합니다.

python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Jetson
python -m pip install --upgrade pip
pip install ultralytics opencv-contrib-python torch torchvision matplotlib ipykernel
pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice
pip install transformers sentencepiece onnx onnxruntime onnxslim ml_dtypes coloredlogs humanfriendly protobuf huggingface_hub python-dotenv sounddevice openai-whisper
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Piper는 별도 가상환경을 사용합니다.

python -m venv .piper_venv
.\.piper_venv\Scripts\Activate.ps1
pip install piper-tts

<br>

## 창의성 및 확장성

### 아이디어 설명

웹캠에 보이는 장면을 YOLO11n으로 탐지하고 결과를 JSON으로 구조화한 뒤, 로컬 Gemma가 객체 정보를 바탕으로 장면을 설명하는 Vision-LLM 시스템입니다. 사용자는 음성으로 질문하고, OpenAI Whisper가 한국어 음성을 텍스트로 변환하며, Gemma의 답변은 Piper가 음성으로 읽어줍니다.

카메라 탐지와 음성 응답을 별도 프로세스로 실행하여 Gemma 추론 중에도 카메라 탐지를 계속할 수 있습니다. JSON 파일을 두 프로세스 사이의 데이터 전달 형식으로 사용합니다.

<br>

### 구현한 내용

- 객체 탐지: YOLO11n 모델로 프레임 속 COCO 객체를 탐지하고 Bounding Box, 클래스, Confidence Score를 얻습니다.

- JSON 저장: 이미지 너비·높이와 객체 목록을 `src/json/vision_data.json`에 저장합니다. 임시 JSON 작성 후 파일을 교체하여 읽는 중간에 불완전한 데이터가 전달되지 않도록 합니다.

- Vision Context 생성: JSON의 클래스와 Confidence Score를 자연어 문장으로 바꾸어 Gemma 입력 Context로 사용합니다.

- 로컬 LLM: Gemma 4 E2B GGUF 모델을 `llama-cpp-python`으로 실행합니다. 탐지 결과에 없는 객체를 추측하지 않고 한국어 두 문장 이내로 답하도록 Prompt를 설정합니다.

- STT: 기본 마이크에서 5초간 녹음하고 OpenAI Whisper base 모델로 한국어 음성을 텍스트로 변환합니다. 모델 파일이 없으면 첫 실행 때 `src/models/Whisper/base.pt`에 자동 다운로드합니다. Jetson에서는 CUDA가 사용 가능하면 자동으로 CUDA를 사용하고, 아니면 CPU를 사용합니다.

- TTS: 별도 `.piper_venv`에서 한국어 Piper 모델로 Gemma 답변을 WAV로 생성하고 기본 재생 장치로 출력합니다.

- 실행 환경: Windows 또는 Linux/Jetson 웹캠과 로컬 모델 파일을 사용합니다. 모델 추론은 CPU에서 실행하도록 설정했습니다.

- 손 감지: Project01의 MediaPipe Hand Landmarker Tasks API를 사용해 양손 landmark를 검출하고, 관절 각도 160도 기준으로 손가락 개수를 계산합니다.

<br>

### 확장성

- 객체별 규칙을 추가하여 특정 클래스가 나타나거나 사라질 때 음성 알림을 줄 수 있습니다.

- Bounding Box 좌표를 활용해 관심 영역(ROI)을 잘라 Gemma의 이미지 입력용 mmproj와 함께 전달할 수 있습니다.

- JSON 파일 대신 소켓이나 메시지 큐를 사용해 Vision과 LLM 사이의 데이터 전달을 확장할 수 있습니다.

- 녹음 길이, STT 모델 크기, Gemma Context 길이, 응답 길이, 카메라 번호를 사용 환경에 맞게 조정할 수 있습니다.

<br>

## 참고 사항

06번 실습의 전체 처리 흐름과 YOLO 임계값, JSON 필드, Gemma Prompt 제약, Piper 음성 모델을 따릅니다. TensorRT `.engine`·CSI/GStreamer 입력은 아직 연결하지 않았으며, 기본 카메라는 OpenCV V4L2/DirectShow를 사용합니다. STT는 OpenAI Whisper Python 라이브러리를 사용하고, 마이크와 오디오 출력은 `sounddevice`를 사용합니다.

Gemma·YOLO·MediaPipe·Piper 모델은 저장소의 `src/models` 폴더에 있어야 합니다. Whisper base 모델은 `openai-whisper`가 첫 실행 때 `src/models/Whisper/base.pt`로 내려받습니다. 첫 실행에는 인터넷 연결이 필요하며, 이후에는 로컬 파일을 사용합니다. Jetson에서는 JetPack과 호환되는 PyTorch가 설치되어 있어야 합니다. 웹캠과 마이크를 연결한 실제 실행은 아직 확인하지 않았습니다.

<br>

---






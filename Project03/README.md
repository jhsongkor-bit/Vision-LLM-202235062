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
- **whisper.cpp/models/ggml-base.bin**: 06번 실습과 같은 whisper.cpp 한국어 STT 모델
- **src/models/Piper/**: 한국어 음성 합성용 Piper 모델
<br>

## 실행 방식

1. 저장소 루트에서 가상환경 활성화

cd C:\Users\Song\Desktop\Vision-LLM-202235062
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Jetson

2. YOLO 웹캠 객체 탐지 실행

python .\Project03\yolo_camera.py

기본 카메라는 0번입니다. 다른 카메라를 사용하려면 다음과 같이 실행합니다.

python .\Project03\yolo_camera.py --camera 0


3. 로컬 Gemma 텍스트 챗봇 실행

python .\Project03\gemma_chat.py

4. 책을 가리키고 음성으로 질문하기

python3 ./Project03/main.py --camera 0  # Linux/Jetson

Linux/Jetson 기준 0이며 `--camera`로 지정할 수 있습니다. r키로 음성을 5초간 받을 수 있으며 음성 처리 중에도 카메라가 계속 갱신됩니다. `Q` 또는 `Esc`를 눌러 종료합니다.

5. 마이크·STT·TTS 단독 테스트

python .\Project03\audio_test.py

5초간 녹음한 뒤 한국어 STT 결과를 표시하고, 인식된 문장을 Piper가 음성으로 재생합니다. 마이크 장치를 확인하려면 `python .\Project03\audio_test.py --list-devices`, 다른 입력 장치를 지정하려면 `python .\Project03\audio_test.py --device 2`처럼 실행합니다. 녹음과 TTS 파일은 `src/audio/input.wav`, `src/audio/response.wav`에 저장됩니다.

6. 손가락 감지 + STT + Gemma + TTS 실험

python .\Project03\hand_gemma_voice_test.py --camera 2

화면에 손을 보여 주면 Project01과 같은 관절 각도 기준으로 손과 펼친 손가락 수를 표시합니다. 마이크가 상시 대기하며 음성이 끝난 구간을 자동으로 STT 처리하고, 현재 손 감지 결과와 함께 Gemma에 전달합니다. Gemma의 답변을 Piper가 읽습니다. 음성STT,Gemma,TTS 처리는 백그라운드에서 진행되어 카메라 프레임은 계속 갱신됩니다. 마이크를 바꾸려면 `--device 1`을 추가하고,`Q` 또는 `Esc`를 눌러 종료합니다.

가상환경을 새로 만들 때에는 저장소 루트에서 아래 패키지를 설치합니다.

python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # Linux/Jetson
python -m pip install --upgrade pip
pip install ultralytics opencv-contrib-python torch torchvision matplotlib ipykernel
pip install --no-deps mediapipe
pip install absl-py flatbuffers sounddevice
pip install transformers sentencepiece onnx onnxruntime onnxslim ml_dtypes coloredlogs humanfriendly protobuf huggingface_hub python-dotenv sounddevice
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Whisper는 06번 실습처럼 저장소 루트에 `whisper.cpp`를 받고 CPU 버전으로 빌드합니다. Linux/Jetson에서 저장소 루트 기준으로 실행합니다.

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build-cpu
cmake --build build-cpu --target whisper-cli -j4 --config Release
sh ./models/download-ggml-model.sh base
```

Piper는 별도 가상환경을 사용합니다.

python -m venv .piper_venv
.\.piper_venv\Scripts\Activate.ps1
pip install piper-tts

<br>

## 창의성 및 확장성

### 아이디어 설명

나만의 AI 사서로, 도서관이나 문화시설을 돌아다니며 책장에 꽂힌 책들을 보는 상황을 가정할 때, 간혹 어떤 책의 정보를 얻고싶거나 제목이 외국어로 되어있는 등 책에 대해 궁금할 때 세로로 꽂혀있는 책을 직접 꺼내보거나 정보를 직접 인터넷으로 불확실하게 확인하지 않고, 내부에 ~~RAG로 보강한~~ 챗봇으로 책의 정보를 빠르게 파악할 수 있게 합니다.

웹캠에 보이는 장면을 YOLO11n으로 탐지하고 결과를 JSON으로 구조화한 뒤, 로컬 Gemma가 객체 정보를 바탕으로 장면을 설명하는 Vision-LLM 시스템입니다. 사용자는 음성으로 질문하고, whisper.cpp가 한국어 음성을 텍스트로 변환하며, Gemma의 답변은 Piper가 음성으로 읽어줍니다.

카메라 탐지와 음성 응답을 별도 프로세스로 실행하여 Gemma 추론 중에도 카메라 탐지를 계속할 수 있습니다. JSON 파일을 두 프로세스 사이의 데이터 전달 형식으로 사용합니다.

<br>

### 구현한 내용

- 객체 탐지: YOLO11n 모델로 프레임 속 COCO 객체를 탐지하고 Bounding Box, 클래스, Confidence Score를 얻습니다. 그리고 NMS로 책의 중복 감지를 막습니다.

- JSON 저장: 이미지 너비·높이와 객체 목록을 `src/json/vision_data.json`에 저장합니다. 임시 JSON 작성 후 파일을 교체하여 읽는 중간에 불완전한 데이터가 전달되지 않도록 합니다.

- Vision Context 생성: JSON의 클래스와 Confidence Score를 자연어 문장으로 바꾸어 Gemma 입력 Context로 사용합니다.

- 로컬 LLM: Gemma 4버전 모델을 `llama-cpp-python`으로 실행합니다. 탐지 결과에 없는 객체를 추측하지 않고 한국어 두 문장 이내로 답하도록 Prompt를 설정합니다.

- STT: 기본 마이크에서 5초간 녹음하고 06번 실습과 같은 `whisper-cli`를 `-m ggml-base.bin -f input.wav -l ko -nt` 옵션으로 실행해 한국어 음성을 텍스트로 변환합니다. Jetson에서는 CPU 빌드 경로 `whisper.cpp/build-cpu/bin/whisper-cli`를 사용합니다.

- TTS: 별도 `.piper_venv`에서 한국어 Piper 모델로 Gemma 답변을 WAV로 생성하고 기본 재생 장치로 출력합니다.

- 실행 환경: Windows 또는 Linux/Jetson 웹캠과 로컬 모델 파일을 사용합니다. Windows로 할 때 모델 추론은 CPU에서 실행하도록 설정했습니다.

- 손 감지: Project01의 MediaPipe Hand Landmarker Tasks API를 사용해 양손 landmark를 검출하고, 검지의 관절 각도가 160도일때의 검지 끝점을 감지합니다.

<br>

### 확장성

- 녹음 길이, STT 모델 크기, Gemma Context 길이, 응답 길이, 카메라 번호를 사용 환경에 맞게 조정할 수 있습니다.

<br>

### 한계 및 보완사항

- 현재 YOLOn11 모델은 별도의 최적화를 진행하지 않았는데 Jetson의 추론 속도 향상 및 프레임 상승을 위해 INT8 양자화와 ONNX 형식의 모델변환을 적용할 수 있을 것 같습니다.

- YOLO모델과 디바이스의 한계로 책 검출시 위치가 정교하지 않고 특히 걸어다닐때 중복 검출 및 미검출이 매우 많아지는 것이 확인됐습니다. 더 좋은 환경에선 더 안정적인 검출이 가능할 것 같습니다.

- RAG 등을 사용하여 Gemma에 책 관련 내용을 대거 보강하는 것을 생각해 볼 수 있겠습니다.

<br>

## 참고 사항

구현 및 공부할 시간이 부족하여 아이디어 및 파이프라인, 기초 코드만 구현하고 작동하는 코드와 코드 간 연결은 AI의 도움을 받아 작성하였습니다. 앞으로는 공지된 최종 프로젝트 일정을 감안하여 미리 준비 할 시간을 갖도록 하겠습니다. 

<br>

---

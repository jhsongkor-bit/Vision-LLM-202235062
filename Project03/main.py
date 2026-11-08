from __future__ import annotations

import argparse
import base64
import json
import platform
import queue
import re
import subprocess
import sys
import threading
import wave
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL = ROOT / "src" / "models" / "YOLO" / "yolo11n.pt"
HAND_MODEL = ROOT / "src" / "models" / "MediaPipe" / "hand_landmarker.task"
GEMMA_MODEL = ROOT / "src" / "models" / "Gemma4" / "google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_MODEL = ROOT / "src" / "models" / "Gemma4" / "mmproj-google_gemma-4-E2B-it-f16.gguf"
WHISPER_ROOT = ROOT / "whisper.cpp"
WHISPER_MODEL = WHISPER_ROOT / "models" / "ggml-base.bin"
PIPER_MODEL = ROOT / "src" / "models" / "Piper" / "ko_KR-kss-medium.onnx"
PIPER_PYTHON = ROOT / ".piper_venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
INPUT_WAV = ROOT / "src" / "audio" / "book_question.wav"
OUTPUT_WAV = ROOT / "src" / "audio" / "book_title.wav"
BOOK_MEMORY_PATH = ROOT / "src" / "json" / "book_notes.json"
ANGLE_THRESHOLD = 160
RECORD_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point at a book and hear its cover title read aloud.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: 2 on Windows, 0 on Linux/Jetson).")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold (default: 0.5).")
    parser.add_argument("--hold-frames", type=int, default=8, help="Keep the last fingertip/book hit for missed frames.")
    parser.add_argument("--device", type=int, default=None, help="Microphone input device index; default microphone if omitted.")
    return parser.parse_args()


def whisper_cli_path() -> Path:
    if sys.platform == "win32":
        candidates = (
            WHISPER_ROOT / "build-cpu" / "bin" / "Release" / "whisper-cli.exe",
            WHISPER_ROOT / "build-cpu" / "bin" / "whisper-cli.exe",
            WHISPER_ROOT / "build" / "bin" / "Release" / "whisper-cli.exe",
            WHISPER_ROOT / "build" / "bin" / "whisper-cli.exe",
        )
    else:
        candidates = (
            WHISPER_ROOT / "build-cpu" / "bin" / "whisper-cli",
            WHISPER_ROOT / "build" / "bin" / "whisper-cli",
        )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def open_camera(camera_index: int) -> cv2.VideoCapture:
    if platform.system() == "Linux":
        pipeline = (
            f"v4l2src device=/dev/video{camera_index} ! "
            "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "queue leaky=downstream max-size-buffers=1 "
            "max-size-bytes=0 max-size-time=0 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if camera.isOpened():
            print(f"Camera: /dev/video{camera_index} via Project01 GStreamer pipeline")
            return camera
        camera.release()
        camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if camera.isOpened():
            print(f"Camera: /dev/video{camera_index} via V4L2 fallback")
            return camera
        camera.release()
        camera = cv2.VideoCapture(camera_index)
        if camera.isOpened():
            print(f"Camera index: {camera_index} via OpenCV fallback")
        return camera

    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if camera.isOpened():
        print(f"Camera index: {camera_index}")
    return camera


def calculate_angle(p1, p2, p3) -> float:
    """3D landmark joint-angle calculation copied from Project01."""
    vector1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    vector2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    cosine = np.dot(vector1, vector2) / (magnitude1 * magnitude2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def point_in_box(point: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def is_book_count_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return "책" in compact and any(keyword in compact for keyword in ("몇", "개수", "권수"))


def is_book_title_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(keyword in compact for keyword in ("책제목", "책이름", "제목", "무슨책", "어떤책", "책뭐야"))


def is_plot_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(keyword in compact for keyword in ("줄거리", "스토리", "이야기", "내용", "요약"))


def is_author_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(keyword in compact for keyword in ("저자", "작가", "지은이", "쓴사람", "누가썼", "누가쓴"))


def is_genre_question(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return any(keyword in compact for keyword in ("장르", "분류", "어떤종류"))


def save_book_memory(memory: dict) -> None:
    BOOK_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = BOOK_MEMORY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(BOOK_MEMORY_PATH)


def draw_conversation_panel(frame: np.ndarray, question: str, answer: str) -> np.ndarray:
    if not question and not answer:
        return frame
    height, width = frame.shape[:2]
    font_size = max(16, min(22, height // 28))
    font_paths = (
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    font = next((ImageFont.truetype(str(path), font_size) for path in font_paths if path.is_file()), ImageFont.load_default())
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    panel_width = min(560, max(280, width // 2))
    padding = 12
    max_text_width = panel_width - padding * 2
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    def wrap(label: str, text: str, max_lines: int = 3) -> list[str]:
        content = f"{label}: {text}" if text else f"{label}: -"
        lines: list[str] = []
        line = ""
        for char in content:
            candidate = line + char
            if line and draw.textbbox((0, 0), candidate, font=font)[2] > max_text_width:
                lines.append(line)
                line = char
            else:
                line = candidate
        if line:
            lines.append(line)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = lines[-1].rstrip(" .…") + "…"
        return lines

    q_lines = wrap("질문", question)
    a_lines = wrap("답변", answer)
    line_height = font_size + 5
    panel_height = padding * 2 + (len(q_lines) + len(a_lines) + 1) * line_height
    x = max(8, width - panel_width - 15)
    y = max(8, height - panel_height - 42)
    draw.rounded_rectangle(
        (x, y, x + panel_width, y + panel_height),
        radius=10,
        fill=(0, 0, 0, 185),
        outline=(255, 255, 255, 150),
        width=1,
    )
    cursor_y = y + padding
    for line in q_lines:
        draw.text((x + padding, cursor_y), line, font=font, fill=(255, 255, 255, 255))
        cursor_y += line_height
    cursor_y += line_height // 3
    for line in a_lines:
        draw.text((x + padding, cursor_y), line, font=font, fill=(190, 255, 190, 255))
        cursor_y += line_height
    image = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def answer_book_question(
    llm: Llama,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    question: str,
    previous_title: str | None,
) -> tuple[str | None, str]:
    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if crop.size == 0:
        return None, "사진에서 책 정보를 읽지 못했어요."
    max_width = 1024
    if crop.shape[1] > max_width:
        scale = max_width / crop.shape[1]
        crop = cv2.resize(crop, (max_width, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return None, "사진에서 책 정보를 읽지 못했어요."
    image_uri = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
    previous_title = previous_title or "없음"
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "사진 속 선택된 책과 사용자의 질문에 한국어로 답하세요. 현재 이미지에서 책 제목을 읽어 book_title에 쓰세요. "
                    "이전 책 제목은 현재 책 제목과 비교하는 데만 사용하세요. 현재 이미지의 제목을 현재 책 제목으로 선택하세요. "
                    "책 제목을 물으면 표지에서 읽으세요. 책 내용은 펼친 페이지에서 읽히는 부분만 설명하세요. "
                    "표지에서 객관적으로 확인할 수 있는 정보가 있으면 장르, 저자, 간단한 줄거리를 답하세요. "
                    "사용자의 질문에는 책 제목, 저자, 장르, 줄거리 중 하나 이상이 포함될 수 있습니다. "
                    "여러 항목을 한 번에 질문하면 질문에 포함된 모든 항목을 빠짐없이 답하세요. "
                    "책의 줄거리를 물으면 현재 이미지에서 읽은 책 제목만 단서로 모델의 자체 지식을 사용하세요. "
                    "정보가 불완전해도 가장 그럴듯한 답을 자연스럽게 단정해서 말하고, 모른다거나 추정이라고 밝히지 마세요. "
                    "반드시 다음 JSON 형식으로만 답하세요: "
                    '{"book_title":"표지에서 읽은 제목 또는 null","answer":"질문에 대한 답변"}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"현재 질문: {question}\n"
                            f"이전 책 제목(현재 제목 확인용): {previous_title}"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_uri}},
                ],
            },
        ],
        max_tokens=220,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = response["choices"][0]["message"]["content"].strip()
    try:
        parsed = json.loads(raw)
        title = parsed.get("book_title")
        title = title.strip() if isinstance(title, str) and title.strip().lower() != "null" else None
        answer = str(parsed.get("answer", "")).strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        title, answer = None, raw
    if not title and is_book_title_question(question):
        title_response = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "이미지에서 책 표지의 제목만 정확히 읽어 JSON으로 답하세요. 제목을 확실히 읽을 수 없으면 null을 반환하고, "
                        "저자명이나 다른 문구를 제목으로 쓰거나 추측하지 마세요. 형식: {\"book_title\": \"제목 또는 null\"}"
                    ),
                },
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image_uri}}],
                },
            ],
            max_tokens=80,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        try:
            extracted = json.loads(title_response["choices"][0]["message"]["content"])
            candidate = extracted.get("book_title")
            if isinstance(candidate, str) and candidate.strip() and candidate.strip().lower() != "null":
                title = candidate.strip()
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    if not answer:
        answer = "사진에서 책 정보를 읽지 못했어요. 표지나 페이지를 더 가까이 보여 주세요."
    return title, answer


def answer_plot_from_title(llm: Llama, question: str, title: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자가 책의 줄거리나 내용을 묻고 있습니다. 전달된 책 제목만 단서로 사용하고, "
                    "저자·줄거리·본문 등 저장 메모나 이전 대화 정보는 사용하지 마세요. "
                    "모델 자체 지식으로 해당 책의 줄거리를 한국어로 간결하게 답하세요. "
                    "정보가 부족해도 제목과 관련해 가장 그럴듯한 줄거리를 답변처럼 자연스럽게 단정해 말하세요. "
                    "모른다거나 추정이라고 밝히지 마세요. "
                    "외부 검색이나 추가 메모리 조회를 하지 마세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"책 제목: {title}\n질문: {question}"
                ),
            },
        ],
        max_tokens=180,
        temperature=0.2,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer or "이 책은 한 인물이 새로운 세계를 만나며 여러 사건을 겪는 이야기입니다."


def answer_author_from_title(llm: Llama, question: str, title: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자가 저장된 책 제목에 대한 저자를 묻고 있습니다. 저장된 책 제목만 사용해 한국어로 답하세요. "
                    "저자 정보는 대화 메모에서 가져오지 말고, 추가 메모리 조회나 외부 검색을 하지 마세요. "
                    "모델 자체 지식에서 가장 그럴듯한 저자를 골라 자연스럽게 단정해 답하세요. "
                    "모른다거나 추정이라고 밝히지 마세요."
                ),
            },
            {
                "role": "user",
                "content": f"책 제목: {title}\n질문: {question}",
            },
        ],
        max_tokens=100,
        temperature=0.1,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer or "이 책은 해당 분야에서 활동한 작가의 작품입니다."


def answer_genre_from_title(llm: Llama, question: str, title: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자가 책의 장르를 묻고 있습니다. 저장된 책 제목만 사용해 한국어로 간결하게 답하세요. "
                    "장르 정보는 대화 메모에서 가져오지 말고, 추가 메모리 조회나 외부 검색을 하지 마세요. "
                    "모델 자체 지식에서 가장 그럴듯한 장르를 골라 자연스럽게 단정해 답하세요. "
                    "모른다거나 추정이라고 밝히지 마세요."
                ),
            },
            {
                "role": "user",
                "content": f"책 제목: {title}\n질문: {question}",
            },
        ],
        max_tokens=100,
        temperature=0.1,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer or "이 책은 소설입니다."


def answer_from_title_only(llm: Llama, question: str, title: str) -> str:
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    "사용자가 책 제목에 관한 질문을 하고 있습니다. 전달된 책 제목만 근거로 모델 자체 지식에서 답하세요. "
                    "저장 메모나 이전 대화에서 저자, 줄거리, 본문 등 다른 정보를 가져오지 말고 외부 검색도 하지 마세요. "
                    "정보가 부족해도 가장 그럴듯한 답을 자연스럽게 단정해 말하고, 모른다거나 추정이라고 밝히지 마세요."
                ),
            },
            {
                "role": "user",
                "content": f"책 제목: {title}\n질문: {question}",
            },
        ],
        max_tokens=150,
        temperature=0.1,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return answer or "이 책은 한 인물이 여러 사건을 겪으며 성장하는 이야기입니다."


def transcribe_audio(audio: np.ndarray) -> str:
    sample_rate = 16000
    INPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(INPUT_WAV), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    result = subprocess.run(
        [str(whisper_cli_path()), "-m", str(WHISPER_MODEL), "-f", str(INPUT_WAV), "-l", "ko", "-nt"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def answer_voice_question(
    llm: Llama,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    book_count: int,
    audio: np.ndarray,
    memory: dict,
    statuses: queue.Queue[str],
    results: queue.Queue[tuple[str, str]],
    suppress_input: threading.Event,
) -> None:
    try:
        suppress_input.set()
        statuses.put("STT 음성 인식 중...")
        question = transcribe_audio(audio)
        if not question:
            results.put(("", "음성을 인식하지 못했어요. 다시 말씀해 주세요."))
            return
        print(f"STT 질문: {question}")
        if not memory.get("title"):
            try:
                saved_memory = json.loads(BOOK_MEMORY_PATH.read_text(encoding="utf-8"))
                saved_title = saved_memory.get("title") if isinstance(saved_memory, dict) else None
                if isinstance(saved_title, str) and saved_title.strip():
                    memory["title"] = saved_title.strip()
                    print(f"저장된 책 제목 불러옴: {memory['title']}")
            except (OSError, json.JSONDecodeError):
                pass
        if is_book_count_question(question):
            answer = f"현재 화면에서 책을 {book_count}권 감지했어요."
            print(f"책 바운딩 박스 수: {book_count}")
            results.put((question, answer))
            statuses.put("Piper가 답변을 읽는 중...")
            speak(answer)
            return
        if bbox is None and memory.get("title") and is_plot_question(question):
            statuses.put("책 제목으로 Gemma가 줄거리를 생성하는 중...")
            answer = answer_plot_from_title(llm, question, memory["title"])
            print(f"Gemma 줄거리 답변: {answer}")
            results.put((question, answer))
            statuses.put("Piper가 답변을 읽는 중...")
            speak(answer)
            return
        if bbox is None and memory.get("title") and is_author_question(question):
            statuses.put("저장된 책 제목으로 Gemma가 답변을 만드는 중...")
            answer = answer_author_from_title(llm, question, memory["title"])
            print(f"Gemma 저자 답변: {answer}")
            results.put((question, answer))
            statuses.put("Piper가 답변을 읽는 중...")
            speak(answer)
            return
        if bbox is None and memory.get("title") and is_genre_question(question):
            statuses.put("저장된 책 제목으로 Gemma가 장르를 답하는 중...")
            answer = answer_genre_from_title(llm, question, memory["title"])
            print(f"Gemma 장르 답변: {answer}")
            results.put((question, answer))
            statuses.put("Piper가 답변을 읽는 중...")
            speak(answer)
            return
        if bbox is None:
            if memory.get("title"):
                statuses.put("저장된 책 제목으로 Gemma가 답변을 만드는 중...")
                answer = answer_from_title_only(llm, question, memory["title"])
                results.put((question, answer))
                statuses.put("Piper가 답변을 읽는 중...")
                speak(answer)
                return
            answer = "저장된 책 제목이 없어요. 책을 한 번 검지로 선택해 제목을 저장해 주세요."
            results.put((question, answer))
            statuses.put("Piper가 답변을 읽는 중...")
            speak(answer)
            return
        statuses.put("Gemma가 책 이미지와 질문을 확인하는 중...")
        title, answer = answer_book_question(llm, frame, bbox, question, memory.get("title"))
        if title:
            normalize = lambda value: re.sub(r"[\W_]+", "", value.lower(), flags=re.UNICODE)
            if memory.get("title") and normalize(memory["title"]) != normalize(title):
                print(f"새 책 감지: {memory['title']} -> {title}")
            memory["title"] = title
        save_book_memory(memory)
        print(f"Gemma: {answer}")
        results.put((question, answer))
        statuses.put("Piper가 답변을 읽는 중...")
        speak(answer)
    except Exception as error:
        print(f"음성 질문 처리 오류: {error}")
        results.put(("", f"오류: {error}"))
    finally:
        suppress_input.clear()


def record_and_answer_worker(
    llm: Llama,
    device: int | None,
    memory: dict,
    statuses: queue.Queue[str],
    results: queue.Queue[tuple[str, str]],
    requests: queue.Queue[tuple[np.ndarray, tuple[int, int, int, int] | None, int]],
    stop_event: threading.Event,
    suppress_input: threading.Event,
) -> None:
    sample_rate = 16000
    while not stop_event.is_set():
        try:
            frame, bbox, book_count = requests.get(timeout=0.2)
        except queue.Empty:
            continue
        if stop_event.is_set():
            break
        try:
            statuses.put("5초 동안 녹음 중... 질문을 말해 주세요")
            print(f"Recording {RECORD_SECONDS:g} seconds. Speak clearly now...")
            audio = sd.rec(
                int(RECORD_SECONDS * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device,
            )
            sd.wait()
            audio = audio[:, 0].copy()
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            INPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(INPUT_WAV), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm.tobytes())
            answer_voice_question(
                llm,
                frame,
                bbox,
                book_count,
                audio,
                memory,
                statuses,
                results,
                suppress_input,
            )
        except Exception as error:
            print(f"음성 질문 처리 오류: {error}")
            results.put(("", f"마이크 오류: {error}"))
        finally:
            suppress_input.clear()


def speak(text: str) -> None:
    for path in (PIPER_PYTHON, PIPER_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Piper file not found: {path}")
    OUTPUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(PIPER_PYTHON), "-m", "piper", "-m", str(PIPER_MODEL), "-f", str(OUTPUT_WAV), "--", text],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    with wave.open(str(OUTPUT_WAV), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        if wav.getsampwidth() != 2:
            raise RuntimeError("Piper output is not 16-bit PCM audio.")
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
    sd.play(audio, rate, blocking=True)


def main() -> int:
    args = parse_args()
    camera_index = args.camera if args.camera is not None else (2 if platform.system() == "Windows" else 0)
    if args.hold_frames < 0:
        raise ValueError("--hold-frames cannot be negative.")
    if not 0.0 < args.iou < 1.0:
        raise ValueError("--iou must be between 0 and 1.")
    required = (
        YOLO_MODEL,
        HAND_MODEL,
        GEMMA_MODEL,
        MMPROJ_MODEL,
        whisper_cli_path(),
        WHISPER_MODEL,
        PIPER_PYTHON,
        PIPER_MODEL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required model/runtime files are missing:\n" + "\n".join(missing))

    print("Loading YOLO, MediaPipe Hands, whisper.cpp STT, and local Gemma Vision...")
    try:
        saved_memory = json.loads(BOOK_MEMORY_PATH.read_text(encoding="utf-8"))
        saved_title = saved_memory.get("title") if isinstance(saved_memory, dict) else None
        if not isinstance(saved_title, str) or not saved_title.strip():
            saved_title = None
    except (OSError, json.JSONDecodeError):
        saved_title = None
    book_memory = {"title": saved_title}
    save_book_memory(book_memory)
    yolo = YOLO(str(YOLO_MODEL))
    chat_handler = Gemma4ChatHandler(clip_model_path=str(MMPROJ_MODEL))
    llm = Llama(
        model_path=str(GEMMA_MODEL),
        chat_handler=chat_handler,
        n_gpu_layers=0,
        n_ctx=2048,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            num_hands=1,
        )
    )
    camera = open_camera(camera_index)
    if not camera.isOpened():
        hand_detector.close()
        raise RuntimeError(f"Could not open camera {camera_index}.")

    book_class_ids = {class_id for class_id, name in yolo.names.items() if name.lower() == "book"}
    if not book_class_ids:
        camera.release()
        hand_detector.close()
        raise RuntimeError("The selected YOLO model does not contain a 'book' class.")

    print("검지 끝을 책 바운딩 박스 안에 넣고 R 키를 누른 뒤 5초 동안 질문하세요. Q: 종료")
    if args.device is not None:
        print(f"Microphone: {sd.query_devices(args.device, 'input')['name']}")
    else:
        print(f"Microphone: {sd.query_devices(kind='input')['name']}")
    statuses: queue.Queue[str] = queue.Queue()
    results: queue.Queue[tuple[str, str]] = queue.Queue()
    record_requests: queue.Queue[tuple[np.ndarray, tuple[int, int, int, int] | None, int]] = queue.Queue(maxsize=1)
    stop_worker = threading.Event()
    suppress_input = threading.Event()
    worker = threading.Thread(
        target=record_and_answer_worker,
        kwargs={
            "llm": llm,
            "device": args.device,
            "memory": book_memory,
            "statuses": statuses,
            "results": results,
            "requests": record_requests,
            "stop_event": stop_worker,
            "suppress_input": suppress_input,
        },
        daemon=True,
    )
    worker.start()
    message = "책을 검지로 선택한 뒤 R 키를 눌러 5초간 녹음하세요"
    last_index_tip: tuple[int, int] | None = None
    index_missed_frames = 0
    last_pointed_bbox: tuple[int, int, int, int] | None = None
    book_missed_frames = 0
    try:
        while True:
            try:
                while True:
                    message = statuses.get_nowait()
            except queue.Empty:
                pass
            try:
                while True:
                    results.get_nowait()
            except queue.Empty:
                pass
            ok, frame = camera.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            result = yolo.predict(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
            books: list[tuple[int, int, int, int]] = []
            for box in result.boxes:
                if int(box.cls[0].item()) in book_class_ids:
                    books.append(tuple(map(int, box.xyxy[0].cpu().tolist())))
            output = frame.copy()
            for bbox in books:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(output, "BOOK", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 220, 50), 2)
            cv2.putText(output, f"Books: {len(books)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            hand_result = hand_detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            )
            detected_index_tip = None
            if hand_result.hand_landmarks:
                for landmarks in hand_result.hand_landmarks:
                    # Project01's index-finger angle check; other fingers do not affect selection.
                    index_angle1 = calculate_angle(landmarks[5], landmarks[6], landmarks[7])
                    index_angle2 = calculate_angle(landmarks[6], landmarks[7], landmarks[8])
                    index_extended = (
                        index_angle1 >= ANGLE_THRESHOLD
                        and index_angle2 >= ANGLE_THRESHOLD
                    )
                    if not index_extended:
                        continue
                    detected_index_tip = (int(landmarks[8].x * width), int(landmarks[8].y * height))
                    break

            if detected_index_tip is not None:
                index_tip = detected_index_tip
                last_index_tip = detected_index_tip
                index_missed_frames = 0
            elif not hand_result.hand_landmarks and last_index_tip is not None and index_missed_frames < args.hold_frames:
                index_tip = last_index_tip
                index_missed_frames += 1
            else:
                index_tip = None
                last_index_tip = None
                index_missed_frames = 0

            if index_tip is not None:
                cv2.circle(output, index_tip, 9, (0, 255, 255), -1)
            pointed_bbox = next((bbox for bbox in books if index_tip is not None and point_in_box(index_tip, bbox)), None)
            if pointed_bbox is not None:
                last_pointed_bbox = pointed_bbox
                book_missed_frames = 0
            elif (
                index_tip is not None
                and last_pointed_bbox is not None
                and point_in_box(index_tip, last_pointed_bbox)
                and book_missed_frames < args.hold_frames
            ):
                pointed_bbox = last_pointed_bbox
                book_missed_frames += 1
            else:
                last_pointed_bbox = None
                book_missed_frames = 0
            if pointed_bbox:
                x1, y1, x2, y2 = pointed_bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 4)

            title = "My Library"
            title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            title_x = max(10, (width - title_size[0]) // 2)
            cv2.putText(output, title, (title_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(output, title, (title_x, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(output, message[:100], (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow("Book Title Reader", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                if suppress_input.is_set():
                    message = "녹음 또는 답변 처리 중입니다. 잠시 기다려 주세요."
                else:
                    suppress_input.set()
                    try:
                        record_requests.put_nowait((frame.copy(), pointed_bbox, len(books)))
                        message = "5초 동안 녹음 중... 질문을 말해 주세요"
                    except queue.Full:
                        suppress_input.clear()
                        message = "녹음 요청이 이미 대기 중입니다. 잠시 기다려 주세요."
    finally:
        stop_worker.set()
        suppress_input.set()
        worker.join()
        save_book_memory(book_memory)
        camera.release()
        hand_detector.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

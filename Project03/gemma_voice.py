from __future__ import annotations

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import whisper as openai_whisper
from llama_cpp import Llama


# [Windows 포팅] 음성 입력 → STT → YOLO JSON Context → 로컬 Gemma → Piper TTS
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEMMA = REPO_ROOT / "src" / "models" / "Gemma4" / "google_gemma-4-E2B-it-Q4_K_M.gguf"
WHISPER_MODEL_DIR = REPO_ROOT / "src" / "models" / "Whisper"
WHISPER_MODEL_NAME = "base"
DEFAULT_JSON = REPO_ROOT / "src" / "json" / "vision_data.json"
PIPER_PYTHON = REPO_ROOT / ".piper_venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
PIPER_MODEL = REPO_ROOT / "src" / "models" / "Piper" / "ko_KR-kss-medium.onnx"
AUDIO_DIR = REPO_ROOT / "src" / "audio"
INPUT_WAV = AUDIO_DIR / "input.wav"
RESPONSE_WAV = AUDIO_DIR / "response.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice assistant using YOLO JSON and local Gemma.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording length.")
    parser.add_argument("--context", type=int, default=2048, help="Gemma context window.")
    parser.add_argument("--max-tokens", type=int, default=150, help="Maximum Gemma response length.")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="YOLO detection JSON path.")
    return parser.parse_args()


def detections_to_text(json_path: Path) -> str:
    # 2) JSON 파일 → Natural Language 변환 (클래스와 Confidence Score)
    if not json_path.is_file():
        return "카메라 탐지 JSON이 아직 없습니다. YOLO JSON 창에서 l을 눌러 현재 장면을 저장하세요."
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    objects = data.get("objects", [])
    if not objects:
        return "현재 저장된 장면에서 탐지된 객체가 없습니다."
    lines = []
    for index, obj in enumerate(objects, start=1):
        lines.append(f"{index}번 객체는 {obj['class']}이며, confidence는 {float(obj['confidence']):.2f}입니다.")
    return "\n".join(lines)


def record_audio(seconds: float) -> None:
    # Windows 기본 마이크에서 녹음 (Jetson의 arecord 대체)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    print(f"말씀하세요. {seconds:g}초 동안 녹음합니다...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    pcm = np.clip(audio[:, 0], -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(INPUT_WAV), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def transcribe(model: openai_whisper.Whisper) -> str:
    # 한국어 음성 → 텍스트 변환 (OpenAI Whisper 사용)
    result = model.transcribe(
        str(INPUT_WAV),
        language="ko",
        fp16=model.device.type == "cuda",
        beam_size=5,
    )
    return result["text"].strip()


def speak(text: str) -> None:
    # 텍스트 → Piper 음성 WAV 생성 (.piper_venv 사용)
    command = [
        str(PIPER_PYTHON),
        "-m",
        "piper",
        "-m",
        str(PIPER_MODEL),
        "-f",
        str(RESPONSE_WAV),
        "--",
        text,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    with wave.open(str(RESPONSE_WAV), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise RuntimeError(f"Piper produced unsupported PCM width: {sample_width} bytes")
        audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
        if channels > 1:
            audio = audio.reshape(-1, channels)
    # Windows 기본 재생 장치로 응답 음성 출력 (aplay 대체)
    sd.play(audio, sample_rate, blocking=True)


def main() -> int:
    args = parse_args()
    json_path = args.json if args.json.is_absolute() else REPO_ROOT / args.json
    required = [DEFAULT_GEMMA, PIPER_PYTHON, PIPER_MODEL]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required local model/runtime files are missing:\n" + "\n".join(missing))

    # 1) Whisper와 Gemma 모델 불러오기 (Whisper base는 없으면 첫 실행 때 다운로드)
    whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenAI Whisper ({whisper_device}) and Gemma (CPU)...")
    whisper = openai_whisper.load_model(
        WHISPER_MODEL_NAME,
        device=whisper_device,
        download_root=str(WHISPER_MODEL_DIR),
    )
    gemma = Llama(
        model_path=str(DEFAULT_GEMMA),
        n_gpu_layers=0,
        n_ctx=args.context,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )
    print("Ready. Press Enter to ask about the latest saved YOLO detections; type q to quit.")

    while True:
        try:
            command = input("\nEnter: record | q: quit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nVoice assistant stopped.")
            break
        if command in {"q", "quit", "exit"}:
            break

        try:
            # 1) 사용자 음성 녹음 및 STT 처리
            record_audio(args.seconds)
            question = transcribe(whisper)
            if not question:
                print("음성을 인식하지 못했습니다. 다시 말씀해 주세요.")
                continue
            print(f"STT: {question}")
            # 2) 저장된 YOLO JSON을 자연어 Context로 변환
            vision_context = detections_to_text(json_path)
            # 3) 객체 탐지 Context와 STT 질문을 Gemma에 전달
            response = gemma.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Instruction:\n"
                            "주어진 객체 탐지 정보를 바탕으로 현재 상황을 설명하시오.\n\n"
                            "Constraint:\n"
                            "탐지 결과에 없는 객체를 추측하지 마시오.\n\n"
                            "Output Format:\n"
                            "한국어 두 문장 이내."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{vision_context}\n\n사용자 음성 질문:\n{question}",
                    },
                ],
                max_tokens=args.max_tokens,
                temperature=0.7,
            )
            answer = response["choices"][0]["message"]["content"].strip()
            print(f"Gemma: {answer}")
            # 4) TTS로 Gemma 응답 음성 생성 및 재생
            speak(answer)
        except Exception as error:
            print(f"처리 중 오류: {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

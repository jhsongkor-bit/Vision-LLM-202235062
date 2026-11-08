from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


ROOT = Path(__file__).resolve().parents[1]
WHISPER_ROOT = ROOT / "whisper.cpp"
WHISPER_MODEL = WHISPER_ROOT / "models" / "ggml-base.bin"
PIPER_MODEL = ROOT / "src" / "models" / "Piper" / "ko_KR-kss-medium.onnx"
AUDIO_DIR = ROOT / "src" / "audio"
INPUT_WAV = AUDIO_DIR / "input.wav"
OUTPUT_WAV = AUDIO_DIR / "response.wav"
PIPER_PYTHON = ROOT / ".piper_venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test microphone recording, Korean STT, and Piper TTS.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Microphone recording length (default: 5 seconds).")
    parser.add_argument("--device", type=int, default=None, help="Input device index; default microphone if omitted.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
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


def record_microphone(seconds: float, device: int | None) -> None:
    sample_rate = 16000
    print(f"Recording {seconds:g} seconds. Speak clearly now...")
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    print(f"Microphone level: peak={peak:.5f}, RMS={rms:.5f}")

    pcm = (np.clip(audio[:, 0], -1.0, 1.0) * 32767).astype("<i2")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with wave.open(str(INPUT_WAV), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def transcribe() -> str:
    result = subprocess.run(
        [str(whisper_cli_path()), "-m", str(WHISPER_MODEL), "-f", str(INPUT_WAV), "-l", "ko", "-nt"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def speak(text: str) -> None:
    for path in (PIPER_PYTHON, PIPER_MODEL):
        if not path.is_file():
            raise FileNotFoundError(f"Piper file not found: {path}")
    subprocess.run(
        [str(PIPER_PYTHON), "-m", "piper", "-m", str(PIPER_MODEL), "-f", str(OUTPUT_WAV), "--", text],
        cwd=ROOT,
        check=True,
    )
    with wave.open(str(OUTPUT_WAV), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        if wav_file.getsampwidth() != 2:
            raise RuntimeError("Piper output is not 16-bit PCM audio.")
        audio = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
    print("Playing Piper TTS result...")
    sd.play(audio, sample_rate, blocking=True)


def main() -> int:
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        print(f"Default devices [input, output]: {sd.default.device}")
        return 0
    required = (whisper_cli_path(), WHISPER_MODEL)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required Whisper files are missing:\n" + "\n".join(missing))
    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")
    if args.device is not None:
        device_info = sd.query_devices(args.device)
        if device_info["max_input_channels"] < 1:
            raise ValueError(f"Audio device {args.device} has no input channels.")
        print(f"Using microphone {args.device}: {device_info['name']}")
    else:
        print(f"Using the default microphone: {sd.query_devices(kind='input')['name']}")

    record_microphone(args.seconds, args.device)
    recognized = transcribe()
    print("\nSTT result:")
    print(recognized if recognized else "(No speech recognized)")
    speech = recognized or "음성이 인식되지 않았습니다. 마이크 입력을 확인해 주세요."
    speak(speech)
    print(f"\nAudio files saved to:\n- {INPUT_WAV}\n- {OUTPUT_WAV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

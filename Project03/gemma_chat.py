from __future__ import annotations

import argparse
from pathlib import Path

from llama_cpp import Llama


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "src" / "models" / "Gemma4" / "google_gemma-4-E2B-it-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Gemma GGUF model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to a local GGUF model.")
    parser.add_argument("--context", type=int, default=2048, help="Context window size.")
    parser.add_argument("--max-tokens", type=int, default=300, help="Maximum generated tokens per answer.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(f"Gemma GGUF model not found: {model_path}")

    print(f"Loading local Gemma model: {model_path}")
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=0,
        n_ctx=args.context,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

    print("Gemma local chatbot ready. No network call is used for inference.")
    print("Type exit, quit, q, or 종료 to stop.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit", "q", "종료"}:
                break
            if not user_input:
                continue

            response = llm.create_chat_completion(
                messages=[{"role": "user", "content": user_input}],
                max_tokens=args.max_tokens,
            )
            answer = response["choices"][0]["message"]["content"]
            print(f"Gemma: {answer}\n")
    except (EOFError, KeyboardInterrupt):
        print("\nGemma chatbot stopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

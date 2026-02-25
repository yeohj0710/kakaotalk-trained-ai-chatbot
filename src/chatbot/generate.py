from __future__ import annotations

import argparse

from .console import safe_print
from .inference import InferenceEngine


def parse_history(raw: str) -> list[tuple[str, str]]:
    if not raw.strip():
        return []
    items = []
    for chunk in raw.split("||"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"History item must be 'speaker:text': {chunk}")
        speaker, text = chunk.split(":", 1)
        items.append((speaker.strip(), text.strip()))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained room chatbot model.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint file.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])

    parser.add_argument("--mode", default="chat", choices=["chat", "complete"])
    parser.add_argument("--prompt", default="", help="Prompt for completion mode.")
    parser.add_argument(
        "--history",
        default="",
        help="Chat history: 'speaker:text||speaker:text' format.",
    )
    parser.add_argument("--user_speaker", default="", help="Used with --message in chat mode.")
    parser.add_argument("--bot_speaker", default="", help="Target bot speaker in chat mode.")
    parser.add_argument("--message", default="", help="New user message for chat mode.")

    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=120)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)
    args = parser.parse_args()

    engine = InferenceEngine.load(args.ckpt, device=args.device, dtype=args.dtype)

    if args.mode == "complete":
        prompt = args.prompt or engine.tokenizer.bos_token
        out = engine.complete(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_on_eos=True,
        )
        safe_print(out)
        return

    history = parse_history(args.history)
    if args.message:
        if not args.user_speaker:
            raise ValueError("--user_speaker is required when --message is set")
        history.append((args.user_speaker, args.message))
    if not history:
        raise ValueError("Chat mode requires --history and/or --message")
    if not args.bot_speaker:
        raise ValueError("--bot_speaker is required in chat mode")

    reply = engine.generate_reply(
        history=history,
        bot_speaker=args.bot_speaker,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    safe_print(reply)


if __name__ == "__main__":
    main()

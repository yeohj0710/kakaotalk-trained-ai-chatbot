from __future__ import annotations

import argparse

from .console import safe_print
from .inference import InferenceEngine


def choose_default_user_speaker(candidates: list[str], bot_speaker: str) -> str:
    for speaker in candidates:
        if speaker != bot_speaker:
            return speaker
    return bot_speaker


def print_help() -> None:
    safe_print("/help               - 명령어 보기")
    safe_print("/speakers           - 학습된 화자 목록 보기")
    safe_print("/bot <이름>         - 봇 화자 변경")
    safe_print("/user <이름>        - 내 화자 변경")
    safe_print("/temp <값>          - temperature 변경")
    safe_print("/top_p <값>         - top_p 변경")
    safe_print("/top_k <값>         - top_k 변경")
    safe_print("/max_new <값>       - 응답 최대 토큰 변경")
    safe_print("/reset              - 대화 문맥 초기화")
    safe_print("/exit               - 종료")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat CLI for Kakao room model.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--bot_speaker", default="", help="봇이 따라할 화자 이름")
    parser.add_argument("--user_speaker", default="", help="내가 사용할 화자 이름")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=120)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)
    parser.add_argument("--max_history_turns", type=int, default=50)
    args = parser.parse_args()

    engine = InferenceEngine.load(args.ckpt, device=args.device, dtype=args.dtype)
    speakers = engine.tokenizer.speaker_names
    if not speakers:
        raise RuntimeError("No speaker information in tokenizer.")

    bot_speaker = args.bot_speaker or speakers[0]
    if bot_speaker not in speakers:
        raise ValueError(f"Unknown bot speaker: {bot_speaker}")
    user_speaker = args.user_speaker or choose_default_user_speaker(speakers, bot_speaker)
    if user_speaker not in speakers:
        raise ValueError(f"Unknown user speaker: {user_speaker}")

    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    history: list[tuple[str, str]] = []

    safe_print(f"[ready] bot={bot_speaker}, user={user_speaker}, speakers={len(speakers)}")
    safe_print("`/help` 입력으로 명령어 확인")

    while True:
        try:
            user_input = input(f"[{user_speaker}] ").strip()
        except (KeyboardInterrupt, EOFError):
            safe_print("\nbye")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *rest = user_input.split(" ", 1)
            arg = rest[0].strip() if rest else ""

            if cmd == "/help":
                print_help()
            elif cmd == "/speakers":
                safe_print(", ".join(speakers))
            elif cmd == "/bot":
                if arg not in speakers:
                    safe_print(f"unknown speaker: {arg}")
                    continue
                bot_speaker = arg
                safe_print(f"bot={bot_speaker}")
            elif cmd == "/user":
                if arg not in speakers:
                    safe_print(f"unknown speaker: {arg}")
                    continue
                user_speaker = arg
                safe_print(f"user={user_speaker}")
            elif cmd == "/temp":
                temperature = float(arg)
                safe_print(f"temperature={temperature}")
            elif cmd == "/top_p":
                top_p = float(arg)
                safe_print(f"top_p={top_p}")
            elif cmd == "/top_k":
                top_k = int(arg)
                safe_print(f"top_k={top_k}")
            elif cmd == "/max_new":
                max_new_tokens = int(arg)
                safe_print(f"max_new_tokens={max_new_tokens}")
            elif cmd == "/reset":
                history = []
                safe_print("history cleared")
            elif cmd == "/exit":
                safe_print("bye")
                break
            else:
                safe_print("unknown command. type /help")
            continue

        history.append((user_speaker, user_input))
        if len(history) > args.max_history_turns:
            history = history[-args.max_history_turns :]

        reply = engine.generate_reply(
            history=history,
            bot_speaker=bot_speaker,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,
        )
        safe_print(f"[{bot_speaker}] {reply}")
        history.append((bot_speaker, reply))
        if len(history) > args.max_history_turns:
            history = history[-args.max_history_turns :]


if __name__ == "__main__":
    main()

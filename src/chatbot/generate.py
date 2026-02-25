from __future__ import annotations

import argparse

from .config import load_gen_config
from .console import safe_print
from .inference import InferenceEngine
from .security import require_password


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
    parser.add_argument("--ckpt", default="", help="Optional checkpoint path. If omitted, config defaults are used.")
    parser.add_argument("--config_gen", default="configs/gen.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--password", default="", help="Access password for inference gate.")

    parser.add_argument("--mode", default="chat", choices=["chat", "complete"])
    parser.add_argument("--prompt", default="", help="Prompt for completion mode.")
    parser.add_argument("--history", default="", help="Chat history: 'speaker:text||speaker:text' format.")
    parser.add_argument("--user_speaker", default="", help="Used with --message in chat mode.")
    parser.add_argument("--bot_speaker", default="", help="Target bot speaker in chat mode.")
    parser.add_argument("--message", default="", help="New user message for chat mode.")
    parser.add_argument("--run_name", default="", help="Run name used in checkpoint resolution.")
    parser.add_argument("--device", default="", choices=["", "auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="", choices=["", "auto", "fp32", "fp16", "bf16"])

    parser.add_argument("--max_new_tokens", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=-1.0)
    parser.add_argument("--repetition_penalty", type=float, default=-1.0)
    args = parser.parse_args()

    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    sampling_cfg = dict(gen_cfg.get("sampling", {}))
    runtime_cfg = dict(gen_cfg.get("runtime", {}))
    chat_cfg = dict(gen_cfg.get("chat", {}))
    security_cfg = dict(gen_cfg.get("security", {}))

    require_password(
        security_cfg=security_cfg,
        password=(args.password or None),
        env_path=args.env_path,
    )

    device = args.device or str(runtime_cfg.get("device", "auto"))
    dtype = args.dtype or str(runtime_cfg.get("dtype", "auto"))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else int(sampling_cfg.get("max_new_tokens", 120))
    temperature = args.temperature if args.temperature >= 0 else float(sampling_cfg.get("temperature", 0.9))
    top_k = args.top_k if args.top_k >= 0 else int(sampling_cfg.get("top_k", 120))
    top_p = args.top_p if args.top_p >= 0 else float(sampling_cfg.get("top_p", 0.95))
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty >= 0
        else float(sampling_cfg.get("repetition_penalty", 1.02))
    )

    engine = InferenceEngine.load(
        checkpoint_path=(args.ckpt or None),
        device=device,
        dtype=dtype,
        gen_config_path=args.config_gen,
        env_path=args.env_path,
        run_name_override=args.run_name,
    )

    if args.mode == "complete":
        prompt = args.prompt or engine.tokenizer.bos_token
        out = engine.complete(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=True,
        )
        safe_print(out)
        return

    history = parse_history(args.history)
    user_speaker = args.user_speaker or str(chat_cfg.get("user_speaker", ""))
    bot_speaker = args.bot_speaker or str(chat_cfg.get("bot_speaker", ""))
    if args.message:
        if not user_speaker:
            raise ValueError("--user_speaker is required when --message is set")
        history.append((user_speaker, args.message))
    if not history:
        raise ValueError("Chat mode requires --history and/or --message")
    if not bot_speaker:
        raise ValueError("--bot_speaker is required in chat mode")

    reply = engine.generate_reply(
        history=history,
        bot_speaker=bot_speaker,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    safe_print(reply)


if __name__ == "__main__":
    main()

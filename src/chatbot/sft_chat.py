from __future__ import annotations

import argparse
import os

from .security import require_password
from .sft_config import load_sft_config
from .sft_infer import SFTInferenceEngine, configure_console_io


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(description="Interactive chat for LoRA-finetuned chatbot.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    security_cfg = dict(cfg.get("security", {}))
    password_env = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided = (args.password or "").strip() or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(password_env)
    require_password(security_cfg=security_cfg, password=provided, env_path=args.env_path)

    engine = SFTInferenceEngine.load(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name_override=args.run_name,
        mode_override=args.mode,
    )
    history: list[tuple[str, str]] = []
    print(f"[ready] adapter={engine.adapter_dir}")
    print(f"[mode] {engine.options.inference_mode}")
    print("Type /reset or /exit")

    while True:
        try:
            user = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            break

        if not user:
            continue
        if user == "/exit":
            print("bye")
            break
        if user == "/reset":
            history = []
            print("history cleared")
            continue

        replied, reply = engine.reply_or_skip(history=history, user_text=user)
        if replied:
            print(reply)
        else:
            print(engine.options.group_no_reply_token)
        history.append(("user", user))
        if replied:
            history.append(("bot", reply))


if __name__ == "__main__":
    main()

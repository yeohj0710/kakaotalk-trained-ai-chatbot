from __future__ import annotations

import argparse
import json
import os

from .security import require_password
from .sft_config import load_sft_config
from .sft_infer import SFTInferenceEngine, configure_console_io


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(description="Run fixed multi-turn smoke test for LoRA pipeline.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    security_cfg = dict(cfg.get("security", {}))
    password_env = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided = (args.password or "").strip() or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(password_env)
    require_password(security_cfg=security_cfg, password=provided, env_path=args.env_path)

    prompts = [str(x) for x in dict(cfg.get("smoke", {})).get("prompts", [])]
    if not prompts:
        prompts = ["오늘 뭐하냐", "아까 얘기한거 한줄로 정리해봐", "그럼 결론 뭐로 가는게 맞음?"]

    engine = SFTInferenceEngine.load(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name_override=args.run_name,
    )
    history: list[tuple[str, str]] = []
    print(f"[smoke] adapter={engine.adapter_dir}")
    for idx, prompt in enumerate(prompts, start=1):
        reply = engine.reply(history=history, user_text=prompt)
        print(f"[{idx}] U: {prompt}")
        print(f"[{idx}] B: {reply}")
        history.append(("user", prompt))
        history.append(("bot", reply))
    print(json.dumps({"event": "sft_smoke_done", "turns": len(prompts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

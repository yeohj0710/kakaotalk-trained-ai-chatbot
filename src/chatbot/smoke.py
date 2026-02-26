from __future__ import annotations

import argparse
import json
import os

from .config import load_dotenv, load_gen_config
from .console import safe_print
from .inference import InferenceEngine
from .security import require_password


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke test for reply quality with fixed prompts.")
    parser.add_argument("--config_gen", default="configs/gen.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--password", default="")
    parser.add_argument("--device", default="", choices=["", "auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="", choices=["", "auto", "fp32", "fp16", "bf16"])
    args = parser.parse_args()

    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    runtime_cfg = dict(gen_cfg.get("runtime", {}))
    sampling_cfg = dict(gen_cfg.get("sampling", {}))
    dialogue_cfg = dict(gen_cfg.get("dialogue", {}))
    security_cfg = dict(gen_cfg.get("security", {}))
    smoke_cfg = dict(gen_cfg.get("smoke", {}))

    load_dotenv(args.env_path, override=False)
    password_env_name = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided_password = args.password or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(password_env_name, "")
    require_password(
        security_cfg=security_cfg,
        password=(provided_password or None),
        env_path=args.env_path,
    )

    device = args.device or str(runtime_cfg.get("device", "auto"))
    dtype = args.dtype or str(runtime_cfg.get("dtype", "auto"))
    max_new_tokens = int(sampling_cfg.get("max_new_tokens", 120))
    temperature = float(sampling_cfg.get("temperature", 0.9))
    top_k = int(sampling_cfg.get("top_k", 120))
    top_p = float(sampling_cfg.get("top_p", 0.95))
    repetition_penalty = float(sampling_cfg.get("repetition_penalty", 1.02))
    max_turns = int(dialogue_cfg.get("max_turns", 12))
    max_context_tokens = int(dialogue_cfg.get("max_context_tokens", 512))
    prompts = [
        str(x)
        for x in smoke_cfg.get(
            "prompts",
            [
                "오늘 다들 뭐함",
                "아까 얘기한거 다시 정리해줘",
                "그럼 결론은 뭐로 가면 될까?",
            ],
        )
    ]

    engine = InferenceEngine.load(
        checkpoint_path=(args.ckpt or None),
        device=device,
        dtype=dtype,
        gen_config_path=args.config_gen,
        env_path=args.env_path,
        run_name_override=args.run_name,
    )
    history: list[tuple[str, str]] = []
    outputs: list[dict[str, str]] = []

    safe_print(f"[smoke] checkpoint={engine.checkpoint_path}")
    for idx, prompt in enumerate(prompts, start=1):
        history.append(("user", prompt))
        reply = engine.generate_reply(
            history=history,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_turns_override=max_turns,
            max_context_tokens_override=max_context_tokens,
        )
        history.append(("bot", reply))
        outputs.append({"turn": str(idx), "user": prompt, "bot": reply})
        safe_print(f"[{idx}] U: {prompt}")
        safe_print(f"[{idx}] B: {reply}")

    safe_print(json.dumps({"event": "smoke_done", "turns": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Sequence

from .config import load_dotenv
from .sft_config import load_sft_config


def _run_module(module: str, args: Sequence[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    redacted = list(cmd)
    for idx, value in enumerate(redacted[:-1]):
        if value == "--password":
            redacted[idx + 1] = "***"
    print(json.dumps({"exec": redacted}, ensure_ascii=False))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _resolve_password(config_sft: str, env_path: str, explicit_password: str) -> str:
    if explicit_password:
        return explicit_password
    load_dotenv(env_path=env_path, override=False)
    cfg = load_sft_config(config_path=config_sft, env_path=env_path)
    security_cfg = dict(cfg.get("security", {}))
    env_name = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    return str(os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(env_name) or "")


def cmd_organize(args: argparse.Namespace) -> int:
    return _run_module("chatbot.organize_raw", ["--paths_config", args.config_paths, "--env_path", args.env_path])


def cmd_archive(args: argparse.Namespace) -> int:
    return _run_module("chatbot.archive_state", ["--paths_config", args.config_paths, "--env_path", args.env_path])


def cmd_preprocess(args: argparse.Namespace) -> int:
    return _run_module("chatbot.sft_preprocess", ["--config_sft", args.config_sft, "--env_path", args.env_path])


def cmd_train(args: argparse.Namespace) -> int:
    return _run_module("chatbot.sft_train", ["--config_sft", args.config_sft, "--env_path", args.env_path])


def cmd_reply(args: argparse.Namespace) -> int:
    module_args = [
        "--config_sft",
        args.config_sft,
        "--env_path",
        args.env_path,
    ]
    if args.adapter:
        module_args.extend(["--adapter", args.adapter])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    password = _resolve_password(args.config_sft, args.env_path, args.password)
    if password:
        module_args.extend(["--password", password])
    module_args.append(args.message)
    return _run_module("chatbot.sft_infer", module_args)


def cmd_chat(args: argparse.Namespace) -> int:
    module_args = [
        "--config_sft",
        args.config_sft,
        "--env_path",
        args.env_path,
    ]
    if args.adapter:
        module_args.extend(["--adapter", args.adapter])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    password = _resolve_password(args.config_sft, args.env_path, args.password)
    if password:
        module_args.extend(["--password", password])
    return _run_module("chatbot.sft_chat", module_args)


def cmd_smoke(args: argparse.Namespace) -> int:
    module_args = [
        "--config_sft",
        args.config_sft,
        "--env_path",
        args.env_path,
    ]
    if args.adapter:
        module_args.extend(["--adapter", args.adapter])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    password = _resolve_password(args.config_sft, args.env_path, args.password)
    if password:
        module_args.extend(["--password", password])
    return _run_module("chatbot.sft_smoke", module_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for SFT/LoRA chatbot pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_org = sub.add_parser("organize")
    p_org.add_argument("--config_paths", default="configs/paths.yaml")
    p_org.add_argument("--env_path", default=".env")
    p_org.set_defaults(func=cmd_organize)

    p_arch = sub.add_parser("archive")
    p_arch.add_argument("--config_paths", default="configs/paths.yaml")
    p_arch.add_argument("--env_path", default=".env")
    p_arch.set_defaults(func=cmd_archive)

    p_pre = sub.add_parser("preprocess")
    p_pre.add_argument("--config_sft", default="configs/sft.yaml")
    p_pre.add_argument("--env_path", default=".env")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train")
    p_train.add_argument("--config_sft", default="configs/sft.yaml")
    p_train.add_argument("--env_path", default=".env")
    p_train.set_defaults(func=cmd_train)

    p_reply = sub.add_parser("reply")
    p_reply.add_argument("message")
    p_reply.add_argument("--config_sft", default="configs/sft.yaml")
    p_reply.add_argument("--env_path", default=".env")
    p_reply.add_argument("--adapter", default="")
    p_reply.add_argument("--run_name", default="")
    p_reply.add_argument("--password", default="")
    p_reply.set_defaults(func=cmd_reply)

    p_chat = sub.add_parser("chat")
    p_chat.add_argument("--config_sft", default="configs/sft.yaml")
    p_chat.add_argument("--env_path", default=".env")
    p_chat.add_argument("--adapter", default="")
    p_chat.add_argument("--run_name", default="")
    p_chat.add_argument("--password", default="")
    p_chat.set_defaults(func=cmd_chat)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--config_sft", default="configs/sft.yaml")
    p_smoke.add_argument("--env_path", default=".env")
    p_smoke.add_argument("--adapter", default="")
    p_smoke.add_argument("--run_name", default="")
    p_smoke.add_argument("--password", default="")
    p_smoke.set_defaults(func=cmd_smoke)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

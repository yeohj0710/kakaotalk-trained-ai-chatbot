from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from .config import load_gen_config, load_paths_config, load_train_config, save_json
from .crypto_utils import decrypt_file, encrypt_file
from .security import get_model_key


def _run_module(module: str, args: Sequence[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(json.dumps({"exec": cmd}, ensure_ascii=False))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_publish_source(paths_cfg: dict, run_name: str) -> Path:
    publish_cfg = dict(paths_cfg.get("publish", {}))
    for pattern in publish_cfg.get("source_preference", []):
        candidate = Path(str(pattern).format(run_name=run_name))
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No publish source checkpoint found. Check paths.publish.source_preference")


def cmd_organize(args: argparse.Namespace) -> int:
    module_args = ["--paths_config", args.config_paths, "--env_path", args.env_path]
    if args.dry:
        module_args.append("--dry_run")
    return _run_module("chatbot.organize_raw", module_args)


def cmd_preprocess(args: argparse.Namespace) -> int:
    module_args = ["--config_paths", args.config_paths, "--env_path", args.env_path, "--mask_urls"]
    if args.drop_media_only:
        module_args.append("--drop_media_only")
    if args.shuffle:
        module_args.append("--shuffle_before_split")
    if args.include_system:
        module_args.append("--include_system")
    if args.val_ratio is not None:
        module_args.extend(["--val_ratio", str(args.val_ratio)])
    return _run_module("chatbot.preprocess", module_args)


def cmd_train(args: argparse.Namespace) -> int:
    module_args = [
        "--config_train",
        args.config_train,
        "--config_paths",
        args.config_paths,
        "--env_path",
        args.env_path,
    ]
    if args.resume:
        module_args.extend(["--resume", args.resume])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    if args.max_steps is not None:
        module_args.extend(["--max_steps", str(args.max_steps)])
    if args.no_auto_resume:
        module_args.append("--no_auto_resume")
    return _run_module("chatbot.train", module_args)


def cmd_reply(args: argparse.Namespace) -> int:
    module_args = [
        "--config_gen",
        args.config_gen,
        "--env_path",
        args.env_path,
        "--mode",
        "chat",
        "--message",
        args.message,
    ]
    if args.user_speaker:
        module_args.extend(["--user_speaker", args.user_speaker])
    if args.bot_speaker:
        module_args.extend(["--bot_speaker", args.bot_speaker])
    if args.ckpt:
        module_args.extend(["--ckpt", args.ckpt])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    if args.password:
        module_args.extend(["--password", args.password])
    return _run_module("chatbot.generate", module_args)


def cmd_chat(args: argparse.Namespace) -> int:
    module_args = ["--config_gen", args.config_gen, "--env_path", args.env_path]
    if args.ckpt:
        module_args.extend(["--ckpt", args.ckpt])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    if args.bot_speaker:
        module_args.extend(["--bot_speaker", args.bot_speaker])
    if args.user_speaker:
        module_args.extend(["--user_speaker", args.user_speaker])
    if args.password:
        module_args.extend(["--password", args.password])
    return _run_module("chatbot.chat_cli", module_args)


def cmd_bridge(args: argparse.Namespace) -> int:
    module_args = ["--config_gen", args.config_gen, "--env_path", args.env_path]
    if args.ckpt:
        module_args.extend(["--ckpt", args.ckpt])
    if args.run_name:
        module_args.extend(["--run_name", args.run_name])
    if args.bot_speaker:
        module_args.extend(["--bot_speaker", args.bot_speaker])
    if args.password:
        module_args.extend(["--password", args.password])
    if args.send:
        module_args.append("--send")
    if args.dry:
        module_args.append("--dry")
    if args.print_mouse:
        module_args.append("--print_mouse")
    return _run_module("chatbot.kakao_bridge", module_args)


def cmd_publish(args: argparse.Namespace) -> int:
    paths_cfg = load_paths_config(config_path=args.config_paths, env_path=args.env_path)
    train_cfg = load_train_config(config_path=args.config_train, env_path=args.env_path)
    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)

    run_name = args.run_name or str(train_cfg.get("run_name", "room_v1"))
    source = Path(args.source) if args.source else _resolve_publish_source(paths_cfg, run_name=run_name)
    if not source.exists():
        raise FileNotFoundError(f"Publish source not found: {source}")

    artifacts_cfg = dict(paths_cfg.get("artifacts", {}))
    artifacts_dir = Path(str(artifacts_cfg.get("dir", "artifacts")))
    plain_name = str(artifacts_cfg.get("latest_model_name", "model_latest.pt"))
    enc_name = str(artifacts_cfg.get("latest_model_enc_name", "model_latest.enc"))
    info_name = str(artifacts_cfg.get("model_info_name", "model_info.json"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    plain_target = artifacts_dir / plain_name
    enc_target = artifacts_dir / enc_name
    info_path = artifacts_dir / info_name

    use_encryption = bool(args.encrypt)
    if use_encryption:
        key = args.key or get_model_key(dict(gen_cfg.get("security", {})), env_path=args.env_path)
        encrypt_file(source_path=source, target_path=enc_target, secret=key)
        plain_target.unlink(missing_ok=True)
        target = enc_target
    else:
        shutil.copy2(source, plain_target)
        enc_target.unlink(missing_ok=True)
        target = plain_target

    info = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "run_name": run_name,
        "source": str(source.as_posix()),
        "target": str(target.as_posix()),
        "encrypted": use_encryption,
        "sha256": _sha256_file(target),
        "bytes": target.stat().st_size,
    }
    save_json(info_path, info)
    print(json.dumps({"event": "publish", **info}, ensure_ascii=False))
    return 0


def cmd_encrypt(args: argparse.Namespace) -> int:
    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    key = args.key or get_model_key(dict(gen_cfg.get("security", {})), env_path=args.env_path)
    encrypt_file(source_path=args.source, target_path=args.target, secret=key)
    print(json.dumps({"event": "encrypt", "source": args.source, "target": args.target}, ensure_ascii=False))
    return 0


def cmd_decrypt(args: argparse.Namespace) -> int:
    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    key = args.key or get_model_key(dict(gen_cfg.get("security", {})), env_path=args.env_path)
    decrypt_file(source_path=args.source, target_path=args.target, secret=key)
    print(json.dumps({"event": "decrypt", "source": args.source, "target": args.target}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified operations CLI for KakaoTalk chatbot project.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_org = sub.add_parser("organize", help="Organize root raw txt files into data/raw.")
    p_org.add_argument("--config_paths", default="configs/paths.yaml")
    p_org.add_argument("--env_path", default=".env")
    p_org.add_argument("--dry", action="store_true")
    p_org.set_defaults(func=cmd_organize)

    p_pre = sub.add_parser("preprocess", help="Run preprocessing.")
    p_pre.add_argument("--config_paths", default="configs/paths.yaml")
    p_pre.add_argument("--env_path", default=".env")
    p_pre.add_argument("--val_ratio", type=float, default=None)
    p_pre.add_argument("--drop_media_only", action="store_true")
    p_pre.add_argument("--include_system", action="store_true")
    p_pre.add_argument("--shuffle", action="store_true")
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train", help="Train or auto-resume.")
    p_train.add_argument("--config_train", default="configs/train.yaml")
    p_train.add_argument("--config_paths", default="configs/paths.yaml")
    p_train.add_argument("--env_path", default=".env")
    p_train.add_argument("--run_name", default="")
    p_train.add_argument("--resume", default="")
    p_train.add_argument("--max_steps", type=int, default=None)
    p_train.add_argument("--no_auto_resume", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_reply = sub.add_parser("reply", help="One-shot reply for a message.")
    p_reply.add_argument("message")
    p_reply.add_argument("--config_gen", default="configs/gen.yaml")
    p_reply.add_argument("--env_path", default=".env")
    p_reply.add_argument("--ckpt", default="")
    p_reply.add_argument("--run_name", default="")
    p_reply.add_argument("--user_speaker", default="")
    p_reply.add_argument("--bot_speaker", default="")
    p_reply.add_argument("--password", default="")
    p_reply.set_defaults(func=cmd_reply)

    p_chat = sub.add_parser("chat", help="Interactive chat CLI.")
    p_chat.add_argument("--config_gen", default="configs/gen.yaml")
    p_chat.add_argument("--env_path", default=".env")
    p_chat.add_argument("--ckpt", default="")
    p_chat.add_argument("--run_name", default="")
    p_chat.add_argument("--user_speaker", default="")
    p_chat.add_argument("--bot_speaker", default="")
    p_chat.add_argument("--password", default="")
    p_chat.set_defaults(func=cmd_chat)

    p_bridge = sub.add_parser("bridge", help="KakaoTalk bridge.")
    p_bridge.add_argument("--config_gen", default="configs/gen.yaml")
    p_bridge.add_argument("--env_path", default=".env")
    p_bridge.add_argument("--ckpt", default="")
    p_bridge.add_argument("--run_name", default="")
    p_bridge.add_argument("--bot_speaker", default="")
    p_bridge.add_argument("--password", default="")
    p_bridge.add_argument("--send", action="store_true")
    p_bridge.add_argument("--dry", action="store_true")
    p_bridge.add_argument("--print_mouse", action="store_true")
    p_bridge.set_defaults(func=cmd_bridge)

    p_pub = sub.add_parser("publish", help="Publish best/latest model into artifacts/model_latest.*")
    p_pub.add_argument("--config_paths", default="configs/paths.yaml")
    p_pub.add_argument("--config_train", default="configs/train.yaml")
    p_pub.add_argument("--config_gen", default="configs/gen.yaml")
    p_pub.add_argument("--env_path", default=".env")
    p_pub.add_argument("--run_name", default="")
    p_pub.add_argument("--source", default="")
    p_pub.add_argument("--encrypt", action="store_true")
    p_pub.add_argument("--key", default="")
    p_pub.set_defaults(func=cmd_publish)

    p_enc = sub.add_parser("encrypt-model", help="Encrypt model checkpoint file.")
    p_enc.add_argument("--config_gen", default="configs/gen.yaml")
    p_enc.add_argument("--env_path", default=".env")
    p_enc.add_argument("--source", required=True)
    p_enc.add_argument("--target", required=True)
    p_enc.add_argument("--key", default="")
    p_enc.set_defaults(func=cmd_encrypt)

    p_dec = sub.add_parser("decrypt-model", help="Decrypt encrypted model file.")
    p_dec.add_argument("--config_gen", default="configs/gen.yaml")
    p_dec.add_argument("--env_path", default=".env")
    p_dec.add_argument("--source", required=True)
    p_dec.add_argument("--target", required=True)
    p_dec.add_argument("--key", default="")
    p_dec.set_defaults(func=cmd_decrypt)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

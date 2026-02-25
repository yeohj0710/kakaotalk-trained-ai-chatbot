from __future__ import annotations

import argparse

from chatbot.crypto_utils import encrypt_file
from chatbot.security import get_model_key
from chatbot.config import load_gen_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt model file to .enc")
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--key", default="")
    parser.add_argument("--config_gen", default="configs/gen.yaml")
    parser.add_argument("--env_path", default=".env")
    args = parser.parse_args()

    if args.key:
        key = args.key
    else:
        gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
        key = get_model_key(dict(gen_cfg.get("security", {})), env_path=args.env_path)
    encrypt_file(source_path=args.source, target_path=args.target, secret=key)
    print(f"encrypted: {args.target}")


if __name__ == "__main__":
    main()

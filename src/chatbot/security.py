from __future__ import annotations

import getpass
import hashlib
import hmac
import os
import sys
from typing import Any

from .config import load_dotenv


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_expected_secret(security_cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    password_env = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    password_hash_env = str(security_cfg.get("password_hash_env", "CHATBOT_PASSWORD_HASH"))
    expected_plain = os.getenv(password_env)
    expected_hash = os.getenv(password_hash_env)
    return expected_plain, expected_hash


def require_password(
    security_cfg: dict[str, Any] | None,
    password: str | None = None,
    env_path: str = ".env",
) -> None:
    cfg = dict(security_cfg or {})
    if not bool(cfg.get("require_password", True)):
        return

    load_dotenv(env_path, override=False)
    expected_plain, expected_hash = _read_expected_secret(cfg)
    if not expected_plain and not expected_hash:
        raise PermissionError(
            "Inference/bridge access blocked: no password configured. "
            "Set CHATBOT_PASSWORD or CHATBOT_PASSWORD_HASH in .env."
        )

    provided = password or os.getenv("CHATBOT_ACCESS_PASSWORD")
    if not provided and sys.stdin.isatty():
        provided = getpass.getpass("Chatbot password: ").strip()
    if not provided:
        raise PermissionError("Inference/bridge access blocked: missing password.")

    if expected_plain:
        ok = hmac.compare_digest(provided, expected_plain)
    else:
        ok = hmac.compare_digest(sha256_text(provided), str(expected_hash))
    if not ok:
        raise PermissionError("Inference/bridge access blocked: wrong password.")


def get_model_key(
    security_cfg: dict[str, Any] | None,
    env_path: str = ".env",
) -> str:
    cfg = dict(security_cfg or {})
    load_dotenv(env_path, override=False)
    key_env = str(cfg.get("model_key_env", "CHATBOT_MODEL_KEY"))
    key = os.getenv(key_env)
    if not key:
        raise PermissionError(f"Missing model encryption key env var: {key_env}")
    return key

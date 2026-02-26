from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]
ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}$")

DEFAULT_PATHS: ConfigDict = {
    "defaults": {"run_name": "room_v2_context"},
    "raw": {
        "root_glob": "*.txt",
        "root_exclude": ["requirements.txt", "PORTABLE_STATE.txt"],
        "inbox_dir": "data/raw/inbox",
        "organized_dir": "data/raw/organized",
        "manifest_path": "data/raw/manifest.json",
    },
    "processed": {
        "input_glob": "data/raw/inbox/*.txt",
        "output_dir": "data/processed",
        "preprocess": {
            "corpus_mode": "context_windows",
            "context_turns": 16,
            "min_context_turns": 3,
            "sample_stride": 2,
            "session_gap_minutes": 180,
            "merge_same_speaker": True,
            "merge_gap_minutes": 3,
            "max_merged_chars": 400,
            "min_message_chars": 2,
            "min_target_chars": 10,
            "max_message_chars": 400,
            "drop_low_signal": True,
            "response_only_loss": True,
        },
    },
    "checkpoints": {"root_dir": "checkpoints"},
    "artifacts": {
        "dir": "artifacts",
        "latest_model_name": "model_latest.pt",
        "latest_model_enc_name": "model_latest.enc",
        "model_info_name": "model_info.json",
    },
    "publish": {
        "source_preference": [
            "checkpoints/{run_name}/best.pt",
            "checkpoints/{run_name}/latest.pt",
        ]
    },
    "archive": {
        "root_dir": "data/archive",
    },
}

DEFAULT_TRAIN: ConfigDict = {
    "run_name": "room_v2_context",
    "auto_resume": True,
    "resume_path": "",
    "stop_file_name": "STOP",
    "runtime": {"seed": 42, "device": "auto", "dtype": "auto", "compile": False},
    "data": {
        "data_dir": "data/processed",
        "tokenizer_path": "data/processed/tokenizer.json",
        "train_bin": "data/processed/train.bin",
        "val_bin": "data/processed/val.bin",
        "train_loss_mask_bin": "data/processed/train_loss_mask.bin",
        "val_loss_mask_bin": "data/processed/val_loss_mask.bin",
    },
    "model": {
        "block_size": 512,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.1,
        "bias": False,
    },
    "optimization": {
        "batch_size": 16,
        "grad_accum_steps": 2,
        "max_steps": 2147483647,
        "learning_rate": 1.5e-4,
        "min_lr": 1.5e-5,
        "warmup_steps": 2000,
        "lr_schedule": "cosine",
        "lr_decay_steps": 600000,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
    },
    "objective": {
        "loss_mode": "response_only",
        "require_loss_mask": True,
    },
    "logging": {
        "log_interval": 50,
        "eval_interval": 500,
        "eval_iters": 20,
        "save_interval": 500,
        "keep_last_n_snapshots": 8,
        "sample_prompts": ["<|bos|>"],
    },
}

DEFAULT_GEN: ConfigDict = {
    "runtime": {
        "run_name": "room_v2_context",
        "checkpoint": "",
        "checkpoint_preference": [
            "artifacts/model_latest.enc",
            "artifacts/model_latest.pt",
            "checkpoints/{run_name}/best.pt",
            "checkpoints/{run_name}/latest.pt",
        ],
        "device": "auto",
        "dtype": "auto",
    },
    "sampling": {
        "max_new_tokens": 120,
        "temperature": 0.9,
        "top_k": 120,
        "top_p": 0.95,
        "repetition_penalty": 1.02,
    },
    "output": {
        "text_only": True,
        "max_chars": 400,
        "strip_prefix": True,
        "stop_on_next_turn": True,
    },
    "debug": {"return_raw": False},
    "smoke": {
        "prompts": [
            "오늘 다들 뭐함",
            "아까 얘기한거 다시 정리해줘",
            "그럼 결론은 뭐로 가면 될까?",
        ]
    },
    "dialogue": {
        "mode": "anonymous",
        "max_turns": 12,
        "max_context_tokens": 512,
        "user_tag": "<U>",
        "bot_tag": "<B>",
        "user_speaker": "${CHATBOT_DEFAULT_USER:김교수}",
        "bot_speaker": "${CHATBOT_DEFAULT_BOT:구영휴}",
    },
    "chat": {
        "bot_speaker": "${CHATBOT_DEFAULT_BOT:구영휴}",
        "user_speaker": "${CHATBOT_DEFAULT_USER:김교수}",
        "max_history_turns": 50,
    },
    "bridge": {
        "window_title": "카카오톡",
        "chat_xy": "500,320",
        "input_xy": "520,980",
        "poll_sec": 2.0,
        "max_history_turns": 40,
        "dry_run_default": True,
    },
    "security": {
        "require_password": True,
        "password_env": "CHATBOT_PASSWORD",
        "password_hash_env": "CHATBOT_PASSWORD_HASH",
        "model_key_env": "CHATBOT_MODEL_KEY",
    },
}


def project_root() -> Path:
    return Path.cwd()


def _deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged: ConfigDict = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _coerce_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _set_nested(config: ConfigDict, keys: list[str], value: Any) -> None:
    cursor: ConfigDict = config
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _resolve_env_placeholders(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _resolve_env_placeholders(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_env_placeholders(v) for v in node]
    if isinstance(node, str):
        match = ENV_PATTERN.match(node.strip())
        if match is None:
            return node
        env_name = match.group(1)
        fallback = match.group(2)
        env_value = os.getenv(env_name, fallback)
        if env_value is None:
            return ""
        return _coerce_scalar(env_value)
    return node


def load_dotenv(env_path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    path = Path(env_path)
    if not path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("\ufeff")
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def load_yaml(path: str | Path, default: ConfigDict | None = None) -> ConfigDict:
    p = Path(path)
    if not p.exists():
        return dict(default or {})
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if data is None:
        return dict(default or {})
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {p}")
    return data


def apply_env_overrides(
    config: ConfigDict,
    prefix: str = "CHATBOT_CFG__",
    scope: str = "",
) -> ConfigDict:
    updated = dict(config)
    scoped_name = scope.lower().strip()
    known_scopes = {"train", "gen", "paths"}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path_bits = key[len(prefix) :].split("__")
        if not path_bits:
            continue
        normalized = [bit.lower() for bit in path_bits]
        if scoped_name:
            first = normalized[0]
            if first in known_scopes:
                if first != scoped_name:
                    continue
                normalized = normalized[1:]
            elif first != scoped_name and first in known_scopes:
                continue
        if not normalized:
            continue
        _set_nested(updated, normalized, _coerce_scalar(value))
    return updated


def load_paths_config(config_path: str | Path = "configs/paths.yaml", env_path: str | Path = ".env") -> ConfigDict:
    load_dotenv(env_path=env_path, override=False)
    merged = _deep_merge(DEFAULT_PATHS, load_yaml(config_path))
    merged = _resolve_env_placeholders(merged)
    merged = apply_env_overrides(merged, scope="paths")
    return merged


def load_train_config(config_path: str | Path = "configs/train.yaml", env_path: str | Path = ".env") -> ConfigDict:
    load_dotenv(env_path=env_path, override=False)
    merged = _deep_merge(DEFAULT_TRAIN, load_yaml(config_path))
    merged = _resolve_env_placeholders(merged)
    merged = apply_env_overrides(merged, scope="train")
    return merged


def load_gen_config(config_path: str | Path = "configs/gen.yaml", env_path: str | Path = ".env") -> ConfigDict:
    load_dotenv(env_path=env_path, override=False)
    merged = _deep_merge(DEFAULT_GEN, load_yaml(config_path))
    merged = _resolve_env_placeholders(merged)
    merged = apply_env_overrides(merged, scope="gen")
    return merged


def save_yaml(path: str | Path, payload: ConfigDict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")




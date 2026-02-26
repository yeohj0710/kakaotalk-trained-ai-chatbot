from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .config import load_dotenv


ConfigDict = dict[str, Any]
ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}$")


DEFAULT_SFT_CONFIG: ConfigDict = {
    "project": {"run_name": "room_lora_qwen3b", "seed": 42},
    "paths": {
        "raw_glob": "data/raw/inbox/*.txt",
        "output_dir": "data/sft",
        "train_jsonl": "data/sft/train.jsonl",
        "val_jsonl": "data/sft/val.jsonl",
        "preview_json": "data/sft/preview.json",
        "stats_json": "data/sft/stats.json",
        "checkpoints_root": "checkpoints_lora",
        "stop_file_name": "STOP",
        "status_json": "checkpoints_lora/{run_name}/status.json",
    },
    "data": {
        "val_ratio": 0.02,
        "include_system": False,
        "shuffle_before_split": False,
        "session_gap_minutes": 180,
        "context_turns": 18,
        "min_context_turns": 3,
        "sample_stride": 2,
        "merge_same_speaker": True,
        "merge_gap_minutes": 3,
        "max_merged_chars": 420,
        "min_message_chars": 2,
        "min_target_chars": 10,
        "max_message_chars": 420,
        "drop_low_signal": True,
        "mask_urls": True,
        "mask_numbers": False,
        "drop_media_only": True,
        "max_examples_per_split": 0,
    },
    "prompt": {
        "system": (
            "너는 특정 개인의 자아를 주장하는 AI가 아니다.\n"
            "목표는 단톡방 구성원처럼 자연스럽고 상황 맞는 답변 한 개를 말하는 것이다."
        ),
        "task": "아래 대화 흐름을 보고 다음에 이어질 법한 답변을 한 개 작성하라.",
        "response_one_line": True,
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "local_files_only": False,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
    },
    "lora": {
        "r": 64,
        "alpha": 128,
        "dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    },
    "training": {
        "max_seq_len": 1024,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "grad_accum_steps": 16,
        "learning_rate": 2e-4,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_steps": 300000,
        "num_train_epochs": 1,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "save_total_limit": 6,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "dataloader_num_workers": 0,
        "gradient_accumulation_plugin": "auto",
        "early_stopping_patience": 20,
    },
    "generation": {
        "max_new_tokens": 120,
        "do_sample": True,
        "temperature": 0.72,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.08,
        "max_history_turns": 14,
        "min_reply_chars": 8,
        "regen_attempts": 2,
        "one_line": True,
        "max_chars": 240,
    },
    "smoke": {
        "prompts": [
            "오늘 뭐하냐",
            "아까 얘기한거 한줄로 정리해봐",
            "그럼 결론 뭐로 가는게 맞음?",
        ]
    },
    "security": {"require_password": True, "password_env": "CHATBOT_PASSWORD"},
}


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


def load_sft_config(config_path: str | Path = "configs/sft.yaml", env_path: str | Path = ".env") -> ConfigDict:
    load_dotenv(env_path=env_path, override=False)
    path = Path(config_path)
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"SFT config must be a mapping: {path}")
    else:
        payload = {}
    merged = _deep_merge(DEFAULT_SFT_CONFIG, payload)
    merged = _resolve_env_placeholders(merged)
    return merged


def format_with_run_name(value: str, run_name: str) -> str:
    return value.format(run_name=run_name)

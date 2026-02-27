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
    "project": {"run_name": "room_lora_qwen25_3b_base", "seed": 42},
    "paths": {
        "raw_glob": "data/raw/inbox/*.txt",
        "output_dir": "data/sft",
        "train_jsonl": "data/sft/train.jsonl",
        "val_jsonl": "data/sft/val.jsonl",
        "cpt_train_jsonl": "data/sft/cpt_train.jsonl",
        "cpt_val_jsonl": "data/sft/cpt_val.jsonl",
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
        "context_turns": 24,
        "min_context_turns": 3,
        "sample_stride": 1,
        "merge_same_speaker": True,
        "merge_gap_minutes": 2,
        "max_merged_chars": 360,
        "min_message_chars": 2,
        "min_target_chars": 8,
        "max_message_chars": 360,
        "drop_low_signal": True,
        "mask_urls": True,
        "mask_numbers": False,
        "drop_media_only": True,
        "drop_summary_artifacts": True,
        "summary_bullet_min_count": 1,
        "max_examples_per_split": 0,
    },
    "cpt_data": {
        "window_messages": 64,
        "stride_messages": 16,
        "min_messages": 10,
        "min_chars": 120,
        "max_chars": 2200,
        "use_speaker_prefix": True,
    },
    "pipeline": {
        "enabled": True,
        "run_cpt_first": True,
        "bootstrap_sft_from_cpt": True,
        "cpt_run_name_suffix": "_cpt",
        "skip_cpt_if_sft_has_checkpoint": True,
        "require_cpt_complete_before_sft": True,
    },
    "prompt": {
        "system": (
            "너는 카카오톡 단톡방에 참여한 사람처럼 말한다.\n"
            "상담원 같은 과한 공손체와 설명문을 피하고, 실제 채팅처럼 짧고 자연스럽게 답해라.\n"
            "필요하면 구어체나 비속어를 과장 없이 사용할 수 있다."
        ),
        "task": (
            "아래 대화 흐름을 보고 다음에 이어질 법한 답변 1개를 작성하라.\n"
            "화자 라벨 없이 답변 문장만 출력하라."
        ),
        "response_one_line": True,
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-3B",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "local_files_only": False,
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "gradient_checkpointing": False,
    },
    "lora": {
        "r": 32,
        "alpha": 64,
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
    "cpt_training": {
        "max_seq_len": 768,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "grad_accum_steps": 4,
        "learning_rate": 1.5e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.02,
        "lr_scheduler_type": "cosine",
        "max_steps": 30000,
        "num_train_epochs": 1,
        "eval_steps": 8000,
        "save_steps": 500,
        "logging_steps": 20,
        "save_total_limit": 8,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": False,
        "early_stopping_patience": 0,
    },
    "training": {
        "max_seq_len": 768,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "grad_accum_steps": 4,
        "learning_rate": 1.2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "max_steps": 120000,
        "num_train_epochs": 1,
        "eval_steps": 5000,
        "save_steps": 500,
        "logging_steps": 20,
        "save_total_limit": 8,
        "max_grad_norm": 1.0,
        "tf32": True,
        "bf16": True,
        "fp16": False,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": False,
        "early_stopping_patience": 0,
    },
    "generation": {
        "max_new_tokens": 96,
        "do_sample": True,
        "temperature": 0.85,
        "top_p": 0.92,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "avoid_summary_artifacts": True,
        "max_history_turns": 16,
        "min_reply_chars": 8,
        "regen_attempts": 2,
        "one_line": True,
        "max_chars": 220,
        "use_chat_template": False,
    },
    "smoke": {"prompts": ["오늘 뭐함", "아까 말한 거 요약해봐", "그럼 결론 뭐로 갈까"]},
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

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
    "project": {"run_name": "room_lora_qwen25_7b_group_v2", "seed": 42},
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
        "context_turns": 8,
        "min_context_turns": 2,
        "sample_stride": 1,
        "merge_same_speaker": True,
        "merge_gap_minutes": 2,
        "max_merged_chars": 320,
        "min_message_chars": 2,
        "min_target_chars": 8,
        "max_message_chars": 320,
        "drop_low_signal": True,
        "mask_urls": True,
        "mask_numbers": False,
        "drop_media_only": True,
        "drop_summary_artifacts": True,
        "summary_bullet_min_count": 1,
        "drop_mention_messages": True,
        "max_mentions_per_message": 0,
        "strip_mentions_before_filter": False,
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
        "fail_if_sft_bootstrap_missing": True,
        "fail_if_existing_sft_run_is_fresh": True,
    },
    "prompt": {
        "system": (
            "너는 카카오톡 단체방 멤버처럼 자연스럽게 말한다.\n"
            "과한 해설이나 요약체를 피하고, 실제 채팅처럼 짧고 맥락 있는 한 문장으로 답한다."
        ),
        "task": "아래 대화 흐름을 보고 다음에 이어질 답변 1개를 작성하라. 화자 태그 없이 답변 문장만 출력하라.",
        "response_one_line": True,
    },
    "model": {
        "base_model": "Qwen/Qwen2.5-7B",
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
        "max_steps": 12000,
        "num_train_epochs": 1,
        "eval_steps": 1000,
        "save_steps": 1000,
        "logging_steps": 20,
        "save_total_limit": 30,
        "max_grad_norm": 1.0,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": True,
        "early_stopping_patience": 2,
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
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 20,
        "save_total_limit": 40,
        "max_grad_norm": 1.0,
        "tf32": True,
        "bf16": True,
        "fp16": False,
        "dataloader_num_workers": 2,
        "load_best_model_at_end": True,
        "early_stopping_patience": 4,
        "require_init_adapter_on_fresh_start": True,
    },
    "generation": {
        "inference_mode": "group",
        "max_new_tokens": 96,
        "do_sample": True,
        "temperature": 0.85,
        "top_p": 0.92,
        "top_k": 50,
        "repetition_penalty": 1.06,
        "no_repeat_ngram_size": 4,
        "avoid_summary_artifacts": True,
        "avoid_self_echo": True,
        "avoid_repetitive_output": True,
        "max_mentions": 0,
        "max_history_turns": 8,
        "max_bot_history_turns": 2,
        "min_reply_chars": 8,
        "candidate_count": 3,
        "regen_attempts": 2,
        "one_line": True,
        "max_chars": 220,
        "use_chat_template": False,
        "group_min_user_turns_since_last_bot": 4,
        "group_max_bot_turns_in_window": 2,
        "group_block_consecutive_bot": True,
        "group_no_reply_token": "<NO_REPLY>",
        "enable_confidence_fallback": False,
        "confidence_threshold": 0.04,
        "confidence_min_tokens": 6,
        "confidence_fallback_texts": ["...?", "?", "..?'", "???????"],
        "confidence_fallback_target_rate": 0.0,
        "confidence_fallback_rate_window": 200,
    },
    "smoke": {"prompts": ["오늘 분위기 어때", "아까 얘기 이어서 말해봐", "그럼 지금 뭘 하는게 좋을까"]},
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

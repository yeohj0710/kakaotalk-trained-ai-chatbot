from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from .sft_config import format_with_run_name, load_sft_config


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_hf_env_defaults() -> None:
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.replace("checkpoint-", "", 1))
    except ValueError:
        return -1


def is_resumable_checkpoint(ckpt_dir: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not (ckpt_dir / "trainer_state.json").exists():
        missing.append("trainer_state.json")
    if not (ckpt_dir / "adapter_config.json").exists():
        missing.append("adapter_config.json")
    if not ((ckpt_dir / "adapter_model.safetensors").exists() or (ckpt_dir / "adapter_model.bin").exists()):
        missing.append("adapter_model.(safetensors|bin)")
    return (len(missing) == 0, missing)


def find_latest_resumable_checkpoint(run_dir: Path) -> str | None:
    if not run_dir.exists():
        return None
    candidates = sorted((p for p in run_dir.glob("checkpoint-*") if p.is_dir()), key=checkpoint_step, reverse=True)
    for ckpt in candidates:
        ok, missing = is_resumable_checkpoint(ckpt)
        if ok:
            return str(ckpt)
        print(
            json.dumps(
                {
                    "event": "resume_checkpoint_skipped_invalid",
                    "stage": "cpt",
                    "checkpoint": str(ckpt),
                    "missing": missing,
                },
                ensure_ascii=False,
            )
        )
    return None


def retry_hf_load(fn, attempts: int = 4, base_wait_sec: float = 3.0):
    last_error: Exception | None = None
    for idx in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if idx == attempts - 1:
                break
            wait_sec = base_wait_sec * (idx + 1)
            print(json.dumps({"event": "hf_retry", "attempt": idx + 1, "wait_sec": wait_sec}, ensure_ascii=False))
            time.sleep(wait_sec)
    if last_error is not None:
        raise last_error


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def sanitize_line(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00a0", " ").replace("\u200b", "")
    out = out.strip()
    return out


@dataclass
class TokenizeConfig:
    max_seq_len: int


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase, cfg: TokenizeConfig):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id.")
    max_seq_len = max(8, cfg.max_seq_len)

    def _tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        text = sanitize_line(str(row["text"]))
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            ids = [eos_id]
        if len(ids) >= max_seq_len:
            ids = ids[-(max_seq_len - 1) :]
        input_ids = ids + [eos_id]
        labels = list(input_ids)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return _tokenize_row


class LMDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must provide pad_token_id or eos_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + ([self.pad_id] * pad_len))
            labels.append(item["labels"] + ([-100] * pad_len))
            attention_mask.append(item["attention_mask"] + ([0] * pad_len))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class StopFileCallback(TrainerCallback):
    def __init__(self, stop_file: Path) -> None:
        self.stop_file = stop_file
        self.triggered = False

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        if self.stop_file.exists():
            self.triggered = True
            control.should_save = True
            control.should_training_stop = True
            self.stop_file.unlink(missing_ok=True)
        return control


def force_checkpoint_save(trainer: Trainer) -> str:
    try:
        trainer._save_checkpoint(model=trainer.model, trial=None)  # type: ignore[attr-defined]
        last_ckpt = get_last_checkpoint(trainer.args.output_dir)
        return last_ckpt or ""
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"event": "force_checkpoint_failed", "error": str(exc)}, ensure_ascii=False))
        return ""


def load_json_dataset(path: Path, split_name: str) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} dataset not found: {path}")
    ds = load_dataset("json", data_files={split_name: str(path)}, split=split_name)
    if not isinstance(ds, Dataset):
        raise RuntimeError(f"Failed to load dataset split: {split_name}")
    return ds


def resolve_4bit_config(model_cfg: dict[str, Any], dtype: torch.dtype) -> tuple[Any | None, str]:
    use_4bit = bool(model_cfg.get("load_in_4bit", True))
    if not use_4bit or not torch.cuda.is_available():
        return None, "disabled"
    try:
        import bitsandbytes  # type: ignore # noqa: F401
    except Exception:
        return None, "bitsandbytes_pkg_missing"
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        return None, "bitsandbytes_not_available"

    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
        return bnb_cfg, "enabled"
    except Exception:
        return None, "bitsandbytes_init_failed"


def prepare_model_and_tokenizer(cfg: dict[str, Any], init_adapter_path: str = "") -> tuple[Any, Any, dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}))
    lora_cfg = dict(cfg.get("lora", {}))
    base_model = str(model_cfg.get("base_model", "Qwen/Qwen2.5-3B")).strip()
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    use_fast_tokenizer = bool(model_cfg.get("use_fast_tokenizer", True))
    dtype = resolve_torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
    attn_impl = str(model_cfg.get("attn_implementation", "sdpa")).strip()

    tokenizer = retry_hf_load(
        lambda: AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config, quant_mode = resolve_4bit_config(model_cfg=model_cfg, dtype=dtype)
    if bool(model_cfg.get("load_in_4bit", True)) and quant_mode != "enabled":
        print(
            json.dumps(
                {
                    "event": "quantization_warning",
                    "requested_4bit": True,
                    "quantization": quant_mode,
                    "hint": "Install bitsandbytes for 4bit or keep full precision fallback.",
                },
                ensure_ascii=False,
            )
        )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    model = retry_hf_load(lambda: AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs))
    if quant_config is None and torch.cuda.is_available():
        model.to("cuda")

    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()  # type: ignore[operator]

    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    init_adapter = init_adapter_path.strip()
    if init_adapter:
        init_path = Path(init_adapter)
        if not init_path.exists():
            raise FileNotFoundError(f"CPT init adapter path not found: {init_adapter}")
        model = PeftModel.from_pretrained(model, str(init_path), is_trainable=True)
        init_mode = "from_adapter"
    else:
        task_type = str(lora_cfg.get("task_type", "CAUSAL_LM")).strip().upper()
        peft_task_type = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.CAUSAL_LM
        peft_cfg = LoraConfig(
            task_type=peft_task_type,
            r=int(lora_cfg.get("r", 32)),
            lora_alpha=int(lora_cfg.get("alpha", 64)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=list(lora_cfg.get("target_modules", [])),
        )
        model = get_peft_model(model, peft_cfg)
        init_mode = "fresh_lora"

    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    meta = {
        "quantization": quant_mode,
        "dtype": str(dtype).replace("torch.", ""),
        "base_model": base_model,
        "init_mode": init_mode,
        "init_adapter_path": init_adapter,
    }
    return model, tokenizer, meta


def copy_checkpoint_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 CPT (continued pretraining) for LoRA adapter.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--init_adapter", default="")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    project_cfg = dict(cfg.get("project", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    train_cfg = dict(cfg.get("cpt_training", {}))

    run_name = (args.run_name or str(project_cfg.get("run_name", "room_lora_qwen25_3b_base"))).strip()
    seed = int(project_cfg.get("seed", 42))
    ensure_hf_env_defaults()
    set_seed(seed)

    checkpoints_root = Path(str(paths_cfg.get("checkpoints_root", "checkpoints_lora")))
    run_dir = checkpoints_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / str(paths_cfg.get("stop_file_name", "STOP"))

    train_jsonl = Path(format_with_run_name(str(paths_cfg.get("cpt_train_jsonl", "data/sft/cpt_train.jsonl")), run_name))
    val_jsonl = Path(format_with_run_name(str(paths_cfg.get("cpt_val_jsonl", "data/sft/cpt_val.jsonl")), run_name))
    status_json = Path(format_with_run_name(str(paths_cfg.get("status_json", "checkpoints_lora/{run_name}/status.json")), run_name))

    train_ds = load_json_dataset(train_jsonl, "train")
    val_ds = load_json_dataset(val_jsonl, "val")
    if len(train_ds) < 100:
        raise RuntimeError("CPT train dataset is too small. Run preprocess with enough examples.")
    if len(val_ds) < 20:
        raise RuntimeError("CPT validation dataset is too small. Increase data or lower filters.")

    model, tokenizer, model_meta = prepare_model_and_tokenizer(cfg, init_adapter_path=args.init_adapter)
    tokenize_fn = build_tokenize_fn(
        tokenizer=tokenizer,
        cfg=TokenizeConfig(max_seq_len=int(train_cfg.get("max_seq_len", 768))),
    )
    train_tok = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, desc="Tokenizing CPT train")
    val_tok = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names, desc="Tokenizing CPT val")
    collator = LMDataCollator(tokenizer=tokenizer)

    bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = bool(train_cfg.get("fp16", False)) and not bf16 and torch.cuda.is_available()
    eval_steps = int(train_cfg.get("eval_steps", 5000))
    save_steps = int(train_cfg.get("save_steps", 500))
    load_best_model_at_end = bool(train_cfg.get("load_best_model_at_end", False))
    if load_best_model_at_end and eval_steps > 0 and save_steps % eval_steps != 0:
        raise ValueError(
            "Invalid CPT config: when load_best_model_at_end=true, save_steps must be a multiple of eval_steps. "
            "Set load_best_model_at_end=false to allow frequent saves with sparse eval."
        )
    print(
        json.dumps(
            {
                "event": "train_schedule",
                "stage": "cpt",
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "load_best_model_at_end": load_best_model_at_end,
            },
            ensure_ascii=False,
        )
    )

    args_train = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum_steps", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.02)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        max_steps=int(train_cfg.get("max_steps", 30000)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=int(train_cfg.get("save_total_limit", 8)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        bf16=bf16,
        fp16=fp16,
        tf32=bool(train_cfg.get("tf32", True)),
        gradient_checkpointing=bool(cfg.get("model", {}).get("gradient_checkpointing", False)),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_loss" if load_best_model_at_end else None,
        greater_is_better=False if load_best_model_at_end else None,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    stop_callback = StopFileCallback(stop_file=stop_file)
    callbacks: list[TrainerCallback] = [stop_callback]
    patience = int(train_cfg.get("early_stopping_patience", 0))
    if patience > 0 and load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
    elif patience > 0 and not load_best_model_at_end:
        print(
            json.dumps(
                {
                    "event": "early_stopping_disabled",
                    "stage": "cpt",
                    "reason": "load_best_model_at_end=false",
                },
                ensure_ascii=False,
            )
        )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    last_ckpt = find_latest_resumable_checkpoint(run_dir)
    resume_ckpt = last_ckpt if last_ckpt else None
    if resume_ckpt and args.init_adapter:
        print(json.dumps({"event": "init_adapter_ignored", "stage": "cpt", "reason": "resume_checkpoint_exists"}, ensure_ascii=False))

    started_at = now_iso()
    stopped = False
    stop_reason = ""
    forced_checkpoint = ""
    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except KeyboardInterrupt:
        stopped = True
        stop_reason = "keyboard_interrupt"
        forced_checkpoint = force_checkpoint_save(trainer)
    finally:
        trainer.save_state()

    if stop_callback.triggered:
        stopped = True
        if not stop_reason:
            stop_reason = "stop_file"
        if not forced_checkpoint:
            forced_checkpoint = force_checkpoint_save(trainer)

    latest_adapter_dir = run_dir / "adapter_latest"
    trainer.model.save_pretrained(latest_adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(latest_adapter_dir)

    best_ckpt = trainer.state.best_model_checkpoint
    best_adapter_dir = run_dir / "adapter_best"
    if load_best_model_at_end and best_ckpt and Path(best_ckpt).exists():
        copy_checkpoint_dir(Path(best_ckpt), best_adapter_dir)
        tokenizer.save_pretrained(best_adapter_dir)
        best_mode = "best_checkpoint"
    else:
        copy_checkpoint_dir(latest_adapter_dir, best_adapter_dir)
        tokenizer.save_pretrained(best_adapter_dir)
        best_mode = "latest_alias" if not load_best_model_at_end else "best_checkpoint_missing_fallback_latest"

    max_steps = int(train_cfg.get("max_steps", 30000))
    completed = bool(trainer.state.global_step >= max_steps and not stopped)
    finished_at = now_iso()
    status = {
        "event": "cpt_train_done",
        "stage": "cpt",
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "resumed_from": resume_ckpt or "",
        "best_checkpoint": best_ckpt or "",
        "best_mode": best_mode,
        "best_adapter_dir": str(best_adapter_dir.as_posix()),
        "latest_adapter_dir": str(latest_adapter_dir.as_posix()),
        "global_step": int(trainer.state.global_step),
        "target_max_steps": max_steps,
        "completed": completed,
        "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "stopped": stopped,
        "stop_reason": stop_reason,
        "forced_checkpoint": forced_checkpoint,
        "model_meta": model_meta,
    }
    status_json.parent.mkdir(parents=True, exist_ok=True)
    status_json.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()

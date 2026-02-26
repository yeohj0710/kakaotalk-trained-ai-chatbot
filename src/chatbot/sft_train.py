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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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


def one_line(text: str) -> str:
    out = sanitize_line(text)
    out = " ".join(part for part in out.splitlines() if part.strip())
    out = " ".join(out.split())
    return out.strip()


@dataclass
class TokenizeConfig:
    max_seq_len: int
    one_line_response: bool


def build_tokenize_fn(tokenizer: PreTrainedTokenizerBase, cfg: TokenizeConfig):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must provide eos_token_id.")

    def _tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        prompt = sanitize_line(str(row["prompt"]))
        response = str(row["response"])
        if cfg.one_line_response:
            response = one_line(response)
        else:
            response = sanitize_line(response)

        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = tokenizer(response, add_special_tokens=False).input_ids + [eos_id]

        # Preserve response span first, trim prompt from the left if needed.
        if len(response_ids) >= cfg.max_seq_len:
            response_ids = response_ids[-cfg.max_seq_len :]
            prompt_ids = []
        else:
            max_prompt = cfg.max_seq_len - len(response_ids)
            if len(prompt_ids) > max_prompt:
                prompt_ids = prompt_ids[-max_prompt:]

        input_ids = prompt_ids + response_ids
        labels = ([-100] * len(prompt_ids)) + response_ids
        attention_mask = [1] * len(input_ids)

        if not any(label != -100 for label in labels):
            # Hard fallback: at least learn on the final token.
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return _tokenize_row


class SFTDataCollator:
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


def prepare_model_and_tokenizer(cfg: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    model_cfg = dict(cfg.get("model", {}))
    lora_cfg = dict(cfg.get("lora", {}))
    base_model = str(model_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")).strip()
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
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "dtype": dtype,
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
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  # type: ignore[operator]

    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    task_type = str(lora_cfg.get("task_type", "CAUSAL_LM")).strip().upper()
    peft_task_type = TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType.CAUSAL_LM
    peft_cfg = LoraConfig(
        task_type=peft_task_type,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 128)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=list(lora_cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    meta = {"quantization": quant_mode, "dtype": str(dtype).replace("torch.", ""), "base_model": base_model}
    return model, tokenizer, meta


def copy_checkpoint_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LoRA adapter on SFT chat dataset with auto-resume.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    project_cfg = dict(cfg.get("project", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    train_cfg = dict(cfg.get("training", {}))
    prompt_cfg = dict(cfg.get("prompt", {}))

    run_name = str(project_cfg.get("run_name", "room_lora_qwen3b")).strip() or "room_lora_qwen3b"
    seed = int(project_cfg.get("seed", 42))
    ensure_hf_env_defaults()
    set_seed(seed)

    checkpoints_root = Path(str(paths_cfg.get("checkpoints_root", "checkpoints_lora")))
    run_dir = checkpoints_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / str(paths_cfg.get("stop_file_name", "STOP"))

    train_jsonl = Path(format_with_run_name(str(paths_cfg.get("train_jsonl", "data/sft/train.jsonl")), run_name))
    val_jsonl = Path(format_with_run_name(str(paths_cfg.get("val_jsonl", "data/sft/val.jsonl")), run_name))
    status_json = Path(format_with_run_name(str(paths_cfg.get("status_json", "checkpoints_lora/{run_name}/status.json")), run_name))

    train_ds = load_json_dataset(train_jsonl, "train")
    val_ds = load_json_dataset(val_jsonl, "val")
    if len(train_ds) < 100:
        raise RuntimeError("Train dataset is too small. Run preprocess with enough examples.")
    if len(val_ds) < 20:
        raise RuntimeError("Validation dataset is too small. Increase data or lower filters.")

    model, tokenizer, model_meta = prepare_model_and_tokenizer(cfg)
    tokenize_fn = build_tokenize_fn(
        tokenizer=tokenizer,
        cfg=TokenizeConfig(
            max_seq_len=int(train_cfg.get("max_seq_len", 1024)),
            one_line_response=bool(prompt_cfg.get("response_one_line", True)),
        ),
    )
    train_tok = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names, desc="Tokenizing train")
    val_tok = val_ds.map(tokenize_fn, remove_columns=val_ds.column_names, desc="Tokenizing val")
    collator = SFTDataCollator(tokenizer=tokenizer)

    bf16 = bool(train_cfg.get("bf16", True)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = bool(train_cfg.get("fp16", False)) and not bf16 and torch.cuda.is_available()

    args_train = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        max_steps=int(train_cfg.get("max_steps", 300000)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        eval_steps=int(train_cfg.get("eval_steps", 500)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_total_limit=int(train_cfg.get("save_total_limit", 6)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(cfg.get("model", {}).get("gradient_checkpointing", True)),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
    )

    stop_callback = StopFileCallback(stop_file=stop_file)
    callbacks: list[TrainerCallback] = [stop_callback]
    patience = int(train_cfg.get("early_stopping_patience", 0))
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    last_ckpt = get_last_checkpoint(str(run_dir)) if run_dir.exists() else None
    resume_ckpt = last_ckpt if last_ckpt else None

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
    if best_ckpt and Path(best_ckpt).exists():
        copy_checkpoint_dir(Path(best_ckpt), best_adapter_dir)
        tokenizer.save_pretrained(best_adapter_dir)
    elif not best_adapter_dir.exists():
        copy_checkpoint_dir(latest_adapter_dir, best_adapter_dir)

    finished_at = now_iso()
    status = {
        "event": "sft_train_done",
        "run_name": run_name,
        "started_at": started_at,
        "finished_at": finished_at,
        "resumed_from": resume_ckpt or "",
        "best_checkpoint": best_ckpt or "",
        "best_adapter_dir": str(best_adapter_dir.as_posix()),
        "latest_adapter_dir": str(latest_adapter_dir.as_posix()),
        "global_step": int(trainer.state.global_step),
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

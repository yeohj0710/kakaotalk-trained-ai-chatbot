from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_paths_config, load_train_config, save_json, save_yaml
from .console import to_console_safe
from .model import GPT, GPTConfig
from .text_postprocess import OutputOptions, postprocess_generated_text
from .tokenizer import ByteSpecialTokenizer


IMMUTABLE_MODEL_KEYS = ("block_size", "n_layer", "n_head", "n_embd", "bias", "dropout", "vocab_size")
MUTABLE_PREFIXES = ("optimization.", "logging.", "runtime.", "auto_resume", "resume_path", "stop_file_name")


def create_grad_scaler(enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    if not enabled:
        try:
            return torch.amp.GradScaler("cuda", enabled=False)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(device: str, dtype: str) -> tuple[str, torch.dtype | None]:
    if device == "cpu":
        return "fp32", None
    if dtype == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    if dtype == "bf16":
        return "bf16", torch.bfloat16
    if dtype == "fp16":
        return "fp16", torch.float16
    if dtype == "fp32":
        return "fp32", None
    raise ValueError(f"Unsupported dtype: {dtype}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def flatten_config(node: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_config(value, new_prefix))
    else:
        out[prefix] = node
    return out


def diff_configs(previous: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    prev_flat = flatten_config(previous)
    curr_flat = flatten_config(current)
    diffs: list[dict[str, Any]] = []
    for key in sorted(set(prev_flat.keys()) | set(curr_flat.keys())):
        prev_val = prev_flat.get(key, "<missing>")
        curr_val = curr_flat.get(key, "<missing>")
        if prev_val != curr_val:
            diffs.append({"key": key, "previous": prev_val, "current": curr_val})
    return diffs


def is_mutable_key(key: str) -> bool:
    return key.startswith(MUTABLE_PREFIXES)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
    loss_mask_data: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    max_index = len(data) - block_size - 1
    if max_index <= 0:
        raise ValueError("Dataset is too small for the configured block_size.")
    starts: list[int] = []
    if loss_mask_data is not None:
        attempts = 0
        while len(starts) < batch_size and attempts < 12:
            attempts += 1
            need = (batch_size - len(starts)) * 3
            candidates = torch.randint(0, max_index, (need,))
            for candidate in candidates.tolist():
                start = int(candidate)
                if np.any(loss_mask_data[start + 1 : start + block_size + 1]):
                    starts.append(start)
                    if len(starts) == batch_size:
                        break
    if len(starts) < batch_size:
        fallback = torch.randint(0, max_index, (batch_size - len(starts),)).tolist()
        starts.extend([int(x) for x in fallback])

    x = torch.stack([torch.from_numpy(data[idx : idx + block_size].astype(np.int64)) for idx in starts])
    y = torch.stack(
        [torch.from_numpy(data[idx + 1 : idx + block_size + 1].astype(np.int64)) for idx in starts]
    )
    loss_mask: torch.Tensor | None = None
    if loss_mask_data is not None:
        loss_mask = torch.stack(
            [
                torch.from_numpy(loss_mask_data[idx + 1 : idx + block_size + 1].astype(np.float32))
                for idx in starts
            ]
        )
        loss_mask = loss_mask.to(device)
    return x.to(device), y.to(device), loss_mask


def compute_masked_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    if loss_mask is None:
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    per_token = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction="none",
    ).view_as(targets)
    mask = loss_mask.to(dtype=per_token.dtype)
    denom = mask.sum()
    if denom.item() <= 0:
        return per_token.mean()
    return (per_token * mask).sum() / denom


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    train_loss_mask: np.ndarray | None,
    val_loss_mask: np.ndarray | None,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    autocast_dtype: torch.dtype | None,
    use_response_only_loss: bool,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data, mask_data in (
        ("train", train_data, train_loss_mask),
        ("val", val_data, val_loss_mask),
    ):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y, loss_mask = get_batch(
                data,
                batch_size=batch_size,
                block_size=block_size,
                device=device,
                loss_mask_data=mask_data if use_response_only_loss else None,
            )
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else contextlib.nullcontext()
            )
            with context:
                if use_response_only_loss:
                    logits, _ = model(x)
                    loss = compute_masked_loss(logits=logits, targets=y, loss_mask=loss_mask)
                else:
                    _, loss = model(x, y)
            losses[k] = float(loss.item())
        out[split] = float(losses.mean())
    model.train()
    return out


def get_lr(
    step: int,
    decay_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    if step >= decay_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer: ByteSpecialTokenizer,
    tokenizer_path: Path,
    tokenizer_sha256: str,
    model_config: GPTConfig,
    train_config_snapshot: dict[str, Any],
    paths_config_snapshot: dict[str, Any],
    step: int,
    best_val_loss: float,
) -> None:
    tokenizer_state = tokenizer.to_state()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config_snapshot": train_config_snapshot,
        "paths_config_snapshot": paths_config_snapshot,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_state": {
            "version": tokenizer_state.version,
            "special_tokens": tokenizer_state.special_tokens,
            "speaker_to_token": tokenizer_state.speaker_to_token,
        },
        "step": step,
        "best_val_loss": best_val_loss,
        "saved_at": time.time(),
    }
    torch.save(payload, path)


def prune_snapshots(snapshot_dir: Path, keep_last_n: int) -> None:
    if keep_last_n <= 0:
        return
    snapshots = sorted(snapshot_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    to_delete = snapshots[:-keep_last_n]
    for path in to_delete:
        path.unlink(missing_ok=True)


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_eval_samples(
    model: GPT,
    tokenizer: ByteSpecialTokenizer,
    prompts: list[str],
    step: int,
    device: str,
    autocast_dtype: torch.dtype | None,
    output_dir: Path,
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, str]] = []
    output_options = OutputOptions(
        text_only=True,
        max_chars=0,
        strip_prefix=True,
        stop_on_next_turn=True,
        one_line=True,
    )
    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt)
            xprompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else contextlib.nullcontext()
            )
            with context:
                y = model.generate(
                    idx=xprompt,
                    max_new_tokens=120,
                    temperature=0.9,
                    top_k=120,
                    top_p=0.95,
                    eos_token_id=tokenizer.eos_id,
                )
            sample_raw = tokenizer.decode(y[0].tolist())
            sample = postprocess_generated_text(
                raw_text=sample_raw,
                tokenizer=tokenizer,
                bot_speaker=None,
                options=output_options,
            )
            generated_rows.append({"prompt": prompt, "sample": sample, "sample_raw": sample_raw})

    sample_path = output_dir / f"sample_step_{step:06d}.txt"
    with sample_path.open("w", encoding="utf-8") as handle:
        for row in generated_rows:
            handle.write(f"[PROMPT]\n{row['prompt']}\n\n")
            handle.write(f"[SAMPLE]\n{row['sample']}\n")
            handle.write(f"\n[RAW]\n{row['sample_raw']}\n")
            handle.write("\n" + ("=" * 60) + "\n\n")
    return generated_rows


def choose_resume_checkpoint(
    run_dir: Path,
    train_cfg: dict[str, Any],
    cli_resume: str,
    disable_auto_resume: bool,
) -> Path | None:
    if cli_resume:
        resume_path = Path(cli_resume)
        return resume_path if resume_path.exists() else None

    resume_from_config = str(train_cfg.get("resume_path", "")).strip()
    if resume_from_config:
        resume_path = Path(resume_from_config)
        return resume_path if resume_path.exists() else None

    auto_resume = bool(train_cfg.get("auto_resume", True)) and not disable_auto_resume
    latest = run_dir / "latest.pt"
    if auto_resume and latest.exists():
        return latest
    return None


def validate_resume_compatibility(
    checkpoint: dict[str, Any],
    model_config: GPTConfig,
    tokenizer_sha256: str,
) -> None:
    previous_model_cfg = dict(checkpoint.get("model_config", {}))
    current_model_cfg = asdict(model_config)
    for key in IMMUTABLE_MODEL_KEYS:
        if previous_model_cfg.get(key) != current_model_cfg.get(key):
            raise RuntimeError(
                f"Incompatible resume: model config '{key}' changed "
                f"({previous_model_cfg.get(key)} -> {current_model_cfg.get(key)}). "
                "Start a new run directory for this configuration."
            )

    previous_tokenizer_sha = checkpoint.get("tokenizer_sha256")
    if previous_tokenizer_sha and previous_tokenizer_sha != tokenizer_sha256:
        raise RuntimeError(
            "Incompatible resume: tokenizer content changed. "
            "Start a new run after preprocessing/tokenizer change."
        )


def run_training(
    train_cfg_path: str,
    paths_cfg_path: str,
    env_path: str,
    cli_resume: str,
    cli_run_name: str,
    disable_auto_resume: bool,
    cli_max_steps: int | None,
) -> dict[str, Any]:
    train_cfg = load_train_config(config_path=train_cfg_path, env_path=env_path)
    paths_cfg = load_paths_config(config_path=paths_cfg_path, env_path=env_path)

    run_name = (
        cli_run_name
        or str(train_cfg.get("run_name", "")).strip()
        or str(paths_cfg.get("defaults", {}).get("run_name", "room_v1"))
    )
    train_cfg["run_name"] = run_name
    if cli_max_steps is not None:
        train_cfg.setdefault("optimization", {})["max_steps"] = int(cli_max_steps)

    checkpoints_root = Path(str(paths_cfg.get("checkpoints", {}).get("root_dir", "checkpoints")))
    run_dir = checkpoints_root / run_name
    logs_dir = run_dir / "logs"
    snapshots_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    config_used = {"train": train_cfg, "paths": paths_cfg, "saved_at": datetime_now()}
    save_yaml(run_dir / "config_used.yaml", config_used)

    data_cfg = dict(train_cfg.get("data", {}))
    runtime_cfg = dict(train_cfg.get("runtime", {}))
    model_cfg = dict(train_cfg.get("model", {}))
    optim_cfg = dict(train_cfg.get("optimization", {}))
    logging_cfg = dict(train_cfg.get("logging", {}))
    objective_cfg = dict(train_cfg.get("objective", {}))

    data_dir = Path(str(data_cfg.get("data_dir", "data/processed")))
    tokenizer_path = Path(str(data_cfg.get("tokenizer_path", data_dir / "tokenizer.json")))
    train_bin = Path(str(data_cfg.get("train_bin", data_dir / "train.bin")))
    val_bin = Path(str(data_cfg.get("val_bin", data_dir / "val.bin")))
    train_loss_mask_bin = Path(str(data_cfg.get("train_loss_mask_bin", data_dir / "train_loss_mask.bin")))
    val_loss_mask_bin = Path(str(data_cfg.get("val_loss_mask_bin", data_dir / "val_loss_mask.bin")))
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not train_bin.exists() or not val_bin.exists():
        raise FileNotFoundError(f"Missing train/val data: {train_bin} / {val_bin}")

    tokenizer = ByteSpecialTokenizer.load(tokenizer_path)
    tokenizer_sha = sha256_file(tokenizer_path)

    train_data = np.fromfile(train_bin, dtype=np.uint16)
    val_data = np.fromfile(val_bin, dtype=np.uint16)
    if len(train_data) < 5000:
        raise RuntimeError("Train dataset is too small. Check preprocessing output.")

    loss_mode = str(objective_cfg.get("loss_mode", "response_only")).strip().lower()
    use_response_only_loss = loss_mode == "response_only"
    if loss_mode not in {"response_only", "full_sequence"}:
        raise ValueError("objective.loss_mode must be one of: response_only, full_sequence")
    require_loss_mask = bool(objective_cfg.get("require_loss_mask", use_response_only_loss))

    train_loss_mask: np.ndarray | None = None
    val_loss_mask: np.ndarray | None = None
    if train_loss_mask_bin.exists() and val_loss_mask_bin.exists():
        train_loss_mask = np.fromfile(train_loss_mask_bin, dtype=np.uint8)
        val_loss_mask = np.fromfile(val_loss_mask_bin, dtype=np.uint8)
        if len(train_loss_mask) != len(train_data) or len(val_loss_mask) != len(val_data):
            raise RuntimeError(
                "Loss mask length mismatch. Re-run preprocess to regenerate aligned mask files."
            )
    elif require_loss_mask:
        raise FileNotFoundError(
            "Missing loss mask files (train_loss_mask.bin/val_loss_mask.bin). "
            "Run preprocess before training."
        )
    else:
        use_response_only_loss = False

    seed = int(runtime_cfg.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = resolve_device(str(runtime_cfg.get("device", "auto")))
    dtype_name, autocast_dtype = resolve_dtype(device=device, dtype=str(runtime_cfg.get("dtype", "auto")))
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    gpt_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(model_cfg.get("block_size", 256)),
        n_layer=int(model_cfg.get("n_layer", 6)),
        n_head=int(model_cfg.get("n_head", 6)),
        n_embd=int(model_cfg.get("n_embd", 384)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        bias=bool(model_cfg.get("bias", False)),
    )
    model = GPT(gpt_cfg)
    model.to(device)

    if bool(runtime_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = model.configure_optimizers(
        weight_decay=float(optim_cfg.get("weight_decay", 0.1)),
        learning_rate=float(optim_cfg.get("learning_rate", 3e-4)),
        betas=(float(optim_cfg.get("beta1", 0.9)), float(optim_cfg.get("beta2", 0.95))),
        device_type=device_type,
    )

    scaler = create_grad_scaler(enabled=(device_type == "cuda" and dtype_name == "fp16"))
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=autocast_dtype))
        if autocast_dtype is not None
        else contextlib.nullcontext
    )

    resume_ckpt_path = choose_resume_checkpoint(
        run_dir=run_dir,
        train_cfg=train_cfg,
        cli_resume=cli_resume,
        disable_auto_resume=disable_auto_resume,
    )
    start_step = 0
    best_val_loss = float("inf")
    resume_info: dict[str, Any] = {"resumed": False}

    if resume_ckpt_path is not None and resume_ckpt_path.exists():
        checkpoint = torch.load(resume_ckpt_path, map_location="cpu")
        validate_resume_compatibility(checkpoint=checkpoint, model_config=gpt_cfg, tokenizer_sha256=tokenizer_sha)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Resume-compatible hyperparameters are overridden by current config.
        for idx, group in enumerate(optimizer.param_groups):
            group["betas"] = (float(optim_cfg.get("beta1", 0.9)), float(optim_cfg.get("beta2", 0.95)))
            if idx == 0:
                group["weight_decay"] = float(optim_cfg.get("weight_decay", 0.1))
            else:
                group["weight_decay"] = 0.0

        prev_cfg = checkpoint.get("train_config_snapshot")
        override_log: list[dict[str, Any]] = []
        if isinstance(prev_cfg, dict):
            for diff in diff_configs(prev_cfg, train_cfg):
                diff["mutable"] = is_mutable_key(diff["key"])
                override_log.append(diff)
                if not diff["mutable"] and diff["key"].startswith("model."):
                    raise RuntimeError(
                        f"Incompatible resume: immutable field changed: {diff['key']} "
                        f"({diff['previous']} -> {diff['current']})"
                    )
        if override_log:
            override_payload = {"time": datetime_now(), "resume_from": str(resume_ckpt_path), "changes": override_log}
            save_json(logs_dir / "override_log.json", override_payload)
            write_jsonl(logs_dir / "train_log.jsonl", {"event": "resume_override", **override_payload})

        start_step = int(checkpoint.get("step", 0)) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        resume_info = {
            "resumed": True,
            "resume_checkpoint": str(resume_ckpt_path),
            "resume_step": start_step,
            "best_val_loss": best_val_loss,
        }
        print(json.dumps({"event": "resume", **resume_info}, ensure_ascii=False))

    train_log_path = logs_dir / "train_log.jsonl"
    if start_step == 0 and train_log_path.exists():
        train_log_path.unlink()

    sample_prompts = [str(x) for x in logging_cfg.get("sample_prompts", ["<|bos|>"])]
    log_interval = int(logging_cfg.get("log_interval", 50))
    eval_interval = int(logging_cfg.get("eval_interval", 500))
    eval_iters = int(logging_cfg.get("eval_iters", 20))
    save_interval = int(logging_cfg.get("save_interval", 500))
    keep_last_n = int(logging_cfg.get("keep_last_n_snapshots", 8))

    batch_size = int(optim_cfg.get("batch_size", 32))
    grad_accum_steps = int(optim_cfg.get("grad_accum_steps", 1))
    max_steps = int(optim_cfg.get("max_steps", 10000))
    warmup_steps = int(optim_cfg.get("warmup_steps", 300))
    learning_rate = float(optim_cfg.get("learning_rate", 3e-4))
    min_lr = float(optim_cfg.get("min_lr", 3e-5))
    lr_schedule = str(optim_cfg.get("lr_schedule", "cosine")).strip().lower()
    lr_decay_steps = max(warmup_steps + 1, int(optim_cfg.get("lr_decay_steps", max_steps)))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))

    stop_file = run_dir / str(train_cfg.get("stop_file_name", "STOP"))
    stop_requested = False
    stop_reason = ""
    t_start = time.time()
    final_step = start_step - 1
    objective_info = {
        "event": "objective",
        "loss_mode": "response_only" if use_response_only_loss else "full_sequence",
        "lr_schedule": lr_schedule,
        "lr_decay_steps": lr_decay_steps,
    }
    print(json.dumps(objective_info, ensure_ascii=False))
    write_jsonl(train_log_path, objective_info)

    try:
        model.train()
        for step in range(start_step, max_steps):
            final_step = step

            if stop_file.exists():
                stop_requested = True
                stop_reason = "stop_file"
                stop_file.unlink(missing_ok=True)

            if lr_schedule == "constant":
                lr = learning_rate
            else:
                lr = get_lr(
                    step=step,
                    decay_steps=lr_decay_steps,
                    warmup_steps=warmup_steps,
                    learning_rate=learning_rate,
                    min_lr=min_lr,
                )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for _ in range(grad_accum_steps):
                xb, yb, loss_mask = get_batch(
                    train_data,
                    batch_size=batch_size,
                    block_size=gpt_cfg.block_size,
                    device=device,
                    loss_mask_data=train_loss_mask if use_response_only_loss else None,
                )
                with autocast_context():
                    if use_response_only_loss:
                        logits, _ = model(xb)
                        loss = compute_masked_loss(logits=logits, targets=yb, loss_mask=loss_mask)
                    else:
                        _, loss = model(xb, yb)
                    scaled_loss = loss / grad_accum_steps
                loss_value += float(loss.item())
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if grad_clip > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if step % log_interval == 0:
                log_payload = {
                    "event": "train",
                    "step": step,
                    "loss": loss_value / grad_accum_steps,
                    "lr": lr,
                    "elapsed_sec": round(time.time() - t_start, 2),
                }
                print(json.dumps(log_payload, ensure_ascii=False))
                write_jsonl(train_log_path, log_payload)

            should_eval = (
                (step % eval_interval == 0)
                or (step % save_interval == 0)
                or (step == max_steps - 1)
                or stop_requested
            )
            if should_eval:
                losses = estimate_loss(
                    model=model,
                    train_data=train_data,
                    val_data=val_data,
                    train_loss_mask=train_loss_mask,
                    val_loss_mask=val_loss_mask,
                    eval_iters=eval_iters,
                    batch_size=batch_size,
                    block_size=gpt_cfg.block_size,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    use_response_only_loss=use_response_only_loss,
                )
                eval_payload = {
                    "event": "eval",
                    "step": step,
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "best_val_loss": min(best_val_loss, losses["val"]),
                }
                print(json.dumps(eval_payload, ensure_ascii=False))
                write_jsonl(train_log_path, eval_payload)

                sample_rows = save_eval_samples(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=sample_prompts,
                    step=step,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    output_dir=logs_dir,
                )
                if sample_rows:
                    print("=== SAMPLE ===")
                    print(to_console_safe(sample_rows[0]["sample"][:500]))
                    print("==============")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(
                        path=run_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        tokenizer=tokenizer,
                        tokenizer_path=tokenizer_path,
                        tokenizer_sha256=tokenizer_sha,
                        model_config=gpt_cfg,
                        train_config_snapshot=train_cfg,
                        paths_config_snapshot=paths_cfg,
                        step=step,
                        best_val_loss=best_val_loss,
                    )

            should_save = (step % save_interval == 0) or (step == max_steps - 1) or stop_requested
            if should_save:
                save_checkpoint(
                    path=run_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    tokenizer_path=tokenizer_path,
                    tokenizer_sha256=tokenizer_sha,
                    model_config=gpt_cfg,
                    train_config_snapshot=train_cfg,
                    paths_config_snapshot=paths_cfg,
                    step=step,
                    best_val_loss=best_val_loss,
                )
                snapshot_path = snapshots_dir / f"step_{step:06d}.pt"
                save_checkpoint(
                    path=snapshot_path,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=tokenizer,
                    tokenizer_path=tokenizer_path,
                    tokenizer_sha256=tokenizer_sha,
                    model_config=gpt_cfg,
                    train_config_snapshot=train_cfg,
                    paths_config_snapshot=paths_cfg,
                    step=step,
                    best_val_loss=best_val_loss,
                )
                prune_snapshots(snapshot_dir=snapshots_dir, keep_last_n=keep_last_n)

            if stop_requested:
                break

    except KeyboardInterrupt:
        stop_requested = True
        stop_reason = "keyboard_interrupt"
    finally:
        if stop_requested:
            save_checkpoint(
                path=run_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                tokenizer=tokenizer,
                tokenizer_path=tokenizer_path,
                tokenizer_sha256=tokenizer_sha,
                model_config=gpt_cfg,
                train_config_snapshot=train_cfg,
                paths_config_snapshot=paths_cfg,
                step=max(final_step, 0),
                best_val_loss=best_val_loss,
            )

    total_elapsed = time.time() - t_start
    result = {
        "status": "stopped" if stop_requested else "done",
        "stop_reason": stop_reason if stop_requested else "",
        "resumed": resume_info["resumed"],
        "run_name": run_name,
        "final_step": final_step,
        "max_steps": max_steps,
        "best_val_loss": best_val_loss,
        "elapsed_sec": round(total_elapsed, 2),
        "run_dir": str(run_dir.resolve()),
        "device": device,
        "dtype": dtype_name,
        "loss_mode": "response_only" if use_response_only_loss else "full_sequence",
    }
    save_json(run_dir / "last_status.json", result)
    write_jsonl(train_log_path, {"event": "final", **result})
    print(json.dumps(result, ensure_ascii=False))
    return result


def datetime_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT with auto-resume and config support.")
    parser.add_argument("--config_train", default="configs/train.yaml")
    parser.add_argument("--config_paths", default="configs/paths.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--resume", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_auto_resume", action="store_true")
    args = parser.parse_args()

    run_training(
        train_cfg_path=args.config_train,
        paths_cfg_path=args.config_paths,
        env_path=args.env_path,
        cli_resume=args.resume,
        cli_run_name=args.run_name,
        disable_auto_resume=args.no_auto_resume,
        cli_max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

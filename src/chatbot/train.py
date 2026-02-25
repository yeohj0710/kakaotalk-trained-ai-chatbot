from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .console import to_console_safe
from .model import GPT, GPTConfig
from .tokenizer import ByteSpecialTokenizer


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


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_index = len(data) - block_size - 1
    if max_index <= 0:
        raise ValueError("Dataset is too small for the configured block_size.")
    starts = torch.randint(0, max_index, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[idx : idx + block_size].astype(np.int64)) for idx in starts]
    )
    y = torch.stack(
        [torch.from_numpy(data[idx + 1 : idx + block_size + 1].astype(np.int64)) for idx in starts]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if autocast_dtype is not None
                else contextlib.nullcontext()
            )
            with context:
                _, loss = model(x, y)
            losses[k] = float(loss.item())
        out[split] = float(losses.mean())
    model.train()
    return out


def get_lr(step: int, max_steps: int, warmup_steps: int, learning_rate: float, min_lr: float) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    tokenizer_path: Path,
    model_config: GPTConfig,
    train_config: argparse.Namespace,
    step: int,
    best_val_loss: float,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": vars(train_config),
        "tokenizer_path": str(tokenizer_path.resolve()),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT from KakaoTalk room exports.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--out_dir", default="checkpoints/base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile if available.")
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from.")

    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bias", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=40)
    parser.add_argument("--save_interval", type=int, default=400)
    parser.add_argument("--log_interval", type=int, default=20)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sample_prompt", default="<|bos|>")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = data_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = ByteSpecialTokenizer.load(tokenizer_path)

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing train/val binaries in {data_dir}")

    train_data = np.fromfile(train_path, dtype=np.uint16)
    val_data = np.fromfile(val_path, dtype=np.uint16)
    if len(train_data) < 5000:
        raise RuntimeError("Train dataset is very small. Check preprocessing output.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    dtype_name, autocast_dtype = resolve_dtype(device, args.dtype)
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(model_config)
    model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(args.beta1, args.beta2),
        device_type=device_type,
    )

    scaler = create_grad_scaler(enabled=(device_type == "cuda" and dtype_name == "fp16"))
    autocast_context = (
        (lambda: torch.autocast(device_type="cuda", dtype=autocast_dtype))
        if autocast_dtype is not None
        else contextlib.nullcontext
    )

    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"Resumed from {args.resume} at step={start_step}")

    stats_path = out_dir / "train_log.jsonl"
    if start_step == 0:
        stats_path.write_text("", encoding="utf-8")

    t_start = time.time()
    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(
            step=step,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(args.grad_accum_steps):
            xb, yb = get_batch(
                train_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
            )
            with autocast_context():
                _, loss = model(xb, yb)
                scaled_loss = loss / args.grad_accum_steps
            loss_value += float(loss.item())
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

        if args.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            msg = {
                "step": step,
                "loss": loss_value / args.grad_accum_steps,
                "lr": lr,
                "elapsed_sec": round(elapsed, 2),
            }
            print(json.dumps(msg))
            with stats_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(msg, ensure_ascii=False) + "\n")

        should_eval = (step % args.eval_interval == 0) or (step == args.max_steps - 1)
        if should_eval:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                block_size=args.block_size,
                device=device,
                autocast_dtype=autocast_dtype,
            )
            eval_msg = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "best_val_loss": min(best_val_loss, losses["val"]),
            }
            print(json.dumps(eval_msg))
            with stats_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(eval_msg, ensure_ascii=False) + "\n")

            prompt_ids = tokenizer.encode(args.sample_prompt)
            xprompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                with autocast_context():
                    y = model.generate(
                        idx=xprompt,
                        max_new_tokens=120,
                        temperature=0.9,
                        top_k=120,
                        top_p=0.95,
                        eos_token_id=tokenizer.eos_id,
                    )
            sample = tokenizer.decode(y[0].tolist())
            print("=== SAMPLE ===")
            print(to_console_safe(sample[:500]))
            print("==============")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    path=out_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    tokenizer_path=tokenizer_path,
                    model_config=model_config,
                    train_config=args,
                    step=step,
                    best_val_loss=best_val_loss,
                )

        should_save = (step % args.save_interval == 0) or (step == args.max_steps - 1)
        if should_save:
            save_checkpoint(
                path=out_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                tokenizer_path=tokenizer_path,
                model_config=model_config,
                train_config=args,
                step=step,
                best_val_loss=best_val_loss,
            )
            save_checkpoint(
                path=out_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                tokenizer_path=tokenizer_path,
                model_config=model_config,
                train_config=args,
                step=step,
                best_val_loss=best_val_loss,
            )

    total_elapsed = time.time() - t_start
    print(
        json.dumps(
            {
                "status": "done",
                "steps": args.max_steps,
                "elapsed_sec": round(total_elapsed, 2),
                "best_val_loss": best_val_loss,
                "out_dir": str(out_dir.resolve()),
                "dtype": dtype_name,
                "device": device,
            }
        )
    )


if __name__ == "__main__":
    main()

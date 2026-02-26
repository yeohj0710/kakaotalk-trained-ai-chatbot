from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .security import require_password
from .sft_config import format_with_run_name, load_sft_config


def configure_console_io() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def sanitize_text(text: str, one_line: bool, max_chars: int) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00a0", " ").replace("\u200b", "")
    out = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", out)
    if one_line:
        out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def resolve_torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


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


def resolve_4bit_config(model_cfg: dict[str, Any], dtype: torch.dtype) -> Any | None:
    use_4bit = bool(model_cfg.get("load_in_4bit", True))
    if not use_4bit or not torch.cuda.is_available():
        return None
    try:
        import bitsandbytes  # type: ignore # noqa: F401
    except Exception:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        return None
    try:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
    except Exception:
        return None


@dataclass
class SFTInferOptions:
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_history_turns: int
    min_reply_chars: int
    regen_attempts: int
    one_line: bool
    max_chars: int


class SFTInferenceEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        options: SFTInferOptions,
        adapter_dir: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.options = options
        self.adapter_dir = adapter_dir

    @classmethod
    def load(
        cls,
        config_sft: str = "configs/sft.yaml",
        env_path: str = ".env",
        adapter_path: str = "",
        run_name_override: str = "",
    ) -> "SFTInferenceEngine":
        ensure_hf_env_defaults()
        cfg = load_sft_config(config_path=config_sft, env_path=env_path)
        project_cfg = dict(cfg.get("project", {}))
        paths_cfg = dict(cfg.get("paths", {}))
        model_cfg = dict(cfg.get("model", {}))
        gen_cfg = dict(cfg.get("generation", {}))
        prompt_cfg = dict(cfg.get("prompt", {}))

        run_name = run_name_override or str(project_cfg.get("run_name", "room_lora_qwen3b"))
        checkpoints_root = Path(str(paths_cfg.get("checkpoints_root", "checkpoints_lora")))
        run_dir = checkpoints_root / run_name
        status_json = Path(format_with_run_name(str(paths_cfg.get("status_json", "checkpoints_lora/{run_name}/status.json")), run_name))

        resolved_adapter: Path | None = Path(adapter_path) if adapter_path else None
        if resolved_adapter is None:
            best_dir = run_dir / "adapter_best"
            latest_dir = run_dir / "adapter_latest"
            if best_dir.exists():
                resolved_adapter = best_dir
            elif latest_dir.exists():
                resolved_adapter = latest_dir
            elif status_json.exists():
                status = json.loads(status_json.read_text(encoding="utf-8"))
                best_from_status = str(status.get("best_adapter_dir", "")).strip()
                latest_from_status = str(status.get("latest_adapter_dir", "")).strip()
                if best_from_status and Path(best_from_status).exists():
                    resolved_adapter = Path(best_from_status)
                elif latest_from_status and Path(latest_from_status).exists():
                    resolved_adapter = Path(latest_from_status)
        if resolved_adapter is None or not resolved_adapter.exists():
            raise FileNotFoundError("No adapter checkpoint found. Train first.")

        base_model = str(model_cfg.get("base_model", "Qwen/Qwen2.5-3B-Instruct")).strip()
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        local_files_only = bool(model_cfg.get("local_files_only", False))
        dtype = resolve_torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
        quant_config = resolve_4bit_config(model_cfg=model_cfg, dtype=dtype)

        tokenizer = retry_hf_load(
            lambda: AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=trust_remote_code,
                use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
                local_files_only=local_files_only,
            )
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "local_files_only": local_files_only,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        base = retry_hf_load(lambda: AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs))
        if quant_config is None and torch.cuda.is_available():
            base.to("cuda")
        model = PeftModel.from_pretrained(base, str(resolved_adapter))
        model.eval()

        options = SFTInferOptions(
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 120)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            temperature=float(gen_cfg.get("temperature", 0.72)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
            top_k=int(gen_cfg.get("top_k", 40)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.08)),
            max_history_turns=max(1, int(gen_cfg.get("max_history_turns", 14))),
            min_reply_chars=max(1, int(gen_cfg.get("min_reply_chars", 8))),
            regen_attempts=max(0, int(gen_cfg.get("regen_attempts", 2))),
            one_line=bool(gen_cfg.get("one_line", True)),
            max_chars=max(1, int(gen_cfg.get("max_chars", 240))),
        )
        system_prompt = str(prompt_cfg.get("system", "")).strip()
        return cls(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            options=options,
            adapter_dir=resolved_adapter,
        )

    def _build_prompt(self, history: list[tuple[str, str]], new_user_input: str) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        tail = history[-(self.options.max_history_turns * 2) :]
        for role, text in tail:
            if role == "bot":
                messages.append({"role": "assistant", "content": text})
            else:
                messages.append({"role": "user", "content": text})
        messages.append({"role": "user", "content": new_user_input})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Fallback template
        chunks = [f"[SYSTEM]\n{self.system_prompt}\n"]
        for msg in messages[1:]:
            role = "USER" if msg["role"] == "user" else "ASSISTANT"
            chunks.append(f"\n[{role}]\n{msg['content']}\n")
        chunks.append("\n[ASSISTANT]\n")
        return "".join(chunks)

    def _generate_once(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.options.max_new_tokens,
                do_sample=self.options.do_sample,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
                top_k=self.options.top_k,
                repetition_penalty=self.options.repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][prompt_len:]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return sanitize_text(raw, one_line=self.options.one_line, max_chars=self.options.max_chars)

    def reply(self, history: list[tuple[str, str]], user_text: str) -> str:
        prompt = self._build_prompt(history=history, new_user_input=user_text)
        result = ""
        for _ in range(self.options.regen_attempts + 1):
            candidate = self._generate_once(prompt)
            if len(candidate.replace(" ", "")) >= self.options.min_reply_chars:
                return candidate
            result = candidate
        return result or "..."


def parse_password_arg(password: str) -> str | None:
    value = (password or "").strip()
    return value if value else None


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(description="Quick single-turn inference test for SFT LoRA.")
    parser.add_argument("message", nargs="?", default="테스트")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    ensure_hf_env_defaults()
    security_cfg = dict(cfg.get("security", {}))
    env_name = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided = parse_password_arg(args.password) or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(env_name)
    require_password(security_cfg=security_cfg, password=provided, env_path=args.env_path)

    engine = SFTInferenceEngine.load(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name_override=args.run_name,
    )
    out = engine.reply(history=[], user_text=args.message)
    print(out)


if __name__ == "__main__":
    main()

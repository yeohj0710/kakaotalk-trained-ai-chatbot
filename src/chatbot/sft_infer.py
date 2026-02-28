from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
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
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00a0", " ").replace("\u200b", "")
    out = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", out)
    if one_line:
        out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def strip_generation_artifacts(text: str) -> str:
    out = (text or "").strip()
    if not out:
        return ""

    if "[ANSWER]" in out:
        out = out.rsplit("[ANSWER]", 1)[-1].strip()
    for marker in ("[SYSTEM]", "[TASK]", "[DIALOGUE]", "[ANSWER]"):
        idx = out.find(marker)
        if idx == 0:
            out = out[len(marker) :].strip()
        elif idx > 0:
            out = out[:idx].strip()
    for marker in ("user:", "assistant:", "bot:", "사용자:", "봇:"):
        idx = out.lower().find(marker)
        if idx > 0:
            out = out[:idx].strip()

    prefixes = ("답변:", "출력:", "assistant:", "bot:", "AI:")
    changed = True
    while changed and out:
        changed = False
        for prefix in prefixes:
            if out.lower().startswith(prefix.lower()):
                out = out[len(prefix) :].strip()
                changed = True
    return out


MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-가-힣]+")
SUMMARY_BULLET_RE = re.compile(r"(?m)^\s*[·•\-]\s+")
SUMMARY_HINT_RE = re.compile(r"(요약|정리|핵심|summary)", re.IGNORECASE)
COMPARE_TEXT_RE = re.compile(r"[^A-Za-z0-9가-힣]+")


def trim_mentions(text: str, max_mentions: int) -> str:
    if max_mentions < 0:
        return text

    if max_mentions == 0:
        out = MENTION_RE.sub("", text)
    else:
        kept = 0

        def _repl(match: re.Match[str]) -> str:
            nonlocal kept
            kept += 1
            return match.group(0) if kept <= max_mentions else ""

        out = MENTION_RE.sub(_repl, text)

    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s*([,;:])\s*", r"\1 ", out)
    return out.strip(" \t,;:")


def looks_like_summary_artifact(text: str) -> bool:
    out = (text or "").strip()
    if not out:
        return False
    if len(SUMMARY_BULLET_RE.findall(out)) >= 1:
        return True
    if SUMMARY_HINT_RE.search(out) and len(out) >= 20:
        return True
    return False


def soften_summary_artifact(text: str) -> str:
    out = (text or "").strip()
    if not out:
        return out
    out = SUMMARY_BULLET_RE.sub("", out)
    out = re.sub(r"(요약|정리|핵심)\s*[:：]?", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip(" \t,;:")


def normalize_compare_text(text: str) -> str:
    out = (text or "").lower()
    out = COMPARE_TEXT_RE.sub("", out)
    return out.strip()


def is_self_echo_candidate(candidate: str, history: list[tuple[str, str]]) -> bool:
    cand = normalize_compare_text(candidate)
    if len(cand) < 8:
        return False

    last_bot = ""
    for role, text in reversed(history):
        if role == "bot":
            last_bot = text
            break
    if not last_bot:
        return False

    prev = normalize_compare_text(last_bot)
    if len(prev) < 8:
        return False
    if cand == prev:
        return True
    if len(cand) >= 12 and (cand in prev or prev in cand):
        return True
    return SequenceMatcher(None, cand, prev).ratio() >= 0.9


def is_repetitive_candidate(text: str) -> bool:
    out = (text or "").strip()
    if not out:
        return False

    compact = re.sub(r"\s+", "", out)
    if re.search(r"(.{2,16})(?:\1){2,}", compact):
        return True

    tokens = out.split()
    if len(tokens) < 10:
        return False

    unique_ratio = len(set(tokens)) / max(1, len(tokens))
    if unique_ratio < 0.55:
        return True

    for n in (2, 3):
        if len(tokens) < n * 3:
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        top_count = Counter(ngrams).most_common(1)[0][1]
        if top_count >= 3 and top_count >= int(len(ngrams) * 0.5):
            return True
    return False


def soften_repetitive_candidate(text: str) -> str:
    tokens = (text or "").split()
    if not tokens:
        return (text or "").strip()

    dedup: list[str] = []
    prev = ""
    run = 0
    for tok in tokens:
        if tok == prev:
            run += 1
        else:
            prev = tok
            run = 1
        if run <= 2:
            dedup.append(tok)

    out = " ".join(dedup).strip()
    out = re.sub(r"(.)\1{5,}", r"\1\1\1", out)
    return out.strip()


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


def find_latest_checkpoint_dir(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if not suffix.isdigit():
            continue
        if not (path / "adapter_config.json").exists():
            continue
        if not ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()):
            continue
        candidates.append((int(suffix), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


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


def normalize_mode(raw: str) -> str:
    mode = (raw or "").strip().lower()
    if mode in {"1to1", "1:1", "one_to_one", "one_on_one", "one-on-one", "single"}:
        return "one_on_one"
    if mode in {"group", "room"}:
        return "group"
    return "group"


@dataclass
class SFTInferOptions:
    inference_mode: str
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    avoid_summary_artifacts: bool
    avoid_self_echo: bool
    avoid_repetitive_output: bool
    max_mentions: int
    max_history_turns: int
    max_bot_history_turns: int
    min_reply_chars: int
    candidate_count: int
    regen_attempts: int
    one_line: bool
    max_chars: int
    use_chat_template: bool
    group_min_user_turns_since_last_bot: int
    group_max_bot_turns_in_window: int
    group_block_consecutive_bot: bool
    group_no_reply_token: str


class SFTInferenceEngine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        system_prompt: str,
        task_prompt: str,
        options: SFTInferOptions,
        adapter_dir: Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.task_prompt = task_prompt
        self.options = options
        self.adapter_dir = adapter_dir

    @classmethod
    def load(
        cls,
        config_sft: str = "configs/sft.yaml",
        env_path: str = ".env",
        adapter_path: str = "",
        run_name_override: str = "",
        mode_override: str = "",
    ) -> "SFTInferenceEngine":
        ensure_hf_env_defaults()
        cfg = load_sft_config(config_path=config_sft, env_path=env_path)
        project_cfg = dict(cfg.get("project", {}))
        paths_cfg = dict(cfg.get("paths", {}))
        model_cfg = dict(cfg.get("model", {}))
        gen_cfg = dict(cfg.get("generation", {}))
        prompt_cfg = dict(cfg.get("prompt", {}))

        run_name = run_name_override or str(project_cfg.get("run_name", "room_lora_qwen25_7b_group_v2"))
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
            checkpoint_fallback = find_latest_checkpoint_dir(run_dir)
            if checkpoint_fallback is not None:
                resolved_adapter = checkpoint_fallback
                print(
                    json.dumps(
                        {"event": "adapter_fallback_checkpoint", "adapter_dir": str(resolved_adapter.as_posix())},
                        ensure_ascii=False,
                    )
                )

        if resolved_adapter is None or not resolved_adapter.exists():
            raise FileNotFoundError("No adapter checkpoint found. Train first.")

        base_model = str(model_cfg.get("base_model", "Qwen/Qwen2.5-7B")).strip()
        trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
        local_files_only = bool(model_cfg.get("local_files_only", False))
        dtype = resolve_torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
        quant_config = resolve_4bit_config(model_cfg=model_cfg, dtype=dtype)
        if bool(model_cfg.get("load_in_4bit", True)) and quant_config is None:
            print(
                json.dumps(
                    {
                        "event": "quantization_warning",
                        "requested_4bit": True,
                        "quantization": "fallback_full_precision",
                        "hint": "Install bitsandbytes for 4bit or keep full precision fallback.",
                    },
                    ensure_ascii=False,
                )
            )

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
            "torch_dtype": dtype,
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

        configured_mode = normalize_mode(str(gen_cfg.get("inference_mode", "group")))
        if mode_override:
            configured_mode = normalize_mode(mode_override)
        options = SFTInferOptions(
            inference_mode=configured_mode,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 96)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            temperature=float(gen_cfg.get("temperature", 0.85)),
            top_p=float(gen_cfg.get("top_p", 0.92)),
            top_k=int(gen_cfg.get("top_k", 50)),
            repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.05)),
            no_repeat_ngram_size=max(0, int(gen_cfg.get("no_repeat_ngram_size", 4))),
            avoid_summary_artifacts=bool(gen_cfg.get("avoid_summary_artifacts", True)),
            avoid_self_echo=bool(gen_cfg.get("avoid_self_echo", True)),
            avoid_repetitive_output=bool(gen_cfg.get("avoid_repetitive_output", True)),
            max_mentions=int(gen_cfg.get("max_mentions", 0)),
            max_history_turns=max(1, int(gen_cfg.get("max_history_turns", 8))),
            max_bot_history_turns=max(0, int(gen_cfg.get("max_bot_history_turns", 2))),
            min_reply_chars=max(1, int(gen_cfg.get("min_reply_chars", 8))),
            candidate_count=max(1, int(gen_cfg.get("candidate_count", 3))),
            regen_attempts=max(0, int(gen_cfg.get("regen_attempts", 2))),
            one_line=bool(gen_cfg.get("one_line", True)),
            max_chars=max(1, int(gen_cfg.get("max_chars", 220))),
            use_chat_template=bool(gen_cfg.get("use_chat_template", False)),
            group_min_user_turns_since_last_bot=max(0, int(gen_cfg.get("group_min_user_turns_since_last_bot", 4))),
            group_max_bot_turns_in_window=max(0, int(gen_cfg.get("group_max_bot_turns_in_window", 2))),
            group_block_consecutive_bot=bool(gen_cfg.get("group_block_consecutive_bot", True)),
            group_no_reply_token=str(gen_cfg.get("group_no_reply_token", "<NO_REPLY>")),
        )
        system_prompt = str(prompt_cfg.get("system", "")).strip()
        task_prompt = str(prompt_cfg.get("task", "")).strip()
        return cls(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            options=options,
            adapter_dir=resolved_adapter,
        )

    def _context_text(self, text: str) -> str:
        out = sanitize_text(text, one_line=True, max_chars=0)
        return out[:700].strip()

    def _select_history(self, history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        selected_rev: list[tuple[str, str]] = []
        bot_left = self.options.max_bot_history_turns
        for role, text in reversed(history):
            if role == "bot" and bot_left <= 0:
                continue
            if role == "bot":
                bot_left -= 1
            selected_rev.append((role, text))
            if len(selected_rev) >= self.options.max_history_turns:
                break
        return list(reversed(selected_rev))

    def _build_prompt(self, history: list[tuple[str, str]], new_user_input: str) -> str:
        tail = self._select_history(history)
        if self.options.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
            for role, text in tail:
                if role == "bot":
                    messages.append({"role": "assistant", "content": text})
                else:
                    messages.append({"role": "user", "content": text})
            messages.append({"role": "user", "content": new_user_input})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        lines: list[str] = []
        for role, text in tail:
            speaker = "bot" if role == "bot" else "user"
            lines.append(f"{speaker}: {self._context_text(text)}")
        lines.append(f"user: {self._context_text(new_user_input)}")
        context_text = "\n".join(lines)
        return (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[TASK]\n{self.task_prompt}\n\n"
            f"[DIALOGUE]\n{context_text}\n\n"
            "[ANSWER]\n"
        )

    def _generate_once(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.options.max_new_tokens,
            "do_sample": self.options.do_sample,
            "repetition_penalty": self.options.repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.options.do_sample:
            gen_kwargs["temperature"] = self.options.temperature
            gen_kwargs["top_p"] = self.options.top_p
            gen_kwargs["top_k"] = self.options.top_k
        if self.options.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = self.options.no_repeat_ngram_size
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][prompt_len:]
        raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned = sanitize_text(raw, one_line=self.options.one_line, max_chars=self.options.max_chars)
        out = strip_generation_artifacts(cleaned)
        return trim_mentions(out, max_mentions=self.options.max_mentions)

    def _score_candidate(self, candidate: str, history: list[tuple[str, str]]) -> tuple[int, str]:
        score = 100
        out = candidate
        if len(out.replace(" ", "")) < self.options.min_reply_chars:
            score -= 40
        if self.options.avoid_summary_artifacts and looks_like_summary_artifact(out):
            score -= 25
            out = soften_summary_artifact(out)
        if self.options.avoid_self_echo and is_self_echo_candidate(out, history):
            score -= 30
        if self.options.avoid_repetitive_output and is_repetitive_candidate(out):
            score -= 30
            out = soften_repetitive_candidate(out)
        if not out:
            score -= 100
        return score, out

    def generate_reply(self, history: list[tuple[str, str]], user_text: str) -> str:
        prompt = self._build_prompt(history=history, new_user_input=user_text)
        total_trials = max(1, self.options.candidate_count + self.options.regen_attempts)
        best_score = -10_000
        best_text = ""
        for _ in range(total_trials):
            candidate = self._generate_once(prompt)
            score, normalized = self._score_candidate(candidate, history)
            if score > best_score:
                best_score = score
                best_text = normalized
            if score >= 100:
                break
        return best_text or "..."

    def should_reply(self, history: list[tuple[str, str]], user_text: str) -> bool:
        _ = user_text
        if self.options.inference_mode != "group":
            return True
        if self.options.group_block_consecutive_bot and history and history[-1][0] == "bot":
            return False

        if self.options.group_min_user_turns_since_last_bot > 0:
            user_since_last_bot = 0
            for role, _text in reversed(history):
                if role == "bot":
                    break
                user_since_last_bot += 1
            if user_since_last_bot < self.options.group_min_user_turns_since_last_bot:
                return False

        if self.options.group_max_bot_turns_in_window > 0:
            tail = history[-self.options.max_history_turns :]
            bot_count = sum(1 for role, _text in tail if role == "bot")
            if bot_count >= self.options.group_max_bot_turns_in_window:
                return False
        return True

    def reply_or_skip(self, history: list[tuple[str, str]], user_text: str, force_reply: bool = False) -> tuple[bool, str]:
        if force_reply or self.should_reply(history=history, user_text=user_text):
            return True, self.generate_reply(history=history, user_text=user_text)
        return False, self.options.group_no_reply_token

    def reply(self, history: list[tuple[str, str]], user_text: str) -> str:
        return self.generate_reply(history=history, user_text=user_text)


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
    parser.add_argument("--mode", default="one_on_one")
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
        mode_override=args.mode,
    )
    out = engine.reply(history=[], user_text=args.message)
    print(out)


if __name__ == "__main__":
    main()

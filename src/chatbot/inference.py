from __future__ import annotations

import contextlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch

from .model import GPT, GPTConfig
from .tokenizer import ByteSpecialTokenizer


SPECIAL_SPEAKER_PATTERN = re.compile(r"<\|spk:[^|>]+?\|>")


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_autocast_dtype(device: str, dtype: str) -> torch.dtype | None:
    if device != "cuda":
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return None
    if dtype == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


class InferenceEngine:
    def __init__(
        self,
        model: GPT,
        tokenizer: ByteSpecialTokenizer,
        device: str,
        autocast_dtype: torch.dtype | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_dtype = autocast_dtype

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        dtype: str = "auto",
    ) -> "InferenceEngine":
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        tokenizer_path = checkpoint.get("tokenizer_path")
        if tokenizer_path is None:
            tokenizer_path = checkpoint_path.parent / "tokenizer.json"
        tokenizer = ByteSpecialTokenizer.load(tokenizer_path)

        config = GPTConfig(**checkpoint["model_config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        resolved_device = resolve_device(device)
        model.to(resolved_device)
        model.eval()

        autocast_dtype = resolve_autocast_dtype(resolved_device, dtype)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            autocast_dtype=autocast_dtype,
        )

    def _autocast_context(self) -> contextlib.AbstractContextManager:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def complete(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 120,
        top_p: float = 0.95,
        repetition_penalty: float = 1.02,
        stop_on_eos: bool = True,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            with self._autocast_context():
                y = self.model.generate(
                    idx=x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_id if stop_on_eos else None,
                    repetition_penalty=repetition_penalty,
                )
        generated_ids = y[0].tolist()[len(input_ids) :]
        decoded = self.tokenizer.decode(generated_ids)
        return decoded

    def build_chat_prompt(self, history: Iterable[tuple[str, str]], next_speaker: str) -> str:
        chunks = []
        for speaker, text in history:
            speaker_token = self.tokenizer.speaker_token(speaker)
            chunks.append(f"{self.tokenizer.bos_token}{speaker_token}{text}{self.tokenizer.eos_token}\n")
        chunks.append(f"{self.tokenizer.bos_token}{self.tokenizer.speaker_token(next_speaker)}")
        return "".join(chunks)

    def extract_reply_text(self, generated_fragment: str) -> str:
        text = generated_fragment
        if self.tokenizer.eos_token in text:
            text = text.split(self.tokenizer.eos_token, 1)[0]
        text = text.replace(self.tokenizer.bos_token, "")
        text = SPECIAL_SPEAKER_PATTERN.sub("", text)
        return text.strip()

    def generate_reply(
        self,
        history: list[tuple[str, str]],
        bot_speaker: str,
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 120,
        top_p: float = 0.95,
        repetition_penalty: float = 1.02,
    ) -> str:
        prompt = self.build_chat_prompt(history=history, next_speaker=bot_speaker)
        raw_fragment = self.complete(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=True,
        )
        reply = self.extract_reply_text(raw_fragment)
        return reply or "..."

    def metadata(self) -> dict[str, object]:
        return {
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
            "speakers": self.tokenizer.speaker_names,
            "config": asdict(self.model.config),
        }

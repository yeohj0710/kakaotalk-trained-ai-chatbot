from __future__ import annotations

import contextlib
import io
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from .config import load_gen_config
from .crypto_utils import decrypt_file_to_bytes
from .model import GPT, GPTConfig
from .security import get_model_key
from .text_postprocess import OutputOptions, postprocess_generated_text, sanitize_for_send
from .tokenizer import ByteSpecialTokenizer


@dataclass
class DialogueOptions:
    mode: str = "anonymous"
    max_turns: int = 12
    max_context_tokens: int = 2048
    user_tag: str = "<U>"
    bot_tag: str = "<B>"
    user_speaker: str = ""
    bot_speaker: str = ""


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


def resolve_checkpoint_path(
    checkpoint_path: str | Path | None = None,
    gen_config_path: str = "configs/gen.yaml",
    env_path: str = ".env",
    run_name_override: str = "",
) -> Path:
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    gen_cfg = load_gen_config(config_path=gen_config_path, env_path=env_path)
    runtime_cfg = dict(gen_cfg.get("runtime", {}))
    configured_ckpt = str(runtime_cfg.get("checkpoint", "")).strip()
    if configured_ckpt:
        configured = Path(configured_ckpt)
        if configured.exists():
            return configured

    run_name = run_name_override or str(runtime_cfg.get("run_name", "room_v1"))
    for pattern in runtime_cfg.get("checkpoint_preference", []):
        candidate = Path(str(pattern).format(run_name=run_name))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No checkpoint found. Checked configured checkpoint and checkpoint_preference in configs/gen.yaml."
    )


def _load_checkpoint_payload(
    ckpt_path: Path,
    gen_config_path: str,
    env_path: str,
) -> dict:
    if ckpt_path.suffix.lower() == ".enc":
        gen_cfg = load_gen_config(config_path=gen_config_path, env_path=env_path)
        security_cfg = dict(gen_cfg.get("security", {}))
        key = get_model_key(security_cfg=security_cfg, env_path=env_path)
        raw = decrypt_file_to_bytes(source_path=ckpt_path, secret=key)
        return torch.load(io.BytesIO(raw), map_location="cpu")
    return torch.load(ckpt_path, map_location="cpu")


class InferenceEngine:
    def __init__(
        self,
        model: GPT,
        tokenizer: ByteSpecialTokenizer,
        device: str,
        autocast_dtype: torch.dtype | None,
        checkpoint_path: Path,
        output_options: OutputOptions,
        debug_return_raw: bool,
        dialogue_options: DialogueOptions,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.checkpoint_path = checkpoint_path
        self.output_options = output_options
        self.debug_return_raw = debug_return_raw
        self.dialogue_options = dialogue_options

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        gen_config_path: str = "configs/gen.yaml",
        env_path: str = ".env",
        run_name_override: str = "",
    ) -> "InferenceEngine":
        gen_cfg = load_gen_config(config_path=gen_config_path, env_path=env_path)
        output_cfg = dict(gen_cfg.get("output", {}))
        debug_cfg = dict(gen_cfg.get("debug", {}))
        chat_cfg = dict(gen_cfg.get("chat", {}))
        dialogue_cfg = dict(gen_cfg.get("dialogue", {}))

        resolved_ckpt = resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
            gen_config_path=gen_config_path,
            env_path=env_path,
            run_name_override=run_name_override,
        )
        checkpoint = _load_checkpoint_payload(
            ckpt_path=resolved_ckpt,
            gen_config_path=gen_config_path,
            env_path=env_path,
        )

        tokenizer_state = checkpoint.get("tokenizer_state")
        if isinstance(tokenizer_state, dict):
            tokenizer = ByteSpecialTokenizer(
                speaker_to_token=dict(tokenizer_state.get("speaker_to_token", {})),
                special_tokens=list(tokenizer_state.get("special_tokens", [])),
            )
        else:
            tokenizer_path = checkpoint.get("tokenizer_path")
            if tokenizer_path is None:
                tokenizer_path = resolved_ckpt.parent / "tokenizer.json"
            tokenizer_path = Path(tokenizer_path)
            if not tokenizer_path.exists():
                fallback = Path("data/processed/tokenizer.json")
                if fallback.exists():
                    tokenizer_path = fallback
                else:
                    raise FileNotFoundError(
                        f"Tokenizer file not found: {tokenizer_path}. "
                        "Run preprocessing or use a checkpoint with embedded tokenizer_state."
                    )
            tokenizer = ByteSpecialTokenizer.load(tokenizer_path)

        config = GPTConfig(**checkpoint["model_config"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        resolved_device = resolve_device(device)
        model.to(resolved_device)
        model.eval()

        autocast_dtype = resolve_autocast_dtype(resolved_device, dtype)
        options = OutputOptions(
            text_only=bool(output_cfg.get("text_only", True)),
            max_chars=int(output_cfg.get("max_chars", 400)),
            strip_prefix=bool(output_cfg.get("strip_prefix", True)),
            stop_on_next_turn=bool(output_cfg.get("stop_on_next_turn", True)),
            user_tag=str(dialogue_cfg.get("user_tag", "<U>")),
            bot_tag=str(dialogue_cfg.get("bot_tag", "<B>")),
        )
        mode = str(dialogue_cfg.get("mode", "anonymous")).strip().lower()
        if mode not in {"anonymous", "named"}:
            mode = "anonymous"
        dialogue_options = DialogueOptions(
            mode=mode,
            max_turns=max(1, int(dialogue_cfg.get("max_turns", chat_cfg.get("max_history_turns", 12)))),
            max_context_tokens=max(1, int(dialogue_cfg.get("max_context_tokens", 2048))),
            user_tag=str(dialogue_cfg.get("user_tag", "<U>")),
            bot_tag=str(dialogue_cfg.get("bot_tag", "<B>")),
            user_speaker=str(dialogue_cfg.get("user_speaker", chat_cfg.get("user_speaker", ""))).strip(),
            bot_speaker=str(dialogue_cfg.get("bot_speaker", chat_cfg.get("bot_speaker", ""))).strip(),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            autocast_dtype=autocast_dtype,
            checkpoint_path=resolved_ckpt,
            output_options=options,
            debug_return_raw=bool(debug_cfg.get("return_raw", False)),
            dialogue_options=dialogue_options,
        )

    def _autocast_context(self) -> contextlib.AbstractContextManager:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    @property
    def dialogue_mode(self) -> str:
        return self.dialogue_options.mode

    def _default_speakers(self) -> tuple[str, str]:
        speakers = self.tokenizer.speaker_names
        if not speakers:
            raise RuntimeError("No speaker information in tokenizer.")
        bot = self.dialogue_options.bot_speaker
        user = self.dialogue_options.user_speaker
        if bot not in speakers:
            bot = speakers[0]
        if user not in speakers:
            user = speakers[0]
            for candidate in speakers:
                if candidate != bot:
                    user = candidate
                    break
        return user, bot

    def resolve_dialogue_speakers(
        self,
        user_speaker: str | None = None,
        bot_speaker: str | None = None,
    ) -> tuple[str, str]:
        default_user, default_bot = self._default_speakers()
        resolved_user = (user_speaker or "").strip() or default_user
        resolved_bot = (bot_speaker or "").strip() or default_bot
        if resolved_bot not in self.tokenizer.speaker_names:
            resolved_bot = default_bot
        if resolved_user not in self.tokenizer.speaker_names:
            resolved_user = default_user
        return resolved_user, resolved_bot

    def _to_role(
        self,
        speaker: str,
        user_speaker: str,
        bot_speaker: str,
    ) -> str:
        normalized = speaker.strip().lower()
        if normalized in {"bot", "assistant", "ai", "b"}:
            return "bot"
        if normalized in {"user", "human", "u", "me"}:
            return "user"
        if speaker == bot_speaker:
            return "bot"
        if speaker == user_speaker:
            return "user"
        return "user"

    def _effective_context_tokens(self, override: int | None = None) -> int:
        configured = int(override or self.dialogue_options.max_context_tokens)
        if configured <= 0:
            configured = self.model.config.block_size
        return int(min(configured, self.model.config.block_size))

    def _trim_history(
        self,
        history: list[tuple[str, str]],
        bot_speaker: str,
        max_turns_override: int | None = None,
        max_context_tokens_override: int | None = None,
    ) -> list[tuple[str, str]]:
        trimmed = list(history)
        max_turns = int(max_turns_override or self.dialogue_options.max_turns)
        if max_turns > 0:
            max_messages = max_turns * 2
            if len(trimmed) > max_messages:
                trimmed = trimmed[-max_messages:]

        max_context_tokens = self._effective_context_tokens(max_context_tokens_override)
        while len(trimmed) > 1:
            prompt = self.build_chat_prompt(
                history=trimmed,
                next_speaker=bot_speaker,
            )
            token_count = len(self.tokenizer.encode(prompt))
            if token_count <= max_context_tokens:
                break
            # Drop the oldest full turn (user+bot) first.
            drop = 2 if len(trimmed) >= 2 else 1
            trimmed = trimmed[drop:]
        return trimmed

    def prepare_dialogue_history(
        self,
        history: Iterable[tuple[str, str]],
        user_speaker: str | None = None,
        bot_speaker: str | None = None,
        max_turns_override: int | None = None,
        max_context_tokens_override: int | None = None,
    ) -> tuple[list[tuple[str, str]], str, str]:
        resolved_user, resolved_bot = self.resolve_dialogue_speakers(
            user_speaker=user_speaker,
            bot_speaker=bot_speaker,
        )
        normalized: list[tuple[str, str]] = []
        for speaker, text in history:
            cleaned = sanitize_for_send(text, max_chars=0)
            if not cleaned:
                continue
            if self.dialogue_mode == "anonymous":
                role = self._to_role(speaker=speaker, user_speaker=resolved_user, bot_speaker=resolved_bot)
                target_speaker = resolved_user if role == "user" else resolved_bot
                normalized.append((target_speaker, cleaned))
            else:
                target = speaker if speaker in self.tokenizer.speaker_names else resolved_user
                normalized.append((target, cleaned))
        trimmed = self._trim_history(
            history=normalized,
            bot_speaker=resolved_bot,
            max_turns_override=max_turns_override,
            max_context_tokens_override=max_context_tokens_override,
        )
        return trimmed, resolved_user, resolved_bot

    def _stop_token_ids(self, bot_speaker: str | None, stop_on_eos: bool, stop_on_next_turn: bool) -> list[int]:
        token_ids: set[int] = set()
        if stop_on_eos:
            token_ids.add(self.tokenizer.eos_id)
        if stop_on_next_turn:
            token_ids.add(self.tokenizer.bos_id)
            current_bot_token = self.tokenizer.speaker_token(bot_speaker) if bot_speaker else None
            for speaker_token in set(self.tokenizer.speaker_to_token.values()):
                if speaker_token == current_bot_token:
                    continue
                token_id = self.tokenizer.token_to_id.get(speaker_token)
                if token_id is not None:
                    token_ids.add(int(token_id))
        return sorted(token_ids)

    def _generate_raw_fragment(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        stop_on_eos: bool,
        stop_on_next_turn: bool,
        bot_speaker: str | None,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        stop_ids = self._stop_token_ids(
            bot_speaker=bot_speaker,
            stop_on_eos=stop_on_eos,
            stop_on_next_turn=stop_on_next_turn,
        )
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
                    stop_token_ids=stop_ids,
                )
        generated_ids = y[0].tolist()[len(input_ids) :]
        return self.tokenizer.decode(generated_ids)

    def _postprocess(self, raw_text: str, bot_speaker: str | None) -> str:
        return postprocess_generated_text(
            raw_text=raw_text,
            tokenizer=self.tokenizer,
            bot_speaker=bot_speaker,
            options=self.output_options,
        )

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
        raw_fragment = self._generate_raw_fragment(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=stop_on_eos,
            stop_on_next_turn=self.output_options.stop_on_next_turn,
            bot_speaker=None,
        )
        return self._postprocess(raw_text=raw_fragment, bot_speaker=None)

    def complete_with_raw(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 120,
        top_p: float = 0.95,
        repetition_penalty: float = 1.02,
        stop_on_eos: bool = True,
    ) -> tuple[str, str]:
        raw_fragment = self._generate_raw_fragment(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=stop_on_eos,
            stop_on_next_turn=self.output_options.stop_on_next_turn,
            bot_speaker=None,
        )
        processed = self._postprocess(raw_text=raw_fragment, bot_speaker=None)
        return processed, sanitize_for_send(raw_fragment, max_chars=0)

    def build_chat_prompt(self, history: Iterable[tuple[str, str]], next_speaker: str) -> str:
        chunks = []
        for speaker, text in history:
            speaker_token = self.tokenizer.speaker_token(speaker)
            chunks.append(f"{self.tokenizer.bos_token}{speaker_token}{text}{self.tokenizer.eos_token}\n")
        chunks.append(f"{self.tokenizer.bos_token}{self.tokenizer.speaker_token(next_speaker)}")
        return "".join(chunks)

    def generate_reply(
        self,
        history: list[tuple[str, str]],
        bot_speaker: str = "",
        user_speaker: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 120,
        top_p: float = 0.95,
        repetition_penalty: float = 1.02,
        max_turns_override: int | None = None,
        max_context_tokens_override: int | None = None,
    ) -> str:
        trimmed_history, _, resolved_bot = self.prepare_dialogue_history(
            history=history,
            user_speaker=(user_speaker or None),
            bot_speaker=(bot_speaker or None),
            max_turns_override=max_turns_override,
            max_context_tokens_override=max_context_tokens_override,
        )
        prompt = self.build_chat_prompt(history=trimmed_history, next_speaker=resolved_bot)
        raw_fragment = self._generate_raw_fragment(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=True,
            stop_on_next_turn=self.output_options.stop_on_next_turn,
            bot_speaker=resolved_bot,
        )
        reply = self._postprocess(raw_text=raw_fragment, bot_speaker=resolved_bot)
        return reply or "..."

    def generate_reply_with_raw(
        self,
        history: list[tuple[str, str]],
        bot_speaker: str = "",
        user_speaker: str = "",
        max_new_tokens: int = 120,
        temperature: float = 0.9,
        top_k: int = 120,
        top_p: float = 0.95,
        repetition_penalty: float = 1.02,
        max_turns_override: int | None = None,
        max_context_tokens_override: int | None = None,
    ) -> tuple[str, str]:
        trimmed_history, _, resolved_bot = self.prepare_dialogue_history(
            history=history,
            user_speaker=(user_speaker or None),
            bot_speaker=(bot_speaker or None),
            max_turns_override=max_turns_override,
            max_context_tokens_override=max_context_tokens_override,
        )
        prompt = self.build_chat_prompt(history=trimmed_history, next_speaker=resolved_bot)
        raw_fragment = self._generate_raw_fragment(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_on_eos=True,
            stop_on_next_turn=self.output_options.stop_on_next_turn,
            bot_speaker=resolved_bot,
        )
        processed = self._postprocess(raw_text=raw_fragment, bot_speaker=resolved_bot) or "..."
        return processed, sanitize_for_send(raw_fragment, max_chars=0)

    def metadata(self) -> dict[str, object]:
        return {
            "device": self.device,
            "checkpoint_path": str(self.checkpoint_path),
            "vocab_size": self.tokenizer.vocab_size,
            "speakers": self.tokenizer.speaker_names,
            "config": asdict(self.model.config),
            "output_options": asdict(self.output_options),
            "debug_return_raw": self.debug_return_raw,
            "dialogue_options": asdict(self.dialogue_options),
        }

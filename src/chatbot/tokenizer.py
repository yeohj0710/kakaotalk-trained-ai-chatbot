from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


UNKNOWN_SPEAKER_KEY = "__UNKNOWN__"


def sanitize_speaker_name(name: str) -> str:
    cleaned = re.sub(r"[|<>\r\n\t]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "UNKNOWN"


def build_speaker_token_map(speakers: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used_tokens: set[str] = set()
    for speaker in sorted(set(speakers)):
        base = sanitize_speaker_name(speaker)
        token = f"<|spk:{base}|>"
        suffix = 2
        while token in used_tokens:
            token = f"<|spk:{base}_{suffix}|>"
            suffix += 1
        mapping[speaker] = token
        used_tokens.add(token)
    return mapping


@dataclass
class TokenizerState:
    version: int
    special_tokens: list[str]
    speaker_to_token: dict[str, str]


class ByteSpecialTokenizer:
    BASE_SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|spk:UNKNOWN|>"]

    def __init__(
        self,
        speaker_to_token: dict[str, str] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self.speaker_to_token: dict[str, str] = dict(speaker_to_token or {})
        if UNKNOWN_SPEAKER_KEY not in self.speaker_to_token:
            self.speaker_to_token[UNKNOWN_SPEAKER_KEY] = "<|spk:UNKNOWN|>"

        speaker_tokens = sorted(set(self.speaker_to_token.values()))
        merged_specials = list(self.BASE_SPECIAL_TOKENS)
        for token in speaker_tokens:
            if token not in merged_specials:
                merged_specials.append(token)
        if special_tokens is not None:
            merged_specials = list(dict.fromkeys(special_tokens))

        self.special_tokens = merged_specials
        self.token_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.byte_offset = len(self.special_tokens)
        self.vocab_size = self.byte_offset + 256
        self._token_pattern = self._build_special_pattern(self.special_tokens)
        self.token_to_speaker = {
            token: speaker
            for speaker, token in self.speaker_to_token.items()
            if speaker != UNKNOWN_SPEAKER_KEY
        }

    @staticmethod
    def _build_special_pattern(special_tokens: list[str]) -> re.Pattern[str]:
        escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
        pattern = "|".join(escaped)
        return re.compile(f"({pattern})")

    @property
    def pad_token(self) -> str:
        return "<|pad|>"

    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def eos_token(self) -> str:
        return "<|eos|>"

    @property
    def unknown_speaker_token(self) -> str:
        return self.speaker_to_token[UNKNOWN_SPEAKER_KEY]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def speaker_names(self) -> list[str]:
        return sorted([name for name in self.speaker_to_token if name != UNKNOWN_SPEAKER_KEY])

    def speaker_token(self, speaker_name: str) -> str:
        return self.speaker_to_token.get(speaker_name, self.unknown_speaker_token)

    def speaker_from_token(self, speaker_token: str) -> str:
        return self.token_to_speaker.get(speaker_token, UNKNOWN_SPEAKER_KEY)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        payload = text
        if add_bos:
            payload = self.bos_token + payload
        if add_eos:
            payload = payload + self.eos_token

        ids: list[int] = []
        for chunk in self._token_pattern.split(payload):
            if not chunk:
                continue
            if chunk in self.token_to_id:
                ids.append(self.token_to_id[chunk])
            else:
                byte_values = chunk.encode("utf-8", errors="replace")
                ids.extend(self.byte_offset + b for b in byte_values)
        return ids

    def decode(self, token_ids: Iterable[int]) -> str:
        out_chunks: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                out_chunks.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id < self.byte_offset:
                flush_bytes()
                out_chunks.append(self.id_to_token.get(token_id, ""))
            else:
                byte_value = token_id - self.byte_offset
                if 0 <= byte_value <= 255:
                    byte_buffer.append(byte_value)
        flush_bytes()
        return "".join(out_chunks)

    def to_state(self) -> TokenizerState:
        return TokenizerState(
            version=1,
            special_tokens=self.special_tokens,
            speaker_to_token=self.speaker_to_token,
        )

    def save(self, path: str | Path) -> None:
        state = self.to_state()
        Path(path).write_text(
            json.dumps(
                {
                    "version": state.version,
                    "special_tokens": state.special_tokens,
                    "speaker_to_token": state.speaker_to_token,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_state(cls, state: TokenizerState) -> "ByteSpecialTokenizer":
        return cls(
            speaker_to_token=state.speaker_to_token,
            special_tokens=state.special_tokens,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ByteSpecialTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        state = TokenizerState(
            version=int(payload["version"]),
            special_tokens=list(payload["special_tokens"]),
            speaker_to_token=dict(payload["speaker_to_token"]),
        )
        return cls.from_state(state)

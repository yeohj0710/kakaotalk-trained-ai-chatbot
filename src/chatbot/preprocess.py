from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_paths_config
from .tokenizer import ByteSpecialTokenizer, build_speaker_token_map


MESSAGE_PATTERN = re.compile(
    r"^(?P<ts>\d{4}\. \d{1,2}\. \d{1,2}\. \d{1,2}:\d{2}), (?P<speaker>[^:]+) : (?P<text>.*)$"
)
SYSTEM_PATTERN = re.compile(
    r"^(?P<ts>\d{4}\. \d{1,2}\. \d{1,2}\. \d{1,2}:\d{2}): (?P<event>.*)$"
)
DATE_HEADER_PATTERN = re.compile(r"^\d{4}년 \d{1,2}월 \d{1,2}일 .+$")
SAVED_AT_PATTERN = re.compile(r"^저장한 날짜 : .+$")
DELETED_PATTERN = re.compile(r"^메시지가 삭제되었습니다\.$")
MEDIA_PATTERN = re.compile(r"^(사진|동영상|이모티콘|음성메시지|파일)(:.*)?$")


@dataclass
class ChatMessage:
    timestamp: datetime
    speaker: str
    text: str
    source_file: str


def read_chat_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("chat", b"", 0, 1, f"Unable to decode file: {path}")


def parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y. %m. %d. %H:%M")


def parse_kakao_export(path: Path, include_system: bool = False) -> list[ChatMessage]:
    text = read_chat_text(path)
    lines = text.splitlines()
    messages: list[ChatMessage] = []

    active_ts: datetime | None = None
    active_speaker: str | None = None
    active_lines: list[str] = []

    def flush_active() -> None:
        nonlocal active_ts, active_speaker, active_lines
        if active_ts is None or active_speaker is None:
            return
        joined = "\n".join(active_lines).strip()
        if joined:
            messages.append(
                ChatMessage(
                    timestamp=active_ts,
                    speaker=active_speaker,
                    text=joined,
                    source_file=path.name,
                )
            )
        active_ts = None
        active_speaker = None
        active_lines = []

    for line in lines:
        line = line.replace("\ufeff", "")
        if MESSAGE_PATTERN.match(line):
            flush_active()
            match = MESSAGE_PATTERN.match(line)
            if match is None:
                continue
            active_ts = parse_timestamp(match.group("ts"))
            active_speaker = match.group("speaker").strip()
            active_lines = [match.group("text")]
            continue

        if SYSTEM_PATTERN.match(line):
            flush_active()
            if include_system:
                match = SYSTEM_PATTERN.match(line)
                if match is None:
                    continue
                messages.append(
                    ChatMessage(
                        timestamp=parse_timestamp(match.group("ts")),
                        speaker="[SYSTEM]",
                        text=match.group("event").strip(),
                        source_file=path.name,
                    )
                )
            continue

        if SAVED_AT_PATTERN.match(line) or DATE_HEADER_PATTERN.match(line):
            continue

        if line.strip().endswith("카카오톡 대화") and "Talk_" in line:
            continue

        if not line.strip():
            if active_lines:
                active_lines.append("")
            continue

        if DELETED_PATTERN.match(line):
            continue

        if active_lines:
            active_lines.append(line)

    flush_active()
    return messages


def clean_message(
    text: str,
    mask_urls: bool = True,
    mask_numbers: bool = False,
    drop_media_only: bool = False,
) -> str:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    cleaned = re.sub(r"\u200b+", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if mask_urls:
        cleaned = re.sub(r"https?://\S+", "<URL>", cleaned)
    if mask_numbers:
        cleaned = re.sub(r"\b\d{3,}\b", "<NUM>", cleaned)
    if drop_media_only and MEDIA_PATTERN.fullmatch(cleaned):
        return ""
    return cleaned


def build_lm_corpus(messages: list[ChatMessage], tokenizer: ByteSpecialTokenizer) -> str:
    rows = []
    for msg in messages:
        speaker_token = tokenizer.speaker_token(msg.speaker)
        rows.append(f"{tokenizer.bos_token}{speaker_token}{msg.text}{tokenizer.eos_token}\n")
    return "".join(rows)


def dump_jsonl(messages: list[ChatMessage], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for msg in messages:
            handle.write(
                json.dumps(
                    {
                        "timestamp": msg.timestamp.isoformat(timespec="minutes"),
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "source_file": msg.source_file,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess KakaoTalk exported text logs.")
    parser.add_argument("--config_paths", default="configs/paths.yaml", help="Paths config yaml.")
    parser.add_argument("--env_path", default=".env", help="Optional .env file path.")
    parser.add_argument("--input_glob", default="", help="Glob for raw chat export files.")
    parser.add_argument("--output_dir", default="", help="Output directory.")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--include_system", action="store_true", help="Keep system messages.")
    parser.add_argument("--disable_dedup", action="store_true", help="Disable duplicate removal.")
    parser.add_argument(
        "--drop_media_only",
        action="store_true",
        help="Drop lines that are only media placeholders like '사진'.",
    )
    parser.add_argument("--mask_urls", action="store_true", help="Mask URLs as <URL>.")
    parser.add_argument("--mask_numbers", action="store_true", help="Mask 3+ digit numbers as <NUM>.")
    parser.add_argument(
        "--shuffle_before_split",
        action="store_true",
        help="Shuffle messages before train/val split instead of chronological split.",
    )
    args = parser.parse_args()

    paths_cfg = load_paths_config(config_path=args.config_paths, env_path=args.env_path)
    processed_cfg = dict(paths_cfg.get("processed", {}))
    input_glob = args.input_glob or str(processed_cfg.get("input_glob", "data/raw/inbox/*.txt"))
    output_dir_value = args.output_dir or str(processed_cfg.get("output_dir", "data/processed"))

    random.seed(args.seed)

    input_paths = sorted(Path(".").glob(input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No files matched input glob: {input_glob}")

    output_dir = Path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_messages: list[ChatMessage] = []
    used_input_files: list[str] = []
    skipped_input_files: dict[str, str] = {}
    parse_errors: dict[str, str] = {}
    for path in tqdm(input_paths, desc="Parsing chat exports"):
        try:
            file_messages = parse_kakao_export(path, include_system=args.include_system)
            if file_messages:
                parsed_messages.extend(file_messages)
                used_input_files.append(path.name)
            else:
                skipped_input_files[path.name] = "no_chat_messages"
        except Exception as exc:  # noqa: BLE001
            parse_errors[path.name] = repr(exc)

    cleaned_messages: list[ChatMessage] = []
    for msg in parsed_messages:
        cleaned = clean_message(
            text=msg.text,
            mask_urls=args.mask_urls,
            mask_numbers=args.mask_numbers,
            drop_media_only=args.drop_media_only,
        )
        if cleaned:
            cleaned_messages.append(
                ChatMessage(
                    timestamp=msg.timestamp,
                    speaker=msg.speaker,
                    text=cleaned,
                    source_file=msg.source_file,
                )
            )

    cleaned_messages.sort(key=lambda item: (item.timestamp, item.source_file))

    if not args.disable_dedup:
        deduped: list[ChatMessage] = []
        seen: set[tuple[str, str, str]] = set()
        for msg in cleaned_messages:
            key = (msg.timestamp.isoformat(timespec="minutes"), msg.speaker, msg.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(msg)
        cleaned_messages = deduped

    if len(cleaned_messages) < 100:
        raise RuntimeError("Too few usable messages after preprocessing.")

    if args.shuffle_before_split:
        random.shuffle(cleaned_messages)

    speakers = [msg.speaker for msg in cleaned_messages]
    speaker_counts = Counter(speakers)
    speaker_to_token = build_speaker_token_map(speaker_counts.keys())
    tokenizer = ByteSpecialTokenizer(speaker_to_token=speaker_to_token)
    tokenizer.save(output_dir / "tokenizer.json")

    split_index = int(len(cleaned_messages) * (1.0 - args.val_ratio))
    split_index = max(1, min(split_index, len(cleaned_messages) - 1))
    train_messages = cleaned_messages[:split_index]
    val_messages = cleaned_messages[split_index:]

    train_text = build_lm_corpus(train_messages, tokenizer)
    val_text = build_lm_corpus(val_messages, tokenizer)
    train_ids = np.asarray(tokenizer.encode(train_text), dtype=np.uint16)
    val_ids = np.asarray(tokenizer.encode(val_text), dtype=np.uint16)

    (output_dir / "train.bin").write_bytes(train_ids.tobytes())
    (output_dir / "val.bin").write_bytes(val_ids.tobytes())
    dump_jsonl(cleaned_messages, output_dir / "messages.jsonl")

    stats = {
        "input_files": used_input_files,
        "skipped_input_files": skipped_input_files,
        "parse_errors": parse_errors,
        "parsed_messages": len(parsed_messages),
        "usable_messages": len(cleaned_messages),
        "train_messages": len(train_messages),
        "val_messages": len(val_messages),
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_tokens,
        "speaker_counts": dict(speaker_counts),
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_preview = [
        {
            "speaker": msg.speaker,
            "timestamp": msg.timestamp.isoformat(timespec="minutes"),
            "text": msg.text,
        }
        for msg in train_messages[:5]
    ]
    (output_dir / "preview.json").write_text(
        json.dumps({"samples": train_preview}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "vocab_size": tokenizer.vocab_size,
                "speaker_count": len(tokenizer.speaker_names),
            },
            ensure_ascii=False,
        )
    )
    print(
        f"Done. Messages={len(cleaned_messages)} train_tokens={train_ids.size} "
        f"val_tokens={val_ids.size} vocab={tokenizer.vocab_size}"
    )


if __name__ == "__main__":
    main()

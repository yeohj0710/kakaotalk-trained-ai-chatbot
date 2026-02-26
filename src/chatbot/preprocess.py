from __future__ import annotations

import argparse
import io
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
DATE_HEADER_PATTERN = re.compile(r"^\d{4}\s*[./년]\s*\d{1,2}\s*[./월]\s*\d{1,2}")
SAVED_AT_PATTERN = re.compile(r"(저장한 날짜|saved\s*date)", re.IGNORECASE)
DELETED_PATTERN = re.compile(r"(메시지가 삭제되었습니다|deleted message)", re.IGNORECASE)
EXPORT_HEADER_PATTERN = re.compile(r"(카카오톡 대화|kakaotalk chats)", re.IGNORECASE)
MEDIA_PATTERN = re.compile(r"^(사진|동영상|음성메시지|파일|이모티콘)(:.*)?$")

LOW_SIGNAL_PUNCT_RE = re.compile(r"^[\s~!?.…·,:;'\"()\[\]{}<>/\\\-_=+*^|`]+$")
LAUGH_CRY_RE = re.compile(r"^[ㅋㅎㅠㅜ]+$")


@dataclass
class ChatMessage:
    timestamp: datetime
    speaker: str
    text: str
    source_file: str


@dataclass
class DialogueSession:
    source_file: str
    session_index: int
    start_ts: datetime
    messages: list[ChatMessage]


@dataclass
class ContextBuildOptions:
    corpus_mode: str = "context_windows"
    context_turns: int = 8
    min_context_turns: int = 2
    sample_stride: int = 1
    session_gap_minutes: int = 180
    merge_same_speaker: bool = True
    merge_gap_minutes: int = 2
    max_merged_chars: int = 320
    min_message_chars: int = 2
    min_target_chars: int = 6
    max_message_chars: int = 320
    drop_low_signal: bool = True
    response_only_loss: bool = True


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

        message_match = MESSAGE_PATTERN.match(line)
        if message_match is not None:
            flush_active()
            active_ts = parse_timestamp(message_match.group("ts"))
            active_speaker = message_match.group("speaker").strip()
            active_lines = [message_match.group("text")]
            continue

        system_match = SYSTEM_PATTERN.match(line)
        if system_match is not None:
            flush_active()
            if include_system:
                messages.append(
                    ChatMessage(
                        timestamp=parse_timestamp(system_match.group("ts")),
                        speaker="[SYSTEM]",
                        text=system_match.group("event").strip(),
                        source_file=path.name,
                    )
                )
            continue

        stripped = line.strip()
        if not stripped:
            if active_lines:
                active_lines.append("")
            continue

        lowered = stripped.lower()
        if SAVED_AT_PATTERN.search(stripped) or DATE_HEADER_PATTERN.match(stripped):
            continue
        if EXPORT_HEADER_PATTERN.search(stripped):
            continue
        if DELETED_PATTERN.search(stripped):
            continue
        if "talk_" in lowered and "txt" in lowered and "kakao" in lowered:
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


def compact_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def is_low_signal_message(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if compact.upper() == "<URL>":
        return True
    if len(compact) <= 8 and LAUGH_CRY_RE.fullmatch(compact):
        return True
    if len(compact) <= 10 and LOW_SIGNAL_PUNCT_RE.fullmatch(compact):
        return True
    if compact in {"ㅇㅇ", "ㄱㄱ", "ㄴㄴ", "ok", "OK"}:
        return True
    return False


def format_row(tokenizer: ByteSpecialTokenizer, msg: ChatMessage) -> str:
    return f"{tokenizer.bos_token}{tokenizer.speaker_token(msg.speaker)}{msg.text}{tokenizer.eos_token}\n"


def encode_row_with_mask(
    tokenizer: ByteSpecialTokenizer,
    msg: ChatMessage,
    response_only_loss: bool,
) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"{tokenizer.bos_token}{tokenizer.speaker_token(msg.speaker)}"
    target = f"{msg.text}{tokenizer.eos_token}\n"
    prefix_ids = tokenizer.encode(prefix)
    target_ids = tokenizer.encode(target)

    token_ids = np.asarray(prefix_ids + target_ids, dtype=np.uint16)
    if response_only_loss:
        mask = np.asarray(([0] * len(prefix_ids)) + ([1] * len(target_ids)), dtype=np.uint8)
    else:
        mask = np.asarray([1] * len(token_ids), dtype=np.uint8)
    return token_ids, mask


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


def merge_consecutive_messages(
    messages: list[ChatMessage],
    merge_gap_minutes: int,
    max_merged_chars: int,
) -> list[ChatMessage]:
    if not messages:
        return []

    grouped: dict[str, list[ChatMessage]] = defaultdict(list)
    for msg in messages:
        grouped[msg.source_file].append(msg)

    merged_all: list[ChatMessage] = []
    for source_file in sorted(grouped.keys()):
        source_messages = sorted(grouped[source_file], key=lambda item: item.timestamp)
        current = source_messages[0]
        for msg in source_messages[1:]:
            gap_min = (msg.timestamp - current.timestamp).total_seconds() / 60.0
            can_merge = (
                msg.speaker == current.speaker
                and 0 <= gap_min <= merge_gap_minutes
                and len(current.text) + len(msg.text) + 1 <= max_merged_chars
            )
            if can_merge:
                current = ChatMessage(
                    timestamp=msg.timestamp,
                    speaker=current.speaker,
                    text=f"{current.text}\n{msg.text}",
                    source_file=current.source_file,
                )
            else:
                merged_all.append(current)
                current = msg
        merged_all.append(current)

    merged_all.sort(key=lambda item: (item.source_file, item.timestamp))
    return merged_all


def split_into_sessions(messages: list[ChatMessage], session_gap_minutes: int) -> list[DialogueSession]:
    grouped: dict[str, list[ChatMessage]] = defaultdict(list)
    for msg in messages:
        grouped[msg.source_file].append(msg)

    sessions: list[DialogueSession] = []
    for source_file in sorted(grouped.keys()):
        source_messages = sorted(grouped[source_file], key=lambda item: item.timestamp)
        if not source_messages:
            continue

        session_index = 0
        current: list[ChatMessage] = [source_messages[0]]
        for msg in source_messages[1:]:
            gap_min = (msg.timestamp - current[-1].timestamp).total_seconds() / 60.0
            if gap_min > session_gap_minutes:
                sessions.append(
                    DialogueSession(
                        source_file=source_file,
                        session_index=session_index,
                        start_ts=current[0].timestamp,
                        messages=current,
                    )
                )
                session_index += 1
                current = [msg]
            else:
                current.append(msg)

        sessions.append(
            DialogueSession(
                source_file=source_file,
                session_index=session_index,
                start_ts=current[0].timestamp,
                messages=current,
            )
        )

    sessions.sort(key=lambda item: (item.start_ts, item.source_file, item.session_index))
    return sessions


def split_train_val_sessions(
    sessions: list[DialogueSession],
    val_ratio: float,
    shuffle_before_split: bool,
    seed: int,
) -> tuple[list[DialogueSession], list[DialogueSession]]:
    if not sessions:
        return [], []

    if len(sessions) == 1:
        only = sessions[0]
        if len(only.messages) < 4:
            return sessions, []
        split_idx = int(len(only.messages) * (1.0 - val_ratio))
        split_idx = max(2, min(split_idx, len(only.messages) - 2))
        left = DialogueSession(
            source_file=only.source_file,
            session_index=only.session_index,
            start_ts=only.messages[0].timestamp,
            messages=only.messages[:split_idx],
        )
        right = DialogueSession(
            source_file=only.source_file,
            session_index=only.session_index + 1,
            start_ts=only.messages[split_idx].timestamp,
            messages=only.messages[split_idx:],
        )
        return [left], [right]

    order = list(sessions)
    if shuffle_before_split:
        rng = random.Random(seed)
        rng.shuffle(order)

    split_idx = int(len(order) * (1.0 - val_ratio))
    split_idx = max(1, min(split_idx, len(order) - 1))
    return order[:split_idx], order[split_idx:]


def write_flat_corpus_bin(
    messages: list[ChatMessage],
    tokenizer: ByteSpecialTokenizer,
    output_path: Path,
    mask_output_path: Path,
    response_only_loss: bool,
    preview_limit: int = 5,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    preview: list[dict[str, str]] = []
    total_tokens = 0
    total_loss_tokens = 0

    with output_path.open("wb") as handle, mask_output_path.open("wb") as mask_handle:
        for msg in messages:
            ids, loss_mask = encode_row_with_mask(
                tokenizer=tokenizer,
                msg=msg,
                response_only_loss=response_only_loss,
            )
            handle.write(ids.tobytes())
            mask_handle.write(loss_mask.tobytes())
            total_tokens += int(ids.size)
            total_loss_tokens += int(loss_mask.sum())
            if len(preview) < preview_limit:
                preview.append(
                    {
                        "speaker": msg.speaker,
                        "timestamp": msg.timestamp.isoformat(timespec="minutes"),
                        "text": msg.text,
                        "source_file": msg.source_file,
                    }
                )

    return {
        "tokens": total_tokens,
        "loss_tokens": total_loss_tokens,
        "loss_token_ratio": round(float(total_loss_tokens / max(1, total_tokens)), 6),
        "examples": len(messages),
        "preview": preview,
        "skipped_short_target": 0,
        "skipped_short_context": 0,
        "avg_context_turns": 0.0,
    }


def write_context_corpus_bin(
    sessions: list[DialogueSession],
    tokenizer: ByteSpecialTokenizer,
    output_path: Path,
    mask_output_path: Path,
    options: ContextBuildOptions,
    preview_limit: int = 5,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_loss_tokens = 0
    examples = 0
    skipped_short_target = 0
    skipped_short_context = 0
    context_turn_total = 0
    preview: list[dict[str, Any]] = []
    stride = max(1, options.sample_stride)

    with output_path.open("wb") as handle, mask_output_path.open("wb") as mask_handle:
        for session in sessions:
            msgs = session.messages
            if len(msgs) < 2:
                continue

            for target_idx in range(1, len(msgs), stride):
                target = msgs[target_idx]
                if compact_length(target.text) < options.min_target_chars:
                    skipped_short_target += 1
                    continue

                start_idx = max(0, target_idx - options.context_turns)
                context = msgs[start_idx:target_idx]
                if len(context) < options.min_context_turns:
                    skipped_short_context += 1
                    continue

                rows = io.StringIO()
                mask_values: list[int] = []
                for item in context:
                    row_text = format_row(tokenizer=tokenizer, msg=item)
                    rows.write(row_text)
                    row_ids = tokenizer.encode(row_text)
                    mask_values.extend([0] * len(row_ids))
                _, target_mask = encode_row_with_mask(
                    tokenizer=tokenizer,
                    msg=target,
                    response_only_loss=options.response_only_loss,
                )
                rows.write(format_row(tokenizer=tokenizer, msg=target))
                sample_text = rows.getvalue()
                sample_ids = tokenizer.encode(sample_text)
                ids = np.asarray(sample_ids, dtype=np.uint16)
                if len(mask_values) + len(target_mask) != len(ids):
                    raise RuntimeError("Loss mask alignment mismatch during preprocessing.")
                loss_mask = np.asarray(mask_values + target_mask.tolist(), dtype=np.uint8)
                handle.write(ids.tobytes())
                mask_handle.write(loss_mask.tobytes())

                total_tokens += int(ids.size)
                total_loss_tokens += int(loss_mask.sum())
                examples += 1
                context_turn_total += len(context)

                if len(preview) < preview_limit:
                    preview.append(
                        {
                            "source_file": session.source_file,
                            "session_index": session.session_index,
                            "context": [
                                {
                                    "speaker": item.speaker,
                                    "text": item.text,
                                }
                                for item in context
                            ],
                            "target": {
                                "speaker": target.speaker,
                                "text": target.text,
                            },
                        }
                    )

    avg_context_turns = float(context_turn_total / examples) if examples > 0 else 0.0
    return {
        "tokens": total_tokens,
        "loss_tokens": total_loss_tokens,
        "loss_token_ratio": round(float(total_loss_tokens / max(1, total_tokens)), 6),
        "examples": examples,
        "preview": preview,
        "skipped_short_target": skipped_short_target,
        "skipped_short_context": skipped_short_context,
        "avg_context_turns": round(avg_context_turns, 3),
    }


def resolve_int(cli_value: int, cfg_value: Any, default: int, minimum: int = 0) -> int:
    if cli_value is not None and cli_value >= 0:
        value = int(cli_value)
    elif cfg_value is not None:
        value = int(cfg_value)
    else:
        value = int(default)
    return max(minimum, value)


def resolve_str(cli_value: str, cfg_value: Any, default: str) -> str:
    if cli_value:
        return str(cli_value)
    if cfg_value is not None:
        return str(cfg_value)
    return default


def resolve_bool(cli_value: bool | None, cfg_value: Any, default: bool) -> bool:
    if cli_value is not None:
        return bool(cli_value)
    if cfg_value is not None:
        return bool(cfg_value)
    return default


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
        help="Drop lines that are only media placeholders.",
    )
    parser.add_argument("--mask_urls", action="store_true", help="Mask URLs as <URL>.")
    parser.add_argument("--mask_numbers", action="store_true", help="Mask 3+ digit numbers as <NUM>.")
    parser.add_argument(
        "--shuffle_before_split",
        action="store_true",
        help="Shuffle before train/val split.",
    )

    parser.add_argument("--corpus_mode", default="", choices=["", "context_windows", "flat"])
    parser.add_argument("--context_turns", type=int, default=-1)
    parser.add_argument("--min_context_turns", type=int, default=-1)
    parser.add_argument("--sample_stride", type=int, default=-1)
    parser.add_argument("--session_gap_minutes", type=int, default=-1)
    parser.add_argument("--merge_gap_minutes", type=int, default=-1)
    parser.add_argument("--max_merged_chars", type=int, default=-1)
    parser.add_argument("--min_message_chars", type=int, default=-1)
    parser.add_argument("--min_target_chars", type=int, default=-1)
    parser.add_argument("--max_message_chars", type=int, default=-1)
    parser.add_argument("--merge_same_speaker", dest="merge_same_speaker", action="store_true", default=None)
    parser.add_argument("--no_merge_same_speaker", dest="merge_same_speaker", action="store_false")
    parser.add_argument("--drop_low_signal", dest="drop_low_signal", action="store_true", default=None)
    parser.add_argument("--keep_low_signal", dest="drop_low_signal", action="store_false")
    parser.add_argument("--response_only_loss", dest="response_only_loss", action="store_true", default=None)
    parser.add_argument("--full_sequence_loss", dest="response_only_loss", action="store_false")

    args = parser.parse_args()

    paths_cfg = load_paths_config(config_path=args.config_paths, env_path=args.env_path)
    processed_cfg = dict(paths_cfg.get("processed", {}))
    preprocess_cfg = dict(processed_cfg.get("preprocess", {}))

    input_glob = args.input_glob or str(processed_cfg.get("input_glob", "data/raw/inbox/*.txt"))
    output_dir_value = args.output_dir or str(processed_cfg.get("output_dir", "data/processed"))

    context_options = ContextBuildOptions(
        corpus_mode=resolve_str(args.corpus_mode, preprocess_cfg.get("corpus_mode"), "context_windows").lower(),
        context_turns=resolve_int(args.context_turns, preprocess_cfg.get("context_turns"), 8, minimum=1),
        min_context_turns=resolve_int(args.min_context_turns, preprocess_cfg.get("min_context_turns"), 2, minimum=1),
        sample_stride=resolve_int(args.sample_stride, preprocess_cfg.get("sample_stride"), 1, minimum=1),
        session_gap_minutes=resolve_int(
            args.session_gap_minutes, preprocess_cfg.get("session_gap_minutes"), 180, minimum=1
        ),
        merge_same_speaker=resolve_bool(args.merge_same_speaker, preprocess_cfg.get("merge_same_speaker"), True),
        merge_gap_minutes=resolve_int(args.merge_gap_minutes, preprocess_cfg.get("merge_gap_minutes"), 2, minimum=1),
        max_merged_chars=resolve_int(args.max_merged_chars, preprocess_cfg.get("max_merged_chars"), 320, minimum=32),
        min_message_chars=resolve_int(args.min_message_chars, preprocess_cfg.get("min_message_chars"), 2, minimum=1),
        min_target_chars=resolve_int(args.min_target_chars, preprocess_cfg.get("min_target_chars"), 6, minimum=1),
        max_message_chars=resolve_int(args.max_message_chars, preprocess_cfg.get("max_message_chars"), 320, minimum=32),
        drop_low_signal=resolve_bool(args.drop_low_signal, preprocess_cfg.get("drop_low_signal"), True),
        response_only_loss=resolve_bool(args.response_only_loss, preprocess_cfg.get("response_only_loss"), True),
    )
    if context_options.corpus_mode not in {"context_windows", "flat"}:
        raise ValueError("corpus_mode must be one of: context_windows, flat")

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
    drop_reasons = Counter()
    for msg in parsed_messages:
        cleaned = clean_message(
            text=msg.text,
            mask_urls=args.mask_urls,
            mask_numbers=args.mask_numbers,
            drop_media_only=args.drop_media_only,
        )
        if not cleaned:
            drop_reasons["empty_after_clean"] += 1
            continue

        if len(cleaned) > context_options.max_message_chars:
            cleaned = cleaned[: context_options.max_message_chars].rstrip()

        if compact_length(cleaned) < context_options.min_message_chars:
            drop_reasons["short_message"] += 1
            continue

        if context_options.drop_low_signal and is_low_signal_message(cleaned):
            drop_reasons["low_signal"] += 1
            continue

        cleaned_messages.append(
            ChatMessage(
                timestamp=msg.timestamp,
                speaker=msg.speaker,
                text=cleaned,
                source_file=msg.source_file,
            )
        )

    cleaned_messages.sort(key=lambda item: (item.source_file, item.timestamp))

    if not args.disable_dedup:
        deduped: list[ChatMessage] = []
        seen: set[tuple[str, str, str, str]] = set()
        for msg in cleaned_messages:
            key = (
                msg.source_file,
                msg.timestamp.isoformat(timespec="minutes"),
                msg.speaker,
                msg.text,
            )
            if key in seen:
                drop_reasons["deduplicated"] += 1
                continue
            seen.add(key)
            deduped.append(msg)
        cleaned_messages = deduped

    if context_options.merge_same_speaker:
        before_merge = len(cleaned_messages)
        cleaned_messages = merge_consecutive_messages(
            messages=cleaned_messages,
            merge_gap_minutes=context_options.merge_gap_minutes,
            max_merged_chars=context_options.max_merged_chars,
        )
        drop_reasons["merged_into_previous"] += max(0, before_merge - len(cleaned_messages))

    if len(cleaned_messages) < 200:
        raise RuntimeError("Too few usable messages after preprocessing.")

    sessions = split_into_sessions(messages=cleaned_messages, session_gap_minutes=context_options.session_gap_minutes)
    train_sessions, val_sessions = split_train_val_sessions(
        sessions=sessions,
        val_ratio=args.val_ratio,
        shuffle_before_split=args.shuffle_before_split,
        seed=args.seed,
    )

    if not train_sessions or not val_sessions:
        raise RuntimeError(
            "Failed to split train/val sessions. Increase data or adjust val_ratio/session_gap_minutes."
        )

    speakers = [msg.speaker for msg in cleaned_messages]
    speaker_counts = Counter(speakers)
    speaker_to_token = build_speaker_token_map(speaker_counts.keys())
    tokenizer = ByteSpecialTokenizer(speaker_to_token=speaker_to_token)
    tokenizer.save(output_dir / "tokenizer.json")

    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "val.bin"
    train_loss_mask_bin = output_dir / "train_loss_mask.bin"
    val_loss_mask_bin = output_dir / "val_loss_mask.bin"

    if context_options.corpus_mode == "context_windows":
        train_stats = write_context_corpus_bin(
            sessions=train_sessions,
            tokenizer=tokenizer,
            output_path=train_bin,
            mask_output_path=train_loss_mask_bin,
            options=context_options,
            preview_limit=5,
        )
        val_stats = write_context_corpus_bin(
            sessions=val_sessions,
            tokenizer=tokenizer,
            output_path=val_bin,
            mask_output_path=val_loss_mask_bin,
            options=context_options,
            preview_limit=2,
        )
    else:
        train_messages = [msg for session in train_sessions for msg in session.messages]
        val_messages = [msg for session in val_sessions for msg in session.messages]
        train_stats = write_flat_corpus_bin(
            messages=train_messages,
            tokenizer=tokenizer,
            output_path=train_bin,
            mask_output_path=train_loss_mask_bin,
            response_only_loss=context_options.response_only_loss,
            preview_limit=5,
        )
        val_stats = write_flat_corpus_bin(
            messages=val_messages,
            tokenizer=tokenizer,
            output_path=val_bin,
            mask_output_path=val_loss_mask_bin,
            response_only_loss=context_options.response_only_loss,
            preview_limit=2,
        )

    if train_stats["tokens"] < 1000 or val_stats["tokens"] < 200:
        raise RuntimeError(
            "Generated corpus is too small. Relax filters (min_target_chars/drop_low_signal) or add more data."
        )

    dump_jsonl(cleaned_messages, output_dir / "messages.jsonl")

    stats = {
        "input_files": used_input_files,
        "skipped_input_files": skipped_input_files,
        "parse_errors": parse_errors,
        "parsed_messages": len(parsed_messages),
        "usable_messages": len(cleaned_messages),
        "session_count": len(sessions),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "train_examples": int(train_stats["examples"]),
        "val_examples": int(val_stats["examples"]),
        "train_tokens": int(train_stats["tokens"]),
        "val_tokens": int(val_stats["tokens"]),
        "train_loss_tokens": int(train_stats.get("loss_tokens", 0)),
        "val_loss_tokens": int(val_stats.get("loss_tokens", 0)),
        "train_loss_token_ratio": float(train_stats.get("loss_token_ratio", 0.0)),
        "val_loss_token_ratio": float(val_stats.get("loss_token_ratio", 0.0)),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_tokens,
        "speaker_counts": dict(speaker_counts),
        "drop_reasons": dict(drop_reasons),
        "preprocess_options": {
            "corpus_mode": context_options.corpus_mode,
            "context_turns": context_options.context_turns,
            "min_context_turns": context_options.min_context_turns,
            "sample_stride": context_options.sample_stride,
            "session_gap_minutes": context_options.session_gap_minutes,
            "merge_same_speaker": context_options.merge_same_speaker,
            "merge_gap_minutes": context_options.merge_gap_minutes,
            "max_merged_chars": context_options.max_merged_chars,
            "min_message_chars": context_options.min_message_chars,
            "min_target_chars": context_options.min_target_chars,
            "max_message_chars": context_options.max_message_chars,
            "drop_low_signal": context_options.drop_low_signal,
            "response_only_loss": context_options.response_only_loss,
            "shuffle_before_split": args.shuffle_before_split,
            "val_ratio": args.val_ratio,
        },
        "context_metrics": {
            "train_avg_context_turns": train_stats.get("avg_context_turns", 0.0),
            "val_avg_context_turns": val_stats.get("avg_context_turns", 0.0),
            "train_skipped_short_target": train_stats.get("skipped_short_target", 0),
            "val_skipped_short_target": val_stats.get("skipped_short_target", 0),
            "train_skipped_short_context": train_stats.get("skipped_short_context", 0),
            "val_skipped_short_context": val_stats.get("skipped_short_context", 0),
        },
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview = {
        "mode": context_options.corpus_mode,
        "samples": train_stats.get("preview", []),
    }
    (output_dir / "preview.json").write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "vocab_size": tokenizer.vocab_size,
                "speaker_count": len(tokenizer.speaker_names),
                "mode": context_options.corpus_mode,
                "train_examples": int(train_stats["examples"]),
                "train_tokens": int(train_stats["tokens"]),
                "train_loss_token_ratio": float(train_stats.get("loss_token_ratio", 0.0)),
                "train_loss_mask_bin": str(train_loss_mask_bin.as_posix()),
            },
            ensure_ascii=False,
        )
    )
    print(
        f"Done. mode={context_options.corpus_mode} usable_messages={len(cleaned_messages)} "
        f"train_tokens={train_stats['tokens']} val_tokens={val_stats['tokens']} vocab={tokenizer.vocab_size}"
    )


if __name__ == "__main__":
    main()

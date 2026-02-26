from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .preprocess import (
    ChatMessage,
    clean_message,
    compact_length,
    is_low_signal_message,
    merge_consecutive_messages,
    parse_kakao_export,
    split_into_sessions,
    split_train_val_sessions,
)
from .sft_config import format_with_run_name, load_sft_config


def one_line(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def build_prompt(
    context: list[ChatMessage],
    system_prompt: str,
    task_prompt: str,
) -> str:
    lines = [f"{msg.speaker}: {one_line(msg.text)}" for msg in context]
    context_text = "\n".join(lines)
    return (
        f"[SYSTEM]\n{system_prompt.strip()}\n\n"
        f"[TASK]\n{task_prompt.strip()}\n\n"
        f"[DIALOGUE]\n{context_text}\n\n"
        f"[ANSWER]\n"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset from KakaoTalk exports.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    project_cfg = dict(cfg.get("project", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    data_cfg = dict(cfg.get("data", {}))
    prompt_cfg = dict(cfg.get("prompt", {}))

    run_name = str(project_cfg.get("run_name", "room_lora_qwen3b")).strip() or "room_lora_qwen3b"
    seed = int(project_cfg.get("seed", 42))
    random.seed(seed)

    raw_glob = str(paths_cfg.get("raw_glob", "data/raw/inbox/*.txt"))
    output_dir = Path(format_with_run_name(str(paths_cfg.get("output_dir", "data/sft")), run_name=run_name))
    train_jsonl = Path(format_with_run_name(str(paths_cfg.get("train_jsonl", output_dir / "train.jsonl")), run_name))
    val_jsonl = Path(format_with_run_name(str(paths_cfg.get("val_jsonl", output_dir / "val.jsonl")), run_name))
    preview_json = Path(format_with_run_name(str(paths_cfg.get("preview_json", output_dir / "preview.json")), run_name))
    stats_json = Path(format_with_run_name(str(paths_cfg.get("stats_json", output_dir / "stats.json")), run_name))

    include_system = bool(data_cfg.get("include_system", False))
    shuffle_before_split = bool(data_cfg.get("shuffle_before_split", False))
    val_ratio = float(data_cfg.get("val_ratio", 0.02))
    session_gap_minutes = int(data_cfg.get("session_gap_minutes", 180))
    context_turns = int(data_cfg.get("context_turns", 18))
    min_context_turns = int(data_cfg.get("min_context_turns", 3))
    sample_stride = max(1, int(data_cfg.get("sample_stride", 2)))
    merge_same_speaker = bool(data_cfg.get("merge_same_speaker", True))
    merge_gap_minutes = int(data_cfg.get("merge_gap_minutes", 3))
    max_merged_chars = int(data_cfg.get("max_merged_chars", 420))
    min_message_chars = int(data_cfg.get("min_message_chars", 2))
    min_target_chars = int(data_cfg.get("min_target_chars", 10))
    max_message_chars = int(data_cfg.get("max_message_chars", 420))
    drop_low_signal = bool(data_cfg.get("drop_low_signal", True))
    mask_urls = bool(data_cfg.get("mask_urls", True))
    mask_numbers = bool(data_cfg.get("mask_numbers", False))
    drop_media_only = bool(data_cfg.get("drop_media_only", True))
    max_examples_per_split = int(data_cfg.get("max_examples_per_split", 0))
    response_one_line = bool(prompt_cfg.get("response_one_line", True))

    system_prompt = str(prompt_cfg.get("system", "")).strip()
    task_prompt = str(prompt_cfg.get("task", "")).strip()
    if not system_prompt or not task_prompt:
        raise ValueError("prompt.system and prompt.task must be configured.")

    raw_files = sorted(Path(".").glob(raw_glob))
    if not raw_files:
        raise FileNotFoundError(f"No files matched raw_glob: {raw_glob}")

    parsed_messages: list[ChatMessage] = []
    parse_errors: dict[str, str] = {}
    used_files: list[str] = []
    for path in tqdm(raw_files, desc="Parsing raw chats"):
        try:
            rows = parse_kakao_export(path, include_system=include_system)
            if rows:
                parsed_messages.extend(rows)
                used_files.append(path.name)
        except Exception as exc:  # noqa: BLE001
            parse_errors[path.name] = repr(exc)

    cleaned_messages: list[ChatMessage] = []
    drop_reasons = Counter()
    for msg in parsed_messages:
        text = clean_message(
            text=msg.text,
            mask_urls=mask_urls,
            mask_numbers=mask_numbers,
            drop_media_only=drop_media_only,
        )
        if not text:
            drop_reasons["empty_after_clean"] += 1
            continue

        if len(text) > max_message_chars:
            text = text[:max_message_chars].rstrip()

        if compact_length(text) < min_message_chars:
            drop_reasons["short_message"] += 1
            continue
        if drop_low_signal and is_low_signal_message(text):
            drop_reasons["low_signal"] += 1
            continue

        cleaned_messages.append(
            ChatMessage(
                timestamp=msg.timestamp,
                speaker=msg.speaker,
                text=text,
                source_file=msg.source_file,
            )
        )

    cleaned_messages.sort(key=lambda item: (item.source_file, item.timestamp))

    deduped: list[ChatMessage] = []
    seen: set[tuple[str, str, str, str]] = set()
    for msg in cleaned_messages:
        key = (msg.source_file, msg.timestamp.isoformat(timespec="minutes"), msg.speaker, msg.text)
        if key in seen:
            drop_reasons["deduplicated"] += 1
            continue
        seen.add(key)
        deduped.append(msg)
    cleaned_messages = deduped

    if merge_same_speaker:
        before_merge = len(cleaned_messages)
        cleaned_messages = merge_consecutive_messages(
            messages=cleaned_messages,
            merge_gap_minutes=merge_gap_minutes,
            max_merged_chars=max_merged_chars,
        )
        drop_reasons["merged_into_previous"] += max(0, before_merge - len(cleaned_messages))

    if len(cleaned_messages) < 200:
        raise RuntimeError("Too few usable messages after cleaning.")

    sessions = split_into_sessions(cleaned_messages, session_gap_minutes=session_gap_minutes)
    train_sessions, val_sessions = split_train_val_sessions(
        sessions=sessions,
        val_ratio=val_ratio,
        shuffle_before_split=shuffle_before_split,
        seed=seed,
    )
    if not train_sessions or not val_sessions:
        raise RuntimeError("Failed to split train/val sessions. Adjust val_ratio or session settings.")

    def build_rows(split_sessions: list[Any], split_name: str) -> tuple[list[dict[str, Any]], Counter]:
        rows: list[dict[str, Any]] = []
        local_drop = Counter()
        for session in split_sessions:
            msgs = session.messages
            if len(msgs) < 2:
                local_drop["short_session"] += 1
                continue
            for target_idx in range(1, len(msgs), sample_stride):
                target = msgs[target_idx]
                response = one_line(target.text) if response_one_line else target.text.strip()
                if compact_length(response) < min_target_chars:
                    local_drop["short_target"] += 1
                    continue
                start_idx = max(0, target_idx - context_turns)
                context = msgs[start_idx:target_idx]
                if len(context) < min_context_turns:
                    local_drop["short_context"] += 1
                    continue
                prompt = build_prompt(context=context, system_prompt=system_prompt, task_prompt=task_prompt)
                rows.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "meta": {
                            "split": split_name,
                            "source_file": target.source_file,
                            "target_speaker": target.speaker,
                            "target_timestamp": target.timestamp.isoformat(timespec="minutes"),
                            "context_turns": len(context),
                        },
                    }
                )
                if max_examples_per_split > 0 and len(rows) >= max_examples_per_split:
                    return rows, local_drop
        return rows, local_drop

    train_rows, train_drop = build_rows(train_sessions, "train")
    val_rows, val_drop = build_rows(val_sessions, "val")
    if not train_rows or not val_rows:
        raise RuntimeError("No train/val examples generated. Relax filters.")

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)

    preview_payload = {
        "run_name": run_name,
        "train_samples": train_rows[:3],
        "val_samples": val_rows[:2],
    }
    preview_json.parent.mkdir(parents=True, exist_ok=True)
    preview_json.write_text(json.dumps(preview_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_payload = {
        "run_name": run_name,
        "raw_glob": raw_glob,
        "used_files": used_files,
        "parse_errors": parse_errors,
        "parsed_messages": len(parsed_messages),
        "usable_messages": len(cleaned_messages),
        "session_count": len(sessions),
        "train_sessions": len(train_sessions),
        "val_sessions": len(val_sessions),
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "drop_reasons_global": dict(drop_reasons),
        "drop_reasons_train": dict(train_drop),
        "drop_reasons_val": dict(val_drop),
        "data_config": data_cfg,
    }
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "sft_preprocess_done",
                "run_name": run_name,
                "train_jsonl": str(train_jsonl.as_posix()),
                "val_jsonl": str(val_jsonl.as_posix()),
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

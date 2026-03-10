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


MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-가-힣]+")
SUMMARY_BULLET_RE = re.compile(r"(?m)^\s*[·•\-]\s+")
SUMMARY_KEYWORD_RE = re.compile(r"(요약|정리|핵심|summary)", re.IGNORECASE)
REPLY_PROMPT_RE = re.compile(
    r"(\?$|[??]|"
    "(뭐|무슨|왜|어때|어떰|어디|언제|누가|누구|몇|어떻게|가능|괜찮|맞지|맞냐|되나|됨|돼|해도|할까|줄래|해줘|알려줘|말해줘|추천))",
    re.IGNORECASE,
)
FILLER_START_RE = re.compile(r"^(그냥|근데|아니|일단|약간|근데도|아무튼)\b")


def one_line(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def count_mentions(text: str) -> int:
    return len(MENTION_RE.findall(text or ""))


def strip_mentions(text: str) -> str:
    out = MENTION_RE.sub("", text or "")
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s*([,;:])\s*", r"\1 ", out)
    return out.strip(" \t,;:")


def is_summary_artifact_message(text: str, bullet_min_count: int = 1) -> bool:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return False
    bullet_count = len(SUMMARY_BULLET_RE.findall(normalized))
    if bullet_count >= max(1, bullet_min_count):
        return True
    if SUMMARY_KEYWORD_RE.search(normalized) and bullet_count > 0:
        return True
    return False


def looks_like_reply_prompt(text: str) -> bool:
    normalized = one_line(text or "")
    if not normalized:
        return False
    return bool(REPLY_PROMPT_RE.search(normalized))


def starts_with_filler_phrase(text: str) -> bool:
    normalized = one_line(text or "")
    if not normalized:
        return False
    return bool(FILLER_START_RE.match(normalized))


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
        "[ANSWER]\n"
    )


def build_projected_prompt(
    context: list[ChatMessage],
    system_prompt: str,
    task_prompt: str,
    bot_speakers: set[str],
    project_user_names: bool,
) -> str:
    lines: list[str] = []
    for msg in context:
        text = one_line(msg.text)
        if msg.speaker in bot_speakers:
            lines.append(f"bot: {text}")
            continue
        if project_user_names:
            lines.append(f"user: [{msg.speaker}] {text}")
        else:
            lines.append(f"user: {text}")
    context_text = "\n".join(lines)
    return (
        f"[SYSTEM]\n{system_prompt.strip()}\n\n"
        f"[TASK]\n{task_prompt.strip()}\n\n"
        f"[DIALOGUE]\n{context_text}\n\n"
        "[ANSWER]\n"
    )


def build_projected_messages(
    context: list[ChatMessage],
    system_prompt: str,
    task_prompt: str,
    bot_speakers: set[str],
    project_user_names: bool,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": f"{system_prompt.strip()}\n\n{task_prompt.strip()}".strip(),
        }
    ]
    for msg in context:
        text = one_line(msg.text)
        if msg.speaker in bot_speakers:
            messages.append({"role": "assistant", "content": text})
        elif project_user_names:
            messages.append({"role": "user", "content": f"[{msg.speaker}] {text}"})
        else:
            messages.append({"role": "user", "content": text})
    return messages


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT/CPT datasets from KakaoTalk exports.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    project_cfg = dict(cfg.get("project", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    data_cfg = dict(cfg.get("data", {}))
    prompt_cfg = dict(cfg.get("prompt", {}))

    run_name = str(project_cfg.get("run_name", "room_lora_qwen25_7b_group_v2")).strip() or "room_lora_qwen25_7b_group_v2"
    seed = int(project_cfg.get("seed", 42))
    random.seed(seed)

    raw_glob = str(paths_cfg.get("raw_glob", "data/raw/inbox/*.txt"))
    output_dir = Path(format_with_run_name(str(paths_cfg.get("output_dir", "data/sft")), run_name=run_name))
    train_jsonl = Path(format_with_run_name(str(paths_cfg.get("train_jsonl", output_dir / "train.jsonl")), run_name))
    val_jsonl = Path(format_with_run_name(str(paths_cfg.get("val_jsonl", output_dir / "val.jsonl")), run_name))
    cpt_train_jsonl = Path(format_with_run_name(str(paths_cfg.get("cpt_train_jsonl", output_dir / "cpt_train.jsonl")), run_name))
    cpt_val_jsonl = Path(format_with_run_name(str(paths_cfg.get("cpt_val_jsonl", output_dir / "cpt_val.jsonl")), run_name))
    preview_json = Path(format_with_run_name(str(paths_cfg.get("preview_json", output_dir / "preview.json")), run_name))
    stats_json = Path(format_with_run_name(str(paths_cfg.get("stats_json", output_dir / "stats.json")), run_name))

    include_system = bool(data_cfg.get("include_system", False))
    shuffle_before_split = bool(data_cfg.get("shuffle_before_split", False))
    val_ratio = float(data_cfg.get("val_ratio", 0.02))
    session_gap_minutes = int(data_cfg.get("session_gap_minutes", 180))
    context_turns = max(1, int(data_cfg.get("context_turns", 8)))
    min_context_turns = max(1, int(data_cfg.get("min_context_turns", 2)))
    sample_stride = max(1, int(data_cfg.get("sample_stride", 1)))
    merge_same_speaker = bool(data_cfg.get("merge_same_speaker", True))
    merge_gap_minutes = int(data_cfg.get("merge_gap_minutes", 2))
    max_merged_chars = int(data_cfg.get("max_merged_chars", 320))
    min_message_chars = int(data_cfg.get("min_message_chars", 2))
    min_target_chars = int(data_cfg.get("min_target_chars", 8))
    max_target_chars = int(data_cfg.get("max_target_chars", 0))
    max_message_chars = int(data_cfg.get("max_message_chars", 320))
    drop_low_signal = bool(data_cfg.get("drop_low_signal", True))
    mask_urls = bool(data_cfg.get("mask_urls", True))
    mask_numbers = bool(data_cfg.get("mask_numbers", False))
    drop_media_only = bool(data_cfg.get("drop_media_only", True))
    drop_summary_artifacts = bool(data_cfg.get("drop_summary_artifacts", True))
    summary_bullet_min_count = max(1, int(data_cfg.get("summary_bullet_min_count", 1)))
    drop_mention_messages = bool(data_cfg.get("drop_mention_messages", True))
    max_mentions_per_message = max(0, int(data_cfg.get("max_mentions_per_message", 0)))
    strip_mentions_before_filter = bool(data_cfg.get("strip_mentions_before_filter", False))
    max_examples_per_split = int(data_cfg.get("max_examples_per_split", 0))
    training_mode = str(data_cfg.get("training_mode", "next_turn_all_speakers")).strip().lower()
    target_speakers = {str(item).strip() for item in data_cfg.get("target_speakers", []) if str(item).strip()}
    project_user_names = bool(data_cfg.get("project_user_names", False))
    min_other_speaker_turns = max(0, int(data_cfg.get("min_other_speaker_turns", 0)))
    require_last_context_from_other_speaker = bool(data_cfg.get("require_last_context_from_other_speaker", False))
    require_last_context_reply_prompt = bool(data_cfg.get("require_last_context_reply_prompt", False))
    drop_target_reply_prompt = bool(data_cfg.get("drop_target_reply_prompt", False))
    drop_target_multiline = bool(data_cfg.get("drop_target_multiline", False))
    drop_target_with_url = bool(data_cfg.get("drop_target_with_url", False))
    drop_target_filler_start = bool(data_cfg.get("drop_target_filler_start", False))
    response_one_line = bool(prompt_cfg.get("response_one_line", True))
    if training_mode not in {"next_turn_all_speakers", "projected_dialogue"}:
        raise ValueError(f"Unsupported data.training_mode: {training_mode}")
    if training_mode == "projected_dialogue" and not target_speakers:
        print(
            json.dumps(
                {
                    "event": "projected_dialogue_warning",
                    "reason": "target_speakers_empty",
                    "hint": "Set data.target_speakers (or CHATBOT_PERSONA_SPEAKERS) for a stable single-persona bot.",
                },
                ensure_ascii=False,
            )
        )

    cpt_cfg = dict(cfg.get("cpt_data", {}))
    cpt_window_messages = max(2, int(cpt_cfg.get("window_messages", 64)))
    cpt_stride_messages = max(1, int(cpt_cfg.get("stride_messages", 16)))
    cpt_min_messages = max(2, int(cpt_cfg.get("min_messages", 10)))
    cpt_min_chars = max(10, int(cpt_cfg.get("min_chars", 120)))
    cpt_max_chars = max(50, int(cpt_cfg.get("max_chars", 2200)))
    cpt_use_speaker_prefix = bool(cpt_cfg.get("use_speaker_prefix", True))

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

        mention_count = count_mentions(text)
        if strip_mentions_before_filter and mention_count > 0:
            text = strip_mentions(text)
            mention_count = count_mentions(text)

        if compact_length(text) < min_message_chars:
            drop_reasons["short_message"] += 1
            continue
        if drop_low_signal and is_low_signal_message(text):
            drop_reasons["low_signal"] += 1
            continue
        if drop_summary_artifacts and is_summary_artifact_message(text, bullet_min_count=summary_bullet_min_count):
            drop_reasons["summary_artifact"] += 1
            continue
        if drop_mention_messages and mention_count > max_mentions_per_message:
            drop_reasons["mention_artifact"] += 1
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
                if max_target_chars > 0 and compact_length(response) > max_target_chars:
                    local_drop["long_target"] += 1
                    continue
                if drop_target_multiline and "\n" in target.text:
                    local_drop["multiline_target"] += 1
                    continue
                if drop_target_with_url and "<URL>" in response:
                    local_drop["target_url"] += 1
                    continue
                if drop_target_reply_prompt and looks_like_reply_prompt(response):
                    local_drop["target_reply_prompt"] += 1
                    continue
                if drop_target_filler_start and starts_with_filler_phrase(response):
                    local_drop["target_filler_start"] += 1
                    continue
                start_idx = max(0, target_idx - context_turns)
                context = msgs[start_idx:target_idx]
                if len(context) < min_context_turns:
                    local_drop["short_context"] += 1
                    continue
                if training_mode == "projected_dialogue":
                    if target_speakers and target.speaker not in target_speakers:
                        local_drop["target_speaker_filtered"] += 1
                        continue
                    bot_speakers = target_speakers if target_speakers else {target.speaker}
                    other_speaker_turns = sum(1 for item in context if item.speaker not in bot_speakers)
                    if other_speaker_turns < min_other_speaker_turns:
                        local_drop["other_speaker_context"] += 1
                        continue
                    if require_last_context_from_other_speaker and context[-1].speaker in bot_speakers:
                        local_drop["last_context_is_bot"] += 1
                        continue
                    if require_last_context_reply_prompt and not looks_like_reply_prompt(context[-1].text):
                        local_drop["last_context_not_reply_prompt"] += 1
                        continue
                    prompt = build_projected_prompt(
                        context=context,
                        system_prompt=system_prompt,
                        task_prompt=task_prompt,
                        bot_speakers=bot_speakers,
                        project_user_names=project_user_names,
                    )
                    messages = build_projected_messages(
                        context=context,
                        system_prompt=system_prompt,
                        task_prompt=task_prompt,
                        bot_speakers=bot_speakers,
                        project_user_names=project_user_names,
                    )
                else:
                    prompt = build_prompt(context=context, system_prompt=system_prompt, task_prompt=task_prompt)
                    messages = []
                rows.append(
                    {
                        "prompt": prompt,
                        "response": response,
                        "messages": messages,
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

    def build_cpt_rows(split_sessions: list[Any], split_name: str) -> tuple[list[dict[str, Any]], Counter]:
        rows: list[dict[str, Any]] = []
        local_drop = Counter()
        for session in split_sessions:
            msgs = session.messages
            if len(msgs) < cpt_min_messages:
                local_drop["short_session"] += 1
                continue

            for start_idx in range(0, len(msgs), cpt_stride_messages):
                chunk = msgs[start_idx : start_idx + cpt_window_messages]
                if len(chunk) < cpt_min_messages:
                    continue
                if cpt_use_speaker_prefix:
                    text = "\n".join(f"{msg.speaker}: {one_line(msg.text)}" for msg in chunk)
                else:
                    text = "\n".join(one_line(msg.text) for msg in chunk)
                text = text.strip()
                if len(text) > cpt_max_chars:
                    text = text[:cpt_max_chars].rstrip()
                if compact_length(text) < cpt_min_chars:
                    local_drop["short_chunk"] += 1
                    continue
                rows.append(
                    {
                        "text": text,
                        "meta": {
                            "split": split_name,
                            "source_file": chunk[-1].source_file,
                            "start_timestamp": chunk[0].timestamp.isoformat(timespec="minutes"),
                            "end_timestamp": chunk[-1].timestamp.isoformat(timespec="minutes"),
                            "message_count": len(chunk),
                        },
                    }
                )
                if max_examples_per_split > 0 and len(rows) >= max_examples_per_split:
                    return rows, local_drop
        return rows, local_drop

    cpt_train_rows, cpt_train_drop = build_cpt_rows(train_sessions, "train")
    cpt_val_rows, cpt_val_drop = build_cpt_rows(val_sessions, "val")
    if not cpt_train_rows or not cpt_val_rows:
        raise RuntimeError("No CPT train/val examples generated. Relax cpt_data filters.")

    write_jsonl(train_jsonl, train_rows)
    write_jsonl(val_jsonl, val_rows)
    write_jsonl(cpt_train_jsonl, cpt_train_rows)
    write_jsonl(cpt_val_jsonl, cpt_val_rows)

    preview_payload = {
        "run_name": run_name,
        "train_samples": train_rows[:3],
        "val_samples": val_rows[:2],
        "cpt_train_samples": cpt_train_rows[:2],
        "cpt_val_samples": cpt_val_rows[:1],
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
        "cpt_train_examples": len(cpt_train_rows),
        "cpt_val_examples": len(cpt_val_rows),
        "drop_reasons_global": dict(drop_reasons),
        "drop_reasons_train": dict(train_drop),
        "drop_reasons_val": dict(val_drop),
        "drop_reasons_cpt_train": dict(cpt_train_drop),
        "drop_reasons_cpt_val": dict(cpt_val_drop),
        "data_config": data_cfg,
        "cpt_data_config": cpt_cfg,
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
                "cpt_train_jsonl": str(cpt_train_jsonl.as_posix()),
                "cpt_val_jsonl": str(cpt_val_jsonl.as_posix()),
                "train_examples": len(train_rows),
                "val_examples": len(val_rows),
                "cpt_train_examples": len(cpt_train_rows),
                "cpt_val_examples": len(cpt_val_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

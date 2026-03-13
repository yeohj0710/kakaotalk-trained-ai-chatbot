from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


QUESTION_RE = re.compile(
    r"(\?$|[?？]|"
    r"(왜|뭐|무슨|몇|어디|언제|어때|어떻게|가능|괜찮|맞[아나음는]?|되나|되는|해야|가야|먹을까|볼까|할까|임\??|냐\??|인가|인가요|지\??))",
    re.IGNORECASE,
)
PROFANITY_RE = re.compile(r"(씨발|시발|병신|개새|좆|fuck|shit|bitch)", re.IGNORECASE)
FILLER_START_RE = re.compile(r"^(아니\s*근데|근데|그니까|그러니까|아니\b|일단\b|음\b|어\b)", re.IGNORECASE)
RESTART_RE = re.compile(r"\b(근데|그런데|그러니까|그래서|아니면|다만)\b")
LAUGH_RE = re.compile(r"(ㅋ{2,}|ㅎ{2,})")


def _one_line(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {lineno} must be an object.")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _last_user_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("role", "")).strip() != "user":
            continue
        return _one_line(str(item.get("content", "")))
    return ""


def _question_like(text: str) -> bool:
    normalized = _one_line(text)
    if not normalized:
        return False
    return bool(QUESTION_RE.search(normalized))


def _safe_direct_reply(text: str, min_chars: int, max_chars: int) -> bool:
    out = _one_line(text)
    if len(out) < min_chars or len(out) > max_chars:
        return False
    if "\n" in text:
        return False
    if PROFANITY_RE.search(out):
        return False
    if "<URL>" in out or "http://" in out or "https://" in out:
        return False
    if out.endswith("?"):
        return False
    if LAUGH_RE.search(out):
        return False
    if FILLER_START_RE.search(out):
        return False
    if len(RESTART_RE.findall(out)) >= 2:
        return False
    return True


def _keyword_overlap(user_text: str, reply_text: str) -> bool:
    user_tokens = {tok for tok in re.split(r"[^0-9A-Za-z가-힣]+", user_text) if len(tok) >= 2}
    reply_tokens = {tok for tok in re.split(r"[^0-9A-Za-z가-힣]+", reply_text) if len(tok) >= 2}
    if not user_tokens or not reply_tokens:
        return True
    return len(user_tokens & reply_tokens) >= 1


def _build_row(row: dict[str, Any], source_split: str) -> dict[str, Any]:
    meta = dict(row.get("meta", {}))
    meta["source"] = "auto_direct_qa"
    meta["source_split"] = source_split
    return {
        "prompt": row.get("prompt", ""),
        "response": _one_line(str(row.get("response", ""))),
        "messages": row.get("messages", []),
        "meta": meta,
    }


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("prompt", "")), str(row.get("response", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _mine_rows(
    rows: list[dict[str, Any]],
    source_split: str,
    min_chars: int,
    max_chars: int,
    require_overlap: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out: list[dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "drop_no_user": 0,
        "drop_not_question": 0,
        "drop_reply_quality": 0,
        "drop_overlap": 0,
    }
    for row in rows:
        user_text = _last_user_text(row)
        if not user_text:
            stats["drop_no_user"] += 1
            continue
        if not _question_like(user_text):
            stats["drop_not_question"] += 1
            continue
        reply_text = _one_line(str(row.get("response", "")))
        if not _safe_direct_reply(reply_text, min_chars=min_chars, max_chars=max_chars):
            stats["drop_reply_quality"] += 1
            continue
        if require_overlap and not _keyword_overlap(user_text, reply_text):
            stats["drop_overlap"] += 1
            continue
        out.append(_build_row(row=row, source_split=source_split))
    out = _dedupe(out)
    stats["kept_rows"] = len(out)
    return out, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-build a direct-answer seed dataset from existing persona-chat train/val JSONL.")
    parser.add_argument("--input_train", required=True)
    parser.add_argument("--input_val", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--preview_output", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_chars", type=int, default=4)
    parser.add_argument("--max_chars", type=int, default=56)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--require_overlap", action="store_true")
    args = parser.parse_args()

    train_rows = _load_jsonl(Path(args.input_train))
    val_rows = _load_jsonl(Path(args.input_val))
    mined_train, train_stats = _mine_rows(
        rows=train_rows,
        source_split="train",
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        require_overlap=args.require_overlap,
    )
    mined_val, val_stats = _mine_rows(
        rows=val_rows,
        source_split="val",
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        require_overlap=args.require_overlap,
    )

    rng = random.Random(args.seed)
    rng.shuffle(mined_train)
    rng.shuffle(mined_val)

    if args.train_limit > 0:
        mined_train = mined_train[: args.train_limit]
    if args.val_limit > 0:
        mined_val = mined_val[: args.val_limit]

    if len(mined_train) < 32 or len(mined_val) < 8:
        raise RuntimeError(
            f"Auto-mined dataset too small: train={len(mined_train)}, val={len(mined_val)}. Relax filters or disable require_overlap."
        )

    _write_jsonl(Path(args.train_output), mined_train)
    _write_jsonl(Path(args.val_output), mined_val)

    if args.preview_output:
        preview = {
            "train_count": len(mined_train),
            "val_count": len(mined_val),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "train_samples": mined_train[:3],
            "val_samples": mined_val[:2],
        }
        preview_path = Path(args.preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "sft_autobuild_direct_qa_done",
                "train_rows": len(mined_train),
                "val_rows": len(mined_val),
                "train_output": str(Path(args.train_output).as_posix()),
                "val_output": str(Path(args.val_output).as_posix()),
                "train_stats": train_stats,
                "val_stats": val_stats,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

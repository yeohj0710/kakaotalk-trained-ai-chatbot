from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


PROFANITY_RE = re.compile(
    r"(씨발|시발|씨바|병신|븅신|좆|좃|지랄|개새끼|씨발련|시발련|fuck|fucking|shit|bitch)",
    re.IGNORECASE,
)


def _load_jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict] = []
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


def _sanitize_corrected_reply(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\r", "\n").replace("\n", " ")).strip()


def _is_safe_correction(text: str) -> bool:
    out = _sanitize_corrected_reply(text)
    if len(out) < 2:
        return False
    if PROFANITY_RE.search(out):
        return False
    return True


def _load_reviewed_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Reviewed file not found: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {lineno} must be an object.")
            corrected = _sanitize_corrected_reply(str(payload.get("corrected_reply", "")))
            if not corrected:
                continue
            if not _is_safe_correction(corrected):
                continue
            payload["corrected_reply"] = corrected
            rows.append(payload)
    if not rows:
        raise RuntimeError("No reviewed rows with corrected_reply found.")
    return rows


def _messages_from_row(row: dict) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in row.get("history", []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if role not in {"user", "bot"} or not text:
            continue
        messages.append({"role": "assistant" if role == "bot" else "user", "content": text})
    prompt = str(row.get("prompt", "")).strip()
    if prompt:
        messages.append({"role": "user", "content": prompt})
    return messages


def _dataset_row(row: dict) -> dict:
    prompt = str(row.get("prompt", "")).strip()
    corrected = _sanitize_corrected_reply(str(row.get("corrected_reply", "")))
    messages = _messages_from_row(row)
    prompt_fallback = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"
    return {
        "prompt": prompt_fallback,
        "response": corrected,
        "messages": messages,
        "meta": {
            "source": "correction_review",
            "index": row.get("index"),
            "tags": row.get("tags", []),
            "notes": row.get("notes", ""),
            "expected": row.get("expected", ""),
        },
    }


def _anchor_dataset_row(row: dict) -> dict | None:
    prompt = str(row.get("prompt", "")).strip()
    response = _sanitize_corrected_reply(str(row.get("response", "")))
    if not prompt or not response:
        return None
    out = {
        "prompt": prompt,
        "response": response,
        "meta": dict(row.get("meta", {})),
    }
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        out["messages"] = messages
    out["meta"]["source"] = "anchor_persona"
    return out


def _sample_anchor_rows(path: Path, sample_count: int, seed: int) -> list[dict]:
    rows = [_anchor_dataset_row(row) for row in _load_jsonl_rows(path)]
    rows = [row for row in rows if row is not None]
    if sample_count <= 0 or len(rows) <= sample_count:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, sample_count)


def _dedupe_rows(rows: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("prompt", "")), str(row.get("response", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small correction SFT dataset from reviewed failure outputs.")
    parser.add_argument("--input", required=True, help="Reviewed JSONL with corrected_reply filled.")
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--preview_output", default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--correction_repeat", type=int, default=1)
    parser.add_argument("--anchor_train_input", default="")
    parser.add_argument("--anchor_val_input", default="")
    parser.add_argument("--anchor_train_count", type=int, default=0)
    parser.add_argument("--anchor_val_count", type=int, default=0)
    args = parser.parse_args()

    rows = _load_reviewed_rows(Path(args.input))
    correction_rows = [_dataset_row(row) for row in rows]
    correction_rows = _dedupe_rows(correction_rows)
    repeat = max(1, int(args.correction_repeat))
    expanded_rows: list[dict] = []
    for _ in range(repeat):
        expanded_rows.extend(correction_rows)

    rng = random.Random(args.seed)
    rng.shuffle(expanded_rows)

    val_count = max(1, int(round(len(correction_rows) * args.val_ratio)))
    val_count = min(val_count, max(1, len(correction_rows) - 1))
    val_seed_rows = correction_rows[:val_count]
    train_seed_rows = correction_rows[val_count:]
    train_rows = list(train_seed_rows) * repeat
    val_rows = list(val_seed_rows) * max(1, min(repeat, 3))

    anchor_train_rows: list[dict] = []
    anchor_val_rows: list[dict] = []
    if args.anchor_train_input:
        anchor_train_rows = _sample_anchor_rows(Path(args.anchor_train_input), args.anchor_train_count, args.seed + 11)
        train_rows.extend(anchor_train_rows)
    if args.anchor_val_input:
        anchor_val_rows = _sample_anchor_rows(Path(args.anchor_val_input), args.anchor_val_count, args.seed + 29)
        val_rows.extend(anchor_val_rows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    _write_jsonl(Path(args.train_output), train_rows)
    _write_jsonl(Path(args.val_output), val_rows)

    if args.preview_output:
        preview_path = Path(args.preview_output)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview = {
            "train_samples": train_rows[:3],
            "val_samples": val_rows[:2],
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "correction_count": len(correction_rows),
            "correction_repeat": repeat,
            "anchor_train_count": len(anchor_train_rows),
            "anchor_val_count": len(anchor_val_rows),
        }
        preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "sft_correction_build_done",
                "input_rows": len(rows),
                "correction_rows": len(correction_rows),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "anchor_train_rows": len(anchor_train_rows),
                "anchor_val_rows": len(anchor_val_rows),
                "train_output": str(Path(args.train_output).as_posix()),
                "val_output": str(Path(args.val_output).as_posix()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

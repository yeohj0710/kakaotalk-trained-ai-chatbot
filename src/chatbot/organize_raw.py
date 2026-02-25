from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_paths_config
from .preprocess import MESSAGE_PATTERN, read_chat_text


SAVED_AT_PATTERN = re.compile(r"^저장한 날짜 : (?P<saved>.+)$")
ROOM_TITLE_PATTERN = re.compile(r"^(?P<room>.+?)\s+카카오톡 대화$")


@dataclass
class PlanItem:
    source_path: Path
    source_name: str
    sha256: str
    bytes: int
    line_count: int
    message_like_lines: int
    room_guess: str
    saved_at_guess: str
    inbox_path: Path
    organized_path: Path


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w가-힣.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "generated_at": None, "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "entries" not in payload or not isinstance(payload["entries"], list):
        payload["entries"] = []
    return payload


def _save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative(path: Path) -> str:
    return path.as_posix()


def _parse_saved_at(raw: str) -> datetime | None:
    for fmt in ("%Y. %m. %d. %H:%M", "%Y.%m.%d. %H:%M", "%Y. %m. %d. %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _extract_metadata(path: Path) -> tuple[str, str, int, int]:
    text = read_chat_text(path)
    lines = text.splitlines()
    line_count = len(lines)
    message_like = 0
    saved_at_guess = "unknown"
    room_guess = path.stem

    first_non_empty = next((line.strip() for line in lines if line.strip()), "")
    if first_non_empty:
        room_match = ROOM_TITLE_PATTERN.match(first_non_empty)
        if room_match:
            room_guess = room_match.group("room").strip()
        elif "카카오톡 대화" in first_non_empty:
            room_guess = first_non_empty.replace("카카오톡 대화", "").strip() or path.stem

    for line in lines[:120]:
        line = line.strip()
        if not line:
            continue
        if MESSAGE_PATTERN.match(line):
            message_like += 1
        match = SAVED_AT_PATTERN.match(line)
        if match and saved_at_guess == "unknown":
            saved_at_guess = match.group("saved").strip()

    if message_like == 0:
        for line in lines[120:]:
            if MESSAGE_PATTERN.match(line):
                message_like += 1

    return room_guess, saved_at_guess, line_count, message_like


def _unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    idx = 2
    while True:
        candidate = parent / f"{stem}__{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _collect_root_txt(root_glob: str, exclude: set[str]) -> list[Path]:
    root = Path(".")
    files = sorted(root.glob(root_glob))
    out: list[Path] = []
    for path in files:
        if path.is_dir():
            continue
        if path.name in exclude:
            continue
        if path.name.lower().endswith(".txt"):
            out.append(path)
    return out


def build_plan(
    root_glob: str,
    exclude: set[str],
    inbox_dir: Path,
    organized_dir: Path,
) -> list[PlanItem]:
    plan: list[PlanItem] = []
    for src in _collect_root_txt(root_glob=root_glob, exclude=exclude):
        sha = _sha256_file(src)
        room_guess, saved_at_guess, line_count, message_like = _extract_metadata(src)
        saved_dt = _parse_saved_at(saved_at_guess)
        month_bucket = saved_dt.strftime("%Y-%m") if saved_dt else "unknown-date"
        safe_stem = _slugify(src.stem)
        safe_room = _slugify(room_guess)
        target_name = f"{safe_stem}__{sha[:12]}.txt"
        inbox_path = _unique_path(inbox_dir / target_name)
        organized_path = _unique_path(organized_dir / safe_room / month_bucket / target_name)
        plan.append(
            PlanItem(
                source_path=src,
                source_name=src.name,
                sha256=sha,
                bytes=src.stat().st_size,
                line_count=line_count,
                message_like_lines=message_like,
                room_guess=room_guess,
                saved_at_guess=saved_at_guess,
                inbox_path=inbox_path,
                organized_path=organized_path,
            )
        )
    return plan


def execute_plan(plan: list[PlanItem], manifest_path: Path, dry_run: bool) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    entries: list[dict[str, Any]] = manifest["entries"]
    existing_keys = {
        (entry.get("source_name"), entry.get("sha256"), entry.get("inbox_path"))
        for entry in entries
    }

    created = 0
    skipped = 0
    for item in plan:
        key = (item.source_name, item.sha256, _relative(item.inbox_path))
        if key in existing_keys:
            skipped += 1
            continue

        link_method = "none"
        if not dry_run:
            item.inbox_path.parent.mkdir(parents=True, exist_ok=True)
            item.organized_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.source_path), str(item.inbox_path))
            try:
                item.organized_path.hardlink_to(item.inbox_path)
                link_method = "hardlink"
            except OSError:
                shutil.copy2(item.inbox_path, item.organized_path)
                link_method = "copy"
        entry = {
            "moved_at": datetime.now().isoformat(timespec="seconds"),
            "source_path": _relative(item.source_path),
            "source_name": item.source_name,
            "sha256": item.sha256,
            "bytes": item.bytes,
            "line_count": item.line_count,
            "message_like_lines": item.message_like_lines,
            "room_guess": item.room_guess,
            "saved_at_guess": item.saved_at_guess,
            "inbox_path": _relative(item.inbox_path),
            "organized_path": _relative(item.organized_path),
            "link_method": link_method if not dry_run else "planned",
            "dry_run": dry_run,
        }
        entries.append(entry)
        created += 1

    if not dry_run:
        _save_manifest(manifest_path, manifest)
    return {
        "planned": len(plan),
        "created": created,
        "skipped": skipped,
        "dry_run": dry_run,
        "manifest_path": _relative(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize raw KakaoTalk txt exports safely.")
    parser.add_argument("--paths_config", default="configs/paths.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--root_glob", default="", help="Override root glob from paths config.")
    args = parser.parse_args()

    paths_cfg = load_paths_config(config_path=args.paths_config, env_path=args.env_path)
    raw_cfg = dict(paths_cfg.get("raw", {}))
    root_glob = args.root_glob or str(raw_cfg.get("root_glob", "*.txt"))
    exclude = {str(name) for name in raw_cfg.get("root_exclude", [])}
    inbox_dir = Path(str(raw_cfg.get("inbox_dir", "data/raw/inbox")))
    organized_dir = Path(str(raw_cfg.get("organized_dir", "data/raw/organized")))
    manifest_path = Path(str(raw_cfg.get("manifest_path", "data/raw/manifest.json")))

    plan = build_plan(
        root_glob=root_glob,
        exclude=exclude,
        inbox_dir=inbox_dir,
        organized_dir=organized_dir,
    )
    if not plan:
        print(json.dumps({"planned": 0, "message": "No root txt files found."}, ensure_ascii=False))
        return

    for item in plan:
        print(
            json.dumps(
                {
                    "source": _relative(item.source_path),
                    "inbox": _relative(item.inbox_path),
                    "organized": _relative(item.organized_path),
                    "room_guess": item.room_guess,
                    "saved_at_guess": item.saved_at_guess,
                    "sha256": item.sha256[:12],
                },
                ensure_ascii=False,
            )
        )

    summary = execute_plan(plan=plan, manifest_path=manifest_path, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

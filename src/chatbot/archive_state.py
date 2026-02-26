from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_paths_config


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
    return cleaned or "archive"


def _to_rel(path: Path) -> str:
    return path.as_posix()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    idx = 2
    while True:
        candidate = parent / f"{stem}__{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _candidate_sources(paths_cfg: dict[str, Any], include_raw: bool) -> list[Path]:
    processed_dir = Path(str(paths_cfg.get("processed", {}).get("output_dir", "data/processed")))
    checkpoints_dir = Path(str(paths_cfg.get("checkpoints", {}).get("root_dir", "checkpoints")))
    artifacts_dir = Path(str(paths_cfg.get("artifacts", {}).get("dir", "artifacts")))

    sources: list[Path] = [processed_dir, checkpoints_dir]
    if artifacts_dir.exists():
        for item in sorted(artifacts_dir.iterdir()):
            if item.name == ".gitkeep":
                continue
            sources.append(item)

    if include_raw:
        raw_cfg = dict(paths_cfg.get("raw", {}))
        sources.extend(
            [
                Path(str(raw_cfg.get("inbox_dir", "data/raw/inbox"))),
                Path(str(raw_cfg.get("organized_dir", "data/raw/organized"))),
                Path(str(raw_cfg.get("manifest_path", "data/raw/manifest.json"))),
            ]
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for src in sources:
        key = src.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)
    return deduped


def _move_items(
    sources: list[Path],
    archive_root: Path,
    include_raw: bool,
    dry_run: bool,
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_full" if include_raw else "_train_only"
    archive_dir = archive_root / f"{timestamp}{suffix}"

    planned: list[dict[str, str]] = []
    moved = 0
    skipped = 0
    for source in sources:
        if not source.exists():
            skipped += 1
            continue
        target = archive_dir / source
        target = _unique_path(target)
        planned.append({"source": _to_rel(source), "target": _to_rel(target)})
        print(json.dumps({"event": "archive_plan", "source": _to_rel(source), "target": _to_rel(target)}, ensure_ascii=False))

        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        moved += 1

    manifest = {
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "archive_dir": _to_rel(archive_dir),
        "dry_run": dry_run,
        "include_raw": include_raw,
        "moved_count": moved,
        "skipped_missing": skipped,
        "items": planned,
    }
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = archive_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # Recreate tracking-friendly directories after cleanup.
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        gitkeep = artifacts_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive previous training/processed artifacts and clean workspace.")
    parser.add_argument("--paths_config", default="configs/paths.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--include_raw", action="store_true", help="Also archive raw inbox/organized/manifest files.")
    parser.add_argument("--tag", default="", help="Optional archive tag suffix.")
    args = parser.parse_args()

    paths_cfg = load_paths_config(config_path=args.paths_config, env_path=args.env_path)
    archive_root = Path(str(paths_cfg.get("archive", {}).get("root_dir", "data/archive")))
    if args.tag.strip():
        archive_root = archive_root / _slug(args.tag)
    sources = _candidate_sources(paths_cfg=paths_cfg, include_raw=args.include_raw)
    result = _move_items(
        sources=sources,
        archive_root=archive_root,
        include_raw=args.include_raw,
        dry_run=args.dry_run,
    )
    print(json.dumps({"event": "archive_done", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

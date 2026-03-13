from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .security import require_password
from .sft_config import load_sft_config
from .sft_infer import SFTInferenceEngine, configure_console_io


def _read_text_with_fallbacks(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp949", "euc-kr", "utf-16")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return path.read_text(encoding="utf-8")


def _load_prompt_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input prompt file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        raw_text = _read_text_with_fallbacks(path)
        for lineno, raw in enumerate(raw_text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL line {lineno} must be an object.")
            rows.append(payload)
        return rows

    rows = []
    for raw in _read_text_with_fallbacks(path).splitlines():
        prompt = raw.strip()
        if not prompt:
            continue
        rows.append({"prompt": prompt})
    return rows


def _normalize_history(payload: Any) -> list[tuple[str, str]]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise ValueError("history must be a list.")
    history: list[tuple[str, str]] = []
    for item in payload:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip()
            text = str(item.get("text", "")).strip()
        elif isinstance(item, list | tuple) and len(item) == 2:
            role = str(item[0]).strip()
            text = str(item[1]).strip()
        else:
            raise ValueError("history items must be {role,text} or [role, text].")
        if role not in {"user", "bot"}:
            raise ValueError(f"Unsupported history role: {role}")
        if text:
            history.append((role, text))
    return history


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(description="Run prompt batches through the chatbot and save outputs for failure review.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--mode", default="one_on_one")
    parser.add_argument("--password", default="")
    parser.add_argument("--input", required=True, help="Prompt file (.txt or .jsonl).")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chain_history", action="store_true", help="Carry generated turns forward across rows.")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    security_cfg = dict(cfg.get("security", {}))
    password_env = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided = (args.password or "").strip() or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(password_env)
    require_password(security_cfg=security_cfg, password=provided, env_path=args.env_path)

    rows = _load_prompt_rows(Path(args.input))
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("No prompt rows found.")

    engine = SFTInferenceEngine.load(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name_override=args.run_name,
        mode_override=args.mode,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rolling_history: list[tuple[str, str]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows, start=1):
            prompt = str(row.get("prompt") or row.get("user") or "").strip()
            if not prompt:
                raise ValueError(f"Row {idx} missing prompt/user field.")

            base_history = _normalize_history(row.get("history"))
            history = list(rolling_history) if args.chain_history else list(base_history)
            if args.chain_history and base_history:
                history.extend(base_history)

            reply = engine.reply(history=history, user_text=prompt)
            out_row = {
                "index": idx,
                "prompt": prompt,
                "history": [{"role": role, "text": text} for role, text in history],
                "reply": reply,
                "tags": row.get("tags", []),
                "notes": row.get("notes", ""),
                "expected": row.get("expected", ""),
                "corrected_reply": "",
                "meta": {
                    "config_sft": args.config_sft,
                    "run_name": args.run_name,
                    "adapter_dir": str(engine.adapter_dir),
                    "mode": engine.options.inference_mode,
                },
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            print(f"[{idx}] U: {prompt}")
            print(f"[{idx}] B: {reply}")

            if args.chain_history:
                rolling_history.append(("user", prompt))
                rolling_history.append(("bot", reply))

    print(json.dumps({"event": "sft_collect_done", "rows": len(rows), "output": str(output_path.as_posix())}, ensure_ascii=False))


if __name__ == "__main__":
    main()

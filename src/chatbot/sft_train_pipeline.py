from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

from .sft_config import format_with_run_name, load_sft_config


def run_module(module: str, args: list[str]) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(json.dumps({"exec": cmd}, ensure_ascii=False))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def resolve_cpt_init_adapter(cfg: dict, cpt_run_name: str) -> str:
    paths_cfg = dict(cfg.get("paths", {}))
    checkpoints_root = Path(str(paths_cfg.get("checkpoints_root", "checkpoints_lora")))
    cpt_run_dir = checkpoints_root / cpt_run_name
    status_json = Path(format_with_run_name(str(paths_cfg.get("status_json", "checkpoints_lora/{run_name}/status.json")), cpt_run_name))

    candidates: list[Path] = []
    if status_json.exists():
        try:
            status = json.loads(status_json.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        for key in ("best_adapter_dir", "latest_adapter_dir"):
            val = str(status.get(key, "")).strip()
            if val:
                candidates.append(Path(val))
    candidates.extend([cpt_run_dir / "adapter_best", cpt_run_dir / "adapter_latest"])

    for path in candidates:
        if path.exists():
            return str(path.as_posix())
    return ""


def load_status(cfg: dict, run_name: str) -> dict:
    paths_cfg = dict(cfg.get("paths", {}))
    status_json = Path(format_with_run_name(str(paths_cfg.get("status_json", "checkpoints_lora/{run_name}/status.json")), run_name))
    if not status_json.exists():
        return {}
    try:
        return json.loads(status_json.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage training pipeline (CPT -> SFT) with auto-resume.")
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    project_cfg = dict(cfg.get("project", {}))
    paths_cfg = dict(cfg.get("paths", {}))
    pipeline_cfg = dict(cfg.get("pipeline", {}))

    run_name = str(project_cfg.get("run_name", "room_lora_qwen25_3b_base")).strip() or "room_lora_qwen25_3b_base"
    checkpoints_root = Path(str(paths_cfg.get("checkpoints_root", "checkpoints_lora")))
    sft_run_dir = checkpoints_root / run_name
    sft_last_ckpt = get_last_checkpoint(str(sft_run_dir)) if sft_run_dir.exists() else None

    pipeline_enabled = bool(pipeline_cfg.get("enabled", True))
    run_cpt_first = bool(pipeline_cfg.get("run_cpt_first", True))
    bootstrap_sft_from_cpt = bool(pipeline_cfg.get("bootstrap_sft_from_cpt", True))
    skip_cpt_if_sft_has_checkpoint = bool(pipeline_cfg.get("skip_cpt_if_sft_has_checkpoint", True))
    require_cpt_complete_before_sft = bool(pipeline_cfg.get("require_cpt_complete_before_sft", True))
    cpt_suffix = str(pipeline_cfg.get("cpt_run_name_suffix", "_cpt"))
    cpt_run_name = f"{run_name}{cpt_suffix}"

    if pipeline_enabled and run_cpt_first and not (skip_cpt_if_sft_has_checkpoint and sft_last_ckpt):
        cpt_code = run_module(
            "chatbot.sft_cpt_train",
            ["--config_sft", args.config_sft, "--env_path", args.env_path, "--run_name", cpt_run_name],
        )
        if cpt_code != 0:
            raise SystemExit(cpt_code)

        cpt_status = load_status(cfg, cpt_run_name)
        cpt_completed = bool(cpt_status.get("completed", False))
        cpt_stopped = bool(cpt_status.get("stopped", False))
        if cpt_stopped:
            print(json.dumps({"event": "pipeline_waiting", "reason": "cpt_stopped"}, ensure_ascii=False))
            raise SystemExit(0)
        if require_cpt_complete_before_sft and not cpt_completed:
            print(json.dumps({"event": "pipeline_waiting", "reason": "cpt_not_completed"}, ensure_ascii=False))
            raise SystemExit(0)
    else:
        print(
            json.dumps(
                {
                    "event": "cpt_skipped",
                    "pipeline_enabled": pipeline_enabled,
                    "run_cpt_first": run_cpt_first,
                    "skip_due_sft_checkpoint": bool(skip_cpt_if_sft_has_checkpoint and sft_last_ckpt),
                },
                ensure_ascii=False,
            )
        )

    sft_args = ["--config_sft", args.config_sft, "--env_path", args.env_path, "--run_name", run_name]
    sft_last_ckpt = get_last_checkpoint(str(sft_run_dir)) if sft_run_dir.exists() else None
    if bootstrap_sft_from_cpt and not sft_last_ckpt:
        init_adapter = resolve_cpt_init_adapter(cfg, cpt_run_name=cpt_run_name)
        if init_adapter:
            sft_args.extend(["--init_adapter", init_adapter])
            print(json.dumps({"event": "sft_bootstrap_from_cpt", "init_adapter": init_adapter}, ensure_ascii=False))
        else:
            print(json.dumps({"event": "sft_bootstrap_missing", "run_name": cpt_run_name}, ensure_ascii=False))

    sft_code = run_module("chatbot.sft_train", sft_args)
    raise SystemExit(sft_code)


if __name__ == "__main__":
    main()

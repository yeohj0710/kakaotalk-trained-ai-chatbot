from __future__ import annotations

import argparse
import re
import time
from collections import deque
from datetime import datetime

from .config import load_gen_config
from .console import safe_print
from .inference import InferenceEngine
from .security import require_password


EXPORT_LINE_PATTERN = re.compile(
    r"^(?P<ts>\d{4}\. \d{1,2}\. \d{1,2}\. \d{1,2}:\d{2}), (?P<speaker>[^:]+) : (?P<text>.+)$"
)
SIMPLE_LINE_PATTERN = re.compile(r"^(?P<speaker>[^:]{1,30})\s*:\s*(?P<text>.+)$")
SYSTEM_HINT_PATTERN = re.compile(r"(초대했습니다|나갔습니다|들어왔습니다|메시지가 삭제되었습니다)")


def parse_xy(raw: str) -> tuple[int, int]:
    x_str, y_str = raw.split(",", 1)
    return int(x_str.strip()), int(y_str.strip())


def focus_window(window_title: str) -> bool:
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return False
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        return False
    win = windows[0]
    try:
        win.activate()
        time.sleep(0.15)
        return True
    except Exception:
        return False


def copy_chat_text(chat_xy: tuple[int, int]) -> str:
    import pyautogui  # type: ignore
    import pyperclip  # type: ignore

    pyautogui.click(chat_xy[0], chat_xy[1])
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.05)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(0.1)
    return pyperclip.paste()


def send_reply(text: str, input_xy: tuple[int, int], press_enter: bool) -> None:
    import pyautogui  # type: ignore
    import pyperclip  # type: ignore

    pyautogui.click(input_xy[0], input_xy[1])
    time.sleep(0.05)
    pyperclip.copy(text)
    pyautogui.hotkey("ctrl", "v")
    if press_enter:
        pyautogui.press("enter")


def extract_last_message(raw_text: str, bot_speaker: str) -> tuple[str, str, str] | None:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if SYSTEM_HINT_PATTERN.search(line):
            continue
        match = EXPORT_LINE_PATTERN.match(line)
        if match is not None:
            speaker = match.group("speaker").strip()
            text = match.group("text").strip()
            ts = match.group("ts").strip()
            if speaker and text and speaker != bot_speaker:
                return ts, speaker, text
            continue
        match = SIMPLE_LINE_PATTERN.match(line)
        if match is not None:
            speaker = match.group("speaker").strip()
            text = match.group("text").strip()
            ts = datetime.now().isoformat(timespec="seconds")
            if speaker and text and speaker != bot_speaker:
                return ts, speaker, text
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "KakaoTalk bridge via desktop UI automation. "
            "Defaults to DRY-RUN unless --send is explicitly provided."
        )
    )
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--config_gen", default="configs/gen.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--password", default="")
    parser.add_argument("--run_name", default="")

    parser.add_argument("--bot_speaker", default="")
    parser.add_argument("--window_title", default="")
    parser.add_argument("--chat_xy", default="")
    parser.add_argument("--input_xy", default="")
    parser.add_argument("--poll_sec", type=float, default=-1.0)
    parser.add_argument("--max_history_turns", type=int, default=-1)
    parser.add_argument("--device", default="", choices=["", "auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="", choices=["", "auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--temperature", type=float, default=-1.0)
    parser.add_argument("--top_p", type=float, default=-1.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=-1)
    parser.add_argument("--repetition_penalty", type=float, default=-1.0)

    parser.add_argument("--send", action="store_true", help="Actually send message to KakaoTalk.")
    parser.add_argument("--dry", action="store_true", help="Force dry-run mode.")
    parser.add_argument(
        "--print_mouse",
        action="store_true",
        help="Print mouse coordinates continuously and exit (for calibration).",
    )
    parser.add_argument("--no_confirm", action="store_true", help="Skip terminal confirmation before each send.")
    parser.add_argument("--no_enter", action="store_true", help="Paste only; do not press Enter.")
    args = parser.parse_args()

    try:
        import pyautogui  # type: ignore # noqa: F401
        import pyperclip  # type: ignore # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("kakao_bridge requires pyautogui and pyperclip") from exc

    if args.print_mouse:
        import pyautogui  # type: ignore

        safe_print("Press Ctrl+C to stop mouse coordinate capture.")
        try:
            while True:
                x, y = pyautogui.position()
                safe_print(f"mouse={x},{y}", end="\r")
                time.sleep(0.1)
        except KeyboardInterrupt:
            safe_print("\nmouse capture stopped")
        return

    gen_cfg = load_gen_config(config_path=args.config_gen, env_path=args.env_path)
    sampling_cfg = dict(gen_cfg.get("sampling", {}))
    bridge_cfg = dict(gen_cfg.get("bridge", {}))
    chat_cfg = dict(gen_cfg.get("chat", {}))
    runtime_cfg = dict(gen_cfg.get("runtime", {}))
    security_cfg = dict(gen_cfg.get("security", {}))

    require_password(
        security_cfg=security_cfg,
        password=(args.password or None),
        env_path=args.env_path,
    )

    bot_speaker = args.bot_speaker or str(chat_cfg.get("bot_speaker", ""))
    if not bot_speaker:
        raise ValueError("bot_speaker is required (via --bot_speaker or configs/gen.yaml).")

    window_title = args.window_title or str(bridge_cfg.get("window_title", ""))
    chat_xy_raw = args.chat_xy or str(bridge_cfg.get("chat_xy", ""))
    input_xy_raw = args.input_xy or str(bridge_cfg.get("input_xy", ""))
    if not chat_xy_raw or not input_xy_raw:
        raise ValueError("chat_xy and input_xy are required (CLI or config).")
    chat_xy = parse_xy(chat_xy_raw)
    input_xy = parse_xy(input_xy_raw)

    poll_sec = args.poll_sec if args.poll_sec > 0 else float(bridge_cfg.get("poll_sec", 2.0))
    max_history_turns = (
        args.max_history_turns if args.max_history_turns > 0 else int(bridge_cfg.get("max_history_turns", 40))
    )
    temperature = args.temperature if args.temperature >= 0 else float(sampling_cfg.get("temperature", 0.9))
    top_p = args.top_p if args.top_p >= 0 else float(sampling_cfg.get("top_p", 0.95))
    top_k = args.top_k if args.top_k >= 0 else int(sampling_cfg.get("top_k", 120))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else int(sampling_cfg.get("max_new_tokens", 120))
    repetition_penalty = (
        args.repetition_penalty
        if args.repetition_penalty >= 0
        else float(sampling_cfg.get("repetition_penalty", 1.02))
    )

    send_enabled = bool(args.send and not args.dry)

    device = args.device or str(runtime_cfg.get("device", "auto"))
    dtype = args.dtype or str(runtime_cfg.get("dtype", "auto"))
    engine = InferenceEngine.load(
        checkpoint_path=(args.ckpt or None),
        device=device,
        dtype=dtype,
        gen_config_path=args.config_gen,
        env_path=args.env_path,
        run_name_override=args.run_name,
    )
    speakers = engine.tokenizer.speaker_names
    if bot_speaker not in speakers:
        raise ValueError(f"Unknown bot speaker: {bot_speaker}")

    seen_ids: set[tuple[str, str, str]] = set()
    history: deque[tuple[str, str]] = deque(maxlen=max_history_turns)

    mode = "SEND" if send_enabled else "DRY-RUN"
    safe_print(f"[bridge] checkpoint={engine.checkpoint_path}")
    safe_print(f"[bridge] mode={mode} bot={bot_speaker} poll={poll_sec}s")
    if not send_enabled:
        safe_print("[bridge] dry-run mode: no message will be sent.")

    while True:
        try:
            if window_title:
                focus_window(window_title)

            raw = copy_chat_text(chat_xy)
            latest = extract_last_message(raw, bot_speaker=bot_speaker)
            if latest is None:
                time.sleep(poll_sec)
                continue

            ts, speaker, text = latest
            msg_id = (ts, speaker, text)
            if msg_id in seen_ids:
                time.sleep(poll_sec)
                continue
            seen_ids.add(msg_id)

            history.append((speaker, text))
            reply = engine.generate_reply(
                history=list(history),
                bot_speaker=bot_speaker,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            history.append((bot_speaker, reply))

            safe_print(f"[in ] {speaker}: {text}")
            safe_print(f"[out] {bot_speaker}: {reply}")

            if send_enabled:
                do_send = True
                if not args.no_confirm:
                    ans = input("send? [y/N] ").strip().lower()
                    do_send = ans in {"y", "yes"}
                if do_send:
                    if window_title:
                        focus_window(window_title)
                    send_reply(reply, input_xy=input_xy, press_enter=(not args.no_enter))
                    safe_print("[send] delivered")
                else:
                    safe_print("[send] skipped")

            time.sleep(poll_sec)
        except KeyboardInterrupt:
            safe_print("\nbridge stopped")
            break
        except Exception as exc:  # noqa: BLE001
            safe_print(f"[warn] {exc}")
            time.sleep(poll_sec)


if __name__ == "__main__":
    main()

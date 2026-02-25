from __future__ import annotations

import argparse
import re
import time
from collections import deque
from datetime import datetime

from .console import safe_print
from .inference import InferenceEngine


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
            "Use with caution; defaults to dry-run (no sending)."
        )
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--bot_speaker", required=True)
    parser.add_argument("--window_title", default="", help="KakaoTalk window title (optional).")
    parser.add_argument("--chat_xy", required=True, help="chat-area coordinates, e.g. 500,320")
    parser.add_argument("--input_xy", required=True, help="input-box coordinates, e.g. 520,980")
    parser.add_argument("--poll_sec", type=float, default=2.0)
    parser.add_argument("--max_history_turns", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=120)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)
    parser.add_argument("--send", action="store_true", help="Actually send message to KakaoTalk.")
    parser.add_argument(
        "--print_mouse",
        action="store_true",
        help="Print mouse coordinates continuously and exit (for calibration).",
    )
    parser.add_argument(
        "--no_confirm",
        action="store_true",
        help="Skip terminal confirmation before each send.",
    )
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

    engine = InferenceEngine.load(args.ckpt, device=args.device, dtype=args.dtype)
    speakers = engine.tokenizer.speaker_names
    if args.bot_speaker not in speakers:
        raise ValueError(f"Unknown bot speaker: {args.bot_speaker}")

    chat_xy = parse_xy(args.chat_xy)
    input_xy = parse_xy(args.input_xy)

    seen_ids: set[tuple[str, str, str]] = set()
    history: deque[tuple[str, str]] = deque(maxlen=args.max_history_turns)

    mode = "SEND" if args.send else "DRY-RUN"
    safe_print(f"[bridge] mode={mode} bot={args.bot_speaker} poll={args.poll_sec}s")
    if not args.send:
        safe_print("[bridge] --send 미설정: 톡창으로 전송하지 않고 콘솔 출력만 수행")

    while True:
        try:
            if args.window_title:
                focus_window(args.window_title)

            raw = copy_chat_text(chat_xy)
            latest = extract_last_message(raw, bot_speaker=args.bot_speaker)
            if latest is None:
                time.sleep(args.poll_sec)
                continue

            ts, speaker, text = latest
            msg_id = (ts, speaker, text)
            if msg_id in seen_ids:
                time.sleep(args.poll_sec)
                continue
            seen_ids.add(msg_id)

            history.append((speaker, text))
            reply = engine.generate_reply(
                history=list(history),
                bot_speaker=args.bot_speaker,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            history.append((args.bot_speaker, reply))

            safe_print(f"[in ] {speaker}: {text}")
            safe_print(f"[out] {args.bot_speaker}: {reply}")

            if args.send:
                do_send = True
                if not args.no_confirm:
                    ans = input("send? [y/N] ").strip().lower()
                    do_send = ans in {"y", "yes"}
                if do_send:
                    if args.window_title:
                        focus_window(args.window_title)
                    send_reply(reply, input_xy=input_xy, press_enter=(not args.no_enter))
                    safe_print("[send] delivered")
                else:
                    safe_print("[send] skipped")

            time.sleep(args.poll_sec)
        except KeyboardInterrupt:
            safe_print("\nbridge stopped")
            break
        except Exception as exc:  # noqa: BLE001
            safe_print(f"[warn] {exc}")
            time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()

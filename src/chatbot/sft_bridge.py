from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .security import require_password
from .sft_config import load_sft_config
from .sft_infer import SFTInferenceEngine, configure_console_io


DATE_LINE_RE = re.compile(r"^\s*(?P<y>\d{4})년\s*(?P<m>\d{1,2})월\s*(?P<d>\d{1,2})일(?:\s+\S+)?\s*$")
MESSAGE_START_RE = re.compile(
    r"^\[(?P<speaker>[^\]]+)\]\s+\[(?P<ampm>오전|오후)\s+(?P<hm>\d{1,2}:\d{2})\]\s*(?P<text>.*)$"
)


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class ParsedMessage:
    day: date | None
    ampm: str
    hm: str
    speaker: str
    text: str
    uid: str


class SeenBuffer:
    def __init__(self, max_size: int) -> None:
        self.max_size = max(100, int(max_size))
        self._order: deque[str] = deque()
        self._set: set[str] = set()

    def add(self, uid: str) -> None:
        if uid in self._set:
            return
        self._order.append(uid)
        self._set.add(uid)
        while len(self._order) > self.max_size:
            old = self._order.popleft()
            self._set.discard(old)

    def contains(self, uid: str) -> bool:
        return uid in self._set


def parse_xy(raw: str) -> Point:
    x_str, y_str = raw.split(",", 1)
    return Point(x=int(x_str.strip()), y=int(y_str.strip()))


def normalize_rect(start: Point, end: Point) -> Rect:
    x1 = min(start.x, end.x)
    y1 = min(start.y, end.y)
    x2 = max(start.x, end.x)
    y2 = max(start.y, end.y)
    return Rect(x1=x1, y1=y1, x2=x2, y2=y2)


def save_calibration(path: Path, drag_start: Point, drag_end: Point, input_point: Point) -> None:
    payload = {
        "drag_start": [drag_start.x, drag_start.y],
        "drag_end": [drag_end.x, drag_end.y],
        "input_box": [input_point.x, input_point.y],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> tuple[Point, Point, Point]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    drag_start = payload.get("drag_start") or []
    drag_end = payload.get("drag_end") or []
    input_box = payload.get("input_box") or []
    if len(drag_start) != 2 or len(drag_end) != 2 or len(input_box) != 2:
        raise ValueError(f"Invalid calibration format: {path}")
    return (
        Point(int(drag_start[0]), int(drag_start[1])),
        Point(int(drag_end[0]), int(drag_end[1])),
        Point(int(input_box[0]), int(input_box[1])),
    )


def focus_window(window_title: str) -> bool:
    if not window_title:
        return True
    try:
        import pygetwindow as gw  # type: ignore
    except Exception:
        return False
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        return False
    try:
        wins[0].activate()
        time.sleep(0.12)
        return True
    except Exception:
        return False


def capture_point(label: str) -> Point:
    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Calibration requires pyautogui.") from exc

    print(f"[calibrate] Move mouse to '{label}', then press SPACE. (q to cancel)")
    if os.name == "nt":
        import msvcrt

        while True:
            x, y = pyautogui.position()
            sys.stdout.write(f"\r{label:<20} {x:>5},{y:<5}")
            sys.stdout.flush()
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == " ":
                    print()
                    return Point(x=int(x), y=int(y))
                if key.lower() == "q":
                    raise KeyboardInterrupt("Calibration cancelled by user.")
            time.sleep(0.04)

    while True:
        x, y = pyautogui.position()
        print(f"[calibrate] {label}: current={x},{y} press Enter to capture, q then Enter to cancel")
        key = input().strip().lower()
        if key == "q":
            raise KeyboardInterrupt("Calibration cancelled by user.")
        return Point(x=int(x), y=int(y))


def calibrate_points() -> tuple[Point, Point, Point]:
    drag_start = capture_point("drag start (top-left)")
    drag_end = capture_point("drag end (bottom-right)")
    input_point = capture_point("input box")
    return drag_start, drag_end, input_point


def with_clipboard_restore(fn):
    import pyperclip  # type: ignore

    before = pyperclip.paste()
    try:
        return fn()
    finally:
        try:
            pyperclip.copy(before)
        except Exception:
            pass


def copy_chat_region(rect: Rect, drag_duration: float, copy_wait_sec: float) -> str:
    try:
        import pyautogui  # type: ignore
        import pyperclip  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Bridge UI mode requires pyautogui and pyperclip.") from exc

    def _copy() -> str:
        pyautogui.moveTo(rect.x1, rect.y1, duration=0.02)
        pyautogui.mouseDown()
        pyautogui.moveTo(rect.x2, rect.y2, duration=max(0.02, drag_duration))
        pyautogui.mouseUp()
        time.sleep(0.03)
        pyautogui.hotkey("ctrl", "c")
        time.sleep(max(0.05, copy_wait_sec))
        return str(pyperclip.paste() or "")

    return with_clipboard_restore(_copy)


def paste_and_send(text: str, input_point: Point, press_enter: bool) -> None:
    try:
        import pyautogui  # type: ignore
        import pyperclip  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Bridge send mode requires pyautogui and pyperclip.") from exc

    def _send() -> None:
        pyautogui.click(input_point.x, input_point.y)
        time.sleep(0.04)
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        if press_enter:
            pyautogui.press("enter")

    with_clipboard_restore(_send)


def to_24h(ampm: str, hm: str) -> str:
    hour_str, minute_str = hm.split(":", 1)
    hour = int(hour_str)
    minute = int(minute_str)
    if ampm == "오후" and hour < 12:
        hour += 12
    if ampm == "오전" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def normalize_id_text(text: str) -> str:
    out = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    out = re.sub(r"\s+", " ", out)
    return out


def make_message_uid(day: date | None, ampm: str, hm: str, speaker: str, text: str) -> str:
    day_key = day.isoformat() if day is not None else "unknown-day"
    normalized = normalize_id_text(text)
    material = f"{day_key}|{to_24h(ampm, hm)}|{speaker.strip()}|{normalized}"
    return hashlib.sha1(material.encode("utf-8", errors="ignore")).hexdigest()


def parse_kakao_snapshot(raw_text: str) -> list[ParsedMessage]:
    lines = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    active_day: date | None = None
    active: dict[str, object] | None = None
    messages: list[ParsedMessage] = []

    def flush_active() -> None:
        nonlocal active
        if active is None:
            return
        speaker = str(active.get("speaker", "")).strip()
        ampm = str(active.get("ampm", "")).strip()
        hm = str(active.get("hm", "")).strip()
        parts = [str(x) for x in list(active.get("parts", []))]
        text = "\n".join(parts).strip()
        if speaker and ampm and hm and text:
            uid = make_message_uid(
                day=active_day if isinstance(active_day, date) else None,
                ampm=ampm,
                hm=hm,
                speaker=speaker,
                text=text,
            )
            messages.append(
                ParsedMessage(
                    day=active_day if isinstance(active_day, date) else None,
                    ampm=ampm,
                    hm=hm,
                    speaker=speaker,
                    text=text,
                    uid=uid,
                )
            )
        active = None

    for raw_line in lines:
        line = raw_line.replace("\ufeff", "").rstrip()
        stripped = line.strip()

        if not stripped:
            if active is not None:
                active_parts = list(active.get("parts", []))
                active_parts.append("")
                active["parts"] = active_parts
            continue

        date_match = DATE_LINE_RE.match(stripped)
        if date_match is not None:
            flush_active()
            try:
                active_day = date(
                    year=int(date_match.group("y")),
                    month=int(date_match.group("m")),
                    day=int(date_match.group("d")),
                )
            except Exception:
                active_day = None
            continue

        msg_match = MESSAGE_START_RE.match(stripped)
        if msg_match is not None:
            flush_active()
            active = {
                "speaker": msg_match.group("speaker").strip(),
                "ampm": msg_match.group("ampm").strip(),
                "hm": msg_match.group("hm").strip(),
                "parts": [msg_match.group("text")],
            }
            continue

        if active is not None:
            active_parts = list(active.get("parts", []))
            active_parts.append(stripped)
            active["parts"] = active_parts

    flush_active()
    return messages


def render_user_turn(speaker: str, text: str, include_speaker_prefix: bool) -> str:
    if include_speaker_prefix:
        return f"{speaker}: {text}"
    return text


def parse_password_arg(password: str) -> str | None:
    value = (password or "").strip()
    return value if value else None


def main() -> None:
    configure_console_io()
    parser = argparse.ArgumentParser(
        description=(
            "KakaoTalk desktop bridge: drag-copy chat region, detect new messages, "
            "run SFT inference, and optionally auto-send replies."
        )
    )
    parser.add_argument("--config_sft", default="configs/sft.yaml")
    parser.add_argument("--env_path", default=".env")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--mode", default="group")
    parser.add_argument("--password", default="")

    parser.add_argument("--poll_sec", type=float, default=5.0)
    parser.add_argument("--drag_start", default="", help="x,y")
    parser.add_argument("--drag_end", default="", help="x,y")
    parser.add_argument("--input_xy", default="", help="x,y")
    parser.add_argument("--window_title", default="")
    parser.add_argument("--drag_duration", type=float, default=0.18)
    parser.add_argument("--copy_wait_sec", type=float, default=0.10)

    parser.add_argument("--load_calibration", default="")
    parser.add_argument("--save_calibration", default="")
    parser.add_argument("--calibrate", action="store_true")

    parser.add_argument("--send", action="store_true", help="Actually paste+enter into Kakao input box.")
    parser.add_argument("--no_enter", action="store_true", help="Paste only; do not press Enter.")
    parser.add_argument("--bot_name", default="", help="Your Kakao nickname in that room.")
    parser.add_argument("--ignore_speaker", action="append", default=[], help="Ignore these speakers completely.")
    parser.add_argument("--include_speaker_prefix", action="store_true", default=True)
    parser.add_argument("--no_speaker_prefix", dest="include_speaker_prefix", action="store_false")

    parser.add_argument("--history_limit", type=int, default=80)
    parser.add_argument("--seed_messages", type=int, default=24)
    parser.add_argument("--seen_buffer_size", type=int, default=2000)
    parser.add_argument("--max_replies_per_cycle", type=int, default=1)
    parser.add_argument("--min_send_interval_sec", type=float, default=12.0)
    parser.add_argument("--max_sends_per_hour", type=int, default=40)
    parser.add_argument("--stop_file", default="")
    parser.add_argument("--print_snapshot_size", action="store_true")
    args = parser.parse_args()

    cfg = load_sft_config(config_path=args.config_sft, env_path=args.env_path)
    security_cfg = dict(cfg.get("security", {}))
    env_name = str(security_cfg.get("password_env", "CHATBOT_PASSWORD"))
    provided = parse_password_arg(args.password) or os.getenv("CHATBOT_ACCESS_PASSWORD") or os.getenv(env_name)
    require_password(security_cfg=security_cfg, password=provided, env_path=args.env_path)

    drag_start: Point | None = None
    drag_end: Point | None = None
    input_point: Point | None = None

    if args.load_calibration:
        drag_start, drag_end, input_point = load_calibration(Path(args.load_calibration))
        print(f"[bridge] loaded calibration: {args.load_calibration}")

    if args.drag_start:
        drag_start = parse_xy(args.drag_start)
    if args.drag_end:
        drag_end = parse_xy(args.drag_end)
    if args.input_xy:
        input_point = parse_xy(args.input_xy)

    if args.calibrate or drag_start is None or drag_end is None or input_point is None:
        drag_start, drag_end, input_point = calibrate_points()
        if args.save_calibration:
            save_calibration(Path(args.save_calibration), drag_start=drag_start, drag_end=drag_end, input_point=input_point)
            print(f"[bridge] saved calibration: {args.save_calibration}")

    rect = normalize_rect(drag_start, drag_end)
    send_enabled = bool(args.send)
    ignore_speakers = {x.strip() for x in args.ignore_speaker if x and x.strip()}
    if args.bot_name:
        ignore_speakers.discard(args.bot_name)

    engine = SFTInferenceEngine.load(
        config_sft=args.config_sft,
        env_path=args.env_path,
        adapter_path=args.adapter,
        run_name_override=args.run_name,
        mode_override=args.mode,
    )

    history: deque[tuple[str, str]] = deque(maxlen=max(8, int(args.history_limit)))
    seen = SeenBuffer(max_size=max(200, int(args.seen_buffer_size)))
    send_timestamps: deque[float] = deque()
    last_send_ts = 0.0
    initialized = False
    last_snapshot_hash = ""

    stop_path = Path(args.stop_file).resolve() if args.stop_file else None
    print(f"[bridge] adapter={engine.adapter_dir}")
    print(f"[bridge] mode={engine.options.inference_mode}")
    print(f"[bridge] poll={args.poll_sec}s send_enabled={send_enabled}")
    if stop_path is not None:
        print(f"[bridge] stop_file={stop_path}")

    while True:
        try:
            if stop_path is not None and stop_path.exists():
                print("[bridge] stop file detected. exiting.")
                break

            if args.window_title:
                focus_window(args.window_title)

            snapshot = copy_chat_region(
                rect=rect,
                drag_duration=float(args.drag_duration),
                copy_wait_sec=float(args.copy_wait_sec),
            )
            snap_hash = hashlib.sha1(snapshot.encode("utf-8", errors="ignore")).hexdigest()
            if snap_hash == last_snapshot_hash:
                time.sleep(max(0.3, args.poll_sec))
                continue
            last_snapshot_hash = snap_hash
            if args.print_snapshot_size:
                print(f"[bridge] copied chars={len(snapshot)}")

            parsed = parse_kakao_snapshot(snapshot)
            if not parsed:
                time.sleep(max(0.3, args.poll_sec))
                continue

            if not initialized:
                for msg in parsed:
                    seen.add(msg.uid)
                seed_count = max(0, int(args.seed_messages))
                seed_slice = parsed[-seed_count:] if seed_count > 0 else []
                for msg in seed_slice:
                    if msg.speaker in ignore_speakers:
                        continue
                    role = "bot" if args.bot_name and msg.speaker == args.bot_name else "user"
                    if role == "bot":
                        history.append(("bot", msg.text))
                    else:
                        history.append(
                            (
                                "user",
                                render_user_turn(
                                    speaker=msg.speaker,
                                    text=msg.text,
                                    include_speaker_prefix=bool(args.include_speaker_prefix),
                                ),
                            )
                        )
                initialized = True
                print(f"[bridge] initialized. seeded_history={len(history)} seen={len(parsed)}")
                time.sleep(max(0.3, args.poll_sec))
                continue

            new_messages: list[ParsedMessage] = []
            for msg in parsed:
                if seen.contains(msg.uid):
                    continue
                seen.add(msg.uid)
                new_messages.append(msg)

            if not new_messages:
                time.sleep(max(0.3, args.poll_sec))
                continue

            replies_this_cycle = 0
            for msg in new_messages:
                if msg.speaker in ignore_speakers:
                    continue
                if args.bot_name and msg.speaker == args.bot_name:
                    history.append(("bot", msg.text))
                    continue

                user_turn = render_user_turn(
                    speaker=msg.speaker,
                    text=msg.text,
                    include_speaker_prefix=bool(args.include_speaker_prefix),
                )
                should_reply, reply = engine.reply_or_skip(history=list(history), user_text=user_turn)
                history.append(("user", user_turn))

                stamp = msg.day.isoformat() if msg.day else "unknown-day"
                print(f"[in ] {stamp} {msg.ampm} {msg.hm} {msg.speaker}: {msg.text}")

                if not should_reply:
                    print(f"[skip] {engine.options.group_no_reply_token}")
                    continue
                reply = (reply or "").strip()
                if not reply or reply == engine.options.group_no_reply_token:
                    print(f"[skip] {engine.options.group_no_reply_token}")
                    continue
                if replies_this_cycle >= max(1, int(args.max_replies_per_cycle)):
                    print("[skip] max_replies_per_cycle reached")
                    continue

                now = time.time()
                while send_timestamps and now - send_timestamps[0] >= 3600.0:
                    send_timestamps.popleft()
                if args.max_sends_per_hour > 0 and len(send_timestamps) >= int(args.max_sends_per_hour):
                    print("[skip] hourly send limit reached")
                    continue
                if args.min_send_interval_sec > 0 and (now - last_send_ts) < float(args.min_send_interval_sec):
                    print("[skip] min send interval throttled")
                    continue

                print(f"[out] {reply}")
                if send_enabled:
                    if args.window_title:
                        focus_window(args.window_title)
                    paste_and_send(reply, input_point=input_point, press_enter=(not args.no_enter))
                    print("[send] delivered")
                    last_send_ts = now
                    send_timestamps.append(now)
                else:
                    print("[send] dry-run (use --send to enable)")

                history.append(("bot", reply))
                replies_this_cycle += 1

            time.sleep(max(0.3, args.poll_sec))
        except KeyboardInterrupt:
            print("\n[bridge] stopped")
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge][warn] {exc}")
            time.sleep(max(0.3, args.poll_sec))


if __name__ == "__main__":
    main()

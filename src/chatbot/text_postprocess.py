from __future__ import annotations

import re
from dataclasses import dataclass

from .tokenizer import ByteSpecialTokenizer


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SPEAKER_TOKEN_RE = re.compile(r"<\|spk:[^|>]+?\|>")
OPEN_BRACKET_CLASS = r"[\[\(\<\{\u3008\u300a\u3010]"
CLOSE_BRACKET_CLASS = r"[\]\)\>\}\u3009\u300b\u3011]"
COLON_CLASS = r"[:\uFF1A]"
END_DELIM_CLASS = r"[\)\]\>\}\u3009\u300b\u3011]"


@dataclass
class OutputOptions:
    text_only: bool = True
    max_chars: int = 400
    strip_prefix: bool = True
    stop_on_next_turn: bool = True
    one_line: bool = True
    user_tag: str = "<U>"
    bot_tag: str = "<B>"


def _speaker_name_alternatives(tokenizer: ByteSpecialTokenizer, bot_speaker: str | None) -> list[str]:
    names = set(tokenizer.speaker_names)
    if bot_speaker:
        names.add(bot_speaker)
    for raw in list(names):
        stripped = raw.strip()
        if not stripped:
            continue
        names.add(stripped.replace("_", " "))
    return sorted((name for name in names if name), key=len, reverse=True)


def _strip_special_prefix_tokens(text: str, tokenizer: ByteSpecialTokenizer) -> str:
    out = text
    special_tokens = {
        tokenizer.bos_token,
        tokenizer.eos_token,
        tokenizer.pad_token,
        *tokenizer.speaker_to_token.values(),
    }
    changed = True
    while changed and out:
        changed = False
        out = out.lstrip()
        for token in special_tokens:
            if out.startswith(token):
                out = out[len(token) :]
                changed = True
        # Catch malformed or unknown speaker token forms as well.
        replaced = SPEAKER_TOKEN_RE.sub("", out, count=1)
        if replaced != out:
            out = replaced
            changed = True
    return out.lstrip()


def _strip_known_name_prefix(text: str, name_alt: str) -> str:
    patterns = [
        rf"^\s*(?:{name_alt})\s*{COLON_CLASS}\s*",
        rf"^\s*(?:{name_alt})\s*{END_DELIM_CLASS}\s*",
        rf"^\s*{OPEN_BRACKET_CLASS}\s*(?:{name_alt})\s*{CLOSE_BRACKET_CLASS}\s*(?:{COLON_CLASS})?\s*",
        rf"^\s*<\s*SPK[_\-\s]*(?:{name_alt})\s*>\s*(?:{COLON_CLASS})?\s*",
        rf"^\s*SPK[_\-\s]*(?:{name_alt})\s*(?:{COLON_CLASS})?\s*",
    ]
    out = text
    changed = True
    while changed and out:
        changed = False
        for pattern in patterns:
            replaced = re.sub(pattern, "", out, count=1, flags=re.IGNORECASE)
            if replaced != out:
                out = replaced.lstrip()
                changed = True
    return out


def strip_speaker_prefix(
    text: str,
    bot_speaker: str | None,
    tokenizer: ByteSpecialTokenizer,
    user_tag: str = "<U>",
    bot_tag: str = "<B>",
) -> str:
    if not text:
        return ""
    out = text.replace("\ufeff", "").lstrip()
    out = _strip_special_prefix_tokens(text=out, tokenizer=tokenizer)

    names = _speaker_name_alternatives(tokenizer=tokenizer, bot_speaker=bot_speaker)
    if names:
        name_alt = "|".join(re.escape(name) for name in names)
        out = _strip_known_name_prefix(text=out, name_alt=name_alt)

    for tag in (user_tag, bot_tag):
        cleaned_tag = (tag or "").strip()
        if not cleaned_tag:
            continue
        out = re.sub(rf"^\s*{re.escape(cleaned_tag)}\s*", "", out, count=1)
    return out.lstrip()


def _find_textual_turn_positions(text: str, name_alt: str) -> list[int]:
    turn_patterns = [
        rf"\n\s*(?:{name_alt})\s*{COLON_CLASS}\s*",
        rf"\n\s*(?:{name_alt})\s*{END_DELIM_CLASS}\s*",
        rf"\n\s*{OPEN_BRACKET_CLASS}\s*(?:{name_alt})\s*{CLOSE_BRACKET_CLASS}\s*(?:{COLON_CLASS})?\s*",
        rf"\n\s*<\s*SPK[_\-\s]*(?:{name_alt})\s*>\s*(?:{COLON_CLASS})?\s*",
        rf"\n\s*SPK[_\-\s]*(?:{name_alt})\s*(?:{COLON_CLASS})?\s*",
    ]
    positions: list[int] = []
    for pattern in turn_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            positions.append(match.start())
    return positions


def _leading_marker_offset(text: str, tokenizer: ByteSpecialTokenizer) -> int:
    scan = text
    consumed = 0
    ws_len = len(scan) - len(scan.lstrip())
    consumed += ws_len
    scan = scan.lstrip()

    fixed_tokens = (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)
    changed = True
    while changed and scan:
        changed = False
        for token in fixed_tokens:
            if scan.startswith(token):
                scan = scan[len(token) :].lstrip()
                consumed += len(token)
                ws_len = len(scan) - len(scan.lstrip())
                consumed += ws_len
                scan = scan.lstrip()
                changed = True

        speaker_match = SPEAKER_TOKEN_RE.match(scan)
        if speaker_match is not None:
            scan = scan[speaker_match.end() :].lstrip()
            consumed += speaker_match.end()
            ws_len = len(scan) - len(scan.lstrip())
            consumed += ws_len
            scan = scan.lstrip()
            changed = True
    return min(consumed, len(text))


def cut_at_next_turn(
    text: str,
    tokenizer: ByteSpecialTokenizer,
    user_tag: str = "<U>",
    bot_tag: str = "<B>",
) -> str:
    if not text:
        return ""
    offset = _leading_marker_offset(text=text, tokenizer=tokenizer)
    scan_text = text[offset:]
    cut_positions: list[int] = []

    for marker in (tokenizer.eos_token, tokenizer.bos_token):
        idx = scan_text.find(marker)
        if idx >= 0:
            cut_positions.append(offset + idx)

    marker_match = SPEAKER_TOKEN_RE.search(scan_text)
    if marker_match is not None:
        cut_positions.append(offset + marker_match.start())

    for speaker_token in set(tokenizer.speaker_to_token.values()):
        idx = scan_text.find(speaker_token)
        if idx >= 0:
            cut_positions.append(offset + idx)

    names = _speaker_name_alternatives(tokenizer=tokenizer, bot_speaker=None)
    if names:
        name_alt = "|".join(re.escape(name) for name in names)
        cut_positions.extend(
            [offset + pos for pos in _find_textual_turn_positions(text=scan_text, name_alt=name_alt)]
        )

    for tag in (user_tag, bot_tag):
        cleaned_tag = (tag or "").strip()
        if not cleaned_tag:
            continue
        for marker in (f"\n{cleaned_tag}", cleaned_tag):
            idx = scan_text.find(marker)
            if idx > 0:
                cut_positions.append(offset + idx)

    if cut_positions:
        return text[: min(cut_positions)]
    return text


def sanitize_for_send(text: str, max_chars: int = 0, one_line: bool = False) -> str:
    if not text:
        return ""
    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = out.replace("\u00a0", " ")
    out = out.replace("\u200b", "")
    out = CONTROL_CHARS_RE.sub("", out)
    if one_line:
        out = re.sub(r"\s*\n+\s*", " ", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
    else:
        out = re.sub(r"[ \t]+\n", "\n", out)
        out = re.sub(r"\n[ \t]+", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        out = re.sub(r"[ \t]{2,}", " ", out)
    out = out.strip()
    if max_chars > 0 and len(out) > max_chars:
        out = out[:max_chars].rstrip()
    return out


def postprocess_generated_text(
    raw_text: str,
    tokenizer: ByteSpecialTokenizer,
    bot_speaker: str | None,
    options: OutputOptions,
) -> str:
    if not options.text_only:
        return sanitize_for_send(raw_text, max_chars=options.max_chars, one_line=options.one_line)

    out = raw_text
    if options.stop_on_next_turn:
        out = cut_at_next_turn(
            out,
            tokenizer=tokenizer,
            user_tag=options.user_tag,
            bot_tag=options.bot_tag,
        )
    if options.strip_prefix:
        out = strip_speaker_prefix(
            out,
            bot_speaker=bot_speaker,
            tokenizer=tokenizer,
            user_tag=options.user_tag,
            bot_tag=options.bot_tag,
        )
    return sanitize_for_send(out, max_chars=options.max_chars, one_line=options.one_line)

from __future__ import annotations

import sys


def to_console_safe(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def safe_print(*values: object, sep: str = " ", end: str = "\n") -> None:
    if not values:
        print(end=end)
        return
    converted = sep.join(to_console_safe(str(value)) for value in values)
    print(converted, end=end)

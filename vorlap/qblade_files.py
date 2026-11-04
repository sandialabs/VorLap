"""Shared QBlade value-leading keyword and path helpers."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
import re
from typing import Any, Iterable


QBLADE_KEYVALUE_RE = re.compile(r"^(?P<value>.*?)\s+(?P<keyword>[A-Z][A-Z0-9_]+)\s*(?P<rest>-.*)?$")


def render_qblade_value(value: float | int | str) -> str:
    """Render scalar values with the compact formatting used in QBlade text files."""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def parse_qblade_keyword_line(line: str) -> tuple[str, str] | None:
    """Parse one value-leading QBlade keyword line as `(keyword, value)`."""
    text = str(line).strip()
    if not text or re.match(r"^-{3,}", text):
        return None
    match = QBLADE_KEYVALUE_RE.match(text)
    if match is None:
        return None
    return match.group("keyword").strip(), match.group("value").strip()


def parse_qblade_keywords(path: str | Path) -> dict[str, list[str]]:
    """Parse value-leading QBlade keyword lines into `keyword -> values`."""
    qblade_path = Path(path)
    if not qblade_path.is_file():
        raise FileNotFoundError(f"QBlade file does not exist: {qblade_path}")

    parsed: dict[str, list[str]] = {}
    with qblade_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw in file:
            item = parse_qblade_keyword_line(raw)
            if item is None:
                continue
            keyword, value = item
            parsed.setdefault(keyword, []).append(value)
    return parsed


def first_qblade_keyword_value(
    parsed: dict[str, list[str]],
    keyword: str,
    default: str | None = None,
) -> str | None:
    """Return the first parsed value for `keyword`, or `default` when absent."""
    values = parsed.get(keyword, [])
    return values[0] if values else default


def read_qblade_keyword(path: str | Path, keyword: str) -> str | None:
    """Read the first value for a QBlade keyword from a text definition file."""
    return first_qblade_keyword_value(parse_qblade_keywords(path), keyword)


def read_qblade_float_keyword(path: str | Path, keyword: str) -> float:
    """Read a required QBlade keyword value as float."""
    raw = read_qblade_keyword(path, keyword)
    if raw is None:
        raise ValueError(f"Could not find {keyword} in {path}")
    return float(raw)


def is_absolute_qblade_path(path: str | Path) -> bool:
    """Return true for POSIX or Windows absolute paths stored in QBlade files."""
    text = str(path).strip()
    return Path(text).is_absolute() or PureWindowsPath(text).is_absolute()


def resolve_qblade_path(
    base_file: str | Path,
    candidate: str | Path,
    *,
    extra_bases: Iterable[str | Path] | None = None,
) -> Path:
    """Resolve a QBlade relative path against its parent file and optional bases."""
    text = str(candidate).strip()
    if not text:
        return Path("")
    if is_absolute_qblade_path(text):
        return Path(text)

    bases = [Path(base_file).expanduser().resolve(strict=False).parent]
    for base in extra_bases or ():
        if str(base).strip():
            bases.append(Path(base).expanduser().resolve(strict=False))

    for base in bases:
        resolved = (base / text).resolve(strict=False)
        if resolved.exists():
            return resolved
    return (bases[0] / text).resolve(strict=False)


def resolve_qblade_path_text(
    base_file: str | Path,
    candidate: str | Path,
    *,
    extra_bases: Iterable[str | Path] | None = None,
) -> str:
    """Resolve a QBlade path and return it as text for legacy dictionary APIs."""
    text = str(candidate).strip()
    if is_absolute_qblade_path(text):
        return text
    return str(resolve_qblade_path(base_file, text, extra_bases=extra_bases))


def format_qblade_keyword_line(
    value: Any,
    keyword: str,
    comment: str = "",
    *,
    value_width: int = 50,
    keyword_width: int = 18,
) -> str:
    """Render a value-leading QBlade keyword line."""
    value_text = str(value)
    rendered_value = f"{value_text:<{value_width}}"
    if len(value_text) >= value_width:
        rendered_value += " "
    line = f"{rendered_value}{keyword:<{keyword_width}}"
    if comment:
        line += f" - {comment}"
    return line


def replace_qblade_keyword_text(
    text: str,
    keyword: str,
    value: float | int | str,
    *,
    comment: str | None = None,
    value_width: int = 50,
    keyword_width: int = 18,
) -> tuple[str, str, int]:
    """Replace the first value-leading keyword line in text.

    When `comment` is `None`, the original trailing comment/rest text is preserved.
    When `comment` is an empty string, the existing trailing comment is removed.
    """
    rendered = render_qblade_value(value)
    pattern = re.compile(
        rf"^(?P<old>.*?)[ \t]+{re.escape(keyword)}(?P<rest>[ \t]*(?:-.*)?)$",
        re.MULTILINE,
    )
    old_value = ""

    def replacement(match: re.Match[str]) -> str:
        nonlocal old_value
        old_value = match.group("old").strip()
        if comment is None:
            rest = match.group("rest")
            return format_qblade_keyword_line(
                rendered,
                keyword,
                "",
                value_width=value_width,
                keyword_width=keyword_width,
            ) + rest
        return format_qblade_keyword_line(
            rendered,
            keyword,
            comment,
            value_width=value_width,
            keyword_width=keyword_width,
        )

    patched, count = pattern.subn(replacement, text, count=1)
    return patched, old_value, count


def replace_qblade_keyword_file(
    path: str | Path,
    keyword: str,
    value: float | int | str,
    *,
    comment: str | None = None,
    value_width: int = 50,
    keyword_width: int = 18,
) -> dict[str, str]:
    """Replace one QBlade keyword value in a file and return patch metadata."""
    qblade_path = Path(path)
    text = qblade_path.read_text(encoding="utf-8", errors="ignore")
    patched, old_value, count = replace_qblade_keyword_text(
        text,
        keyword,
        value,
        comment=comment,
        value_width=value_width,
        keyword_width=keyword_width,
    )
    if count != 1:
        raise ValueError(f"Could not patch {keyword} in {qblade_path}")
    qblade_path.write_text(patched, encoding="utf-8")
    return {
        "file": str(qblade_path),
        "keyword": keyword,
        "old_value": old_value,
        "new_value": render_qblade_value(value),
    }


__all__ = [
    "QBLADE_KEYVALUE_RE",
    "first_qblade_keyword_value",
    "format_qblade_keyword_line",
    "is_absolute_qblade_path",
    "parse_qblade_keyword_line",
    "parse_qblade_keywords",
    "read_qblade_float_keyword",
    "read_qblade_keyword",
    "render_qblade_value",
    "replace_qblade_keyword_file",
    "replace_qblade_keyword_text",
    "resolve_qblade_path",
    "resolve_qblade_path_text",
]

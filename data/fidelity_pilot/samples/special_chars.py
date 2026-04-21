import re
from typing import Match

# Regex patterns for code parsing
PATTERNS = {
    "function_def": re.compile(r"^(\s*)def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?\s*:", re.MULTILINE),
    "class_def": re.compile(r"^(\s*)class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:", re.MULTILINE),
    "decorator": re.compile(r"^(\s*)@(\w+(?:\.\w+)*(?:\([^)]*\))?)\s*$", re.MULTILINE),
    "import_stmt": re.compile(r"^(?:from\s+([\w.]+)\s+)?import\s+(.+)$", re.MULTILINE),
    "string_literal": re.compile(r"""(?:"(?:[^"\\\\]|\\\\.)*"|'(?:[^'\\\\]|\\\\.)*')"""),
    "comment": re.compile(r"#.*$", re.MULTILINE),
    "type_hint": re.compile(r":\s*([\w\[\], |]+)(?:\s*=)?"),
    "f_string": re.compile(r'f["\'].*?{.*?}.*?["\']'),
}

ESCAPE_MAP = {
    "\\n": "\n", "\\t": "\t", "\\r": "\r",
    "\\\\": "\\", "\\'": "'", '\\"': '"',
    "\\0": "\0", "\\a": "\a", "\\b": "\b",
}


def parse_diff_header(line: str) -> dict[str, int] | None:
    """Parse a unified diff hunk header like @@ -10,5 +12,7 @@.

    Returns:
        dict with keys: old_start, old_count, new_start, new_count
        or None if the line is not a valid hunk header.
    """
    match = re.match(r"^@@\s*-(\d+)(?:,(\d+))?\s*\+(\d+)(?:,(\d+))?\s*@@", line)
    if not match:
        return None
    return {
        "old_start": int(match.group(1)),
        "old_count": int(match.group(2) or "1"),
        "new_start": int(match.group(3)),
        "new_count": int(match.group(4) or "1"),
    }


def format_table(headers: list[str], rows: list[list], widths: list[int] | None = None) -> str:
    """Format data as an ASCII table.

    >>> print(format_table(["Name", "Score"], [["Alice", 95], ["Bob", 87]]))
    | Name  | Score |
    |-------|-------|
    | Alice |    95 |
    | Bob   |    87 |
    """
    if widths is None:
        widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    fmt = lambda vals: "| " + " | ".join(f"{str(v):<{w}}" if isinstance(v, str)
                                          else f"{v:>{w}}" for v, w in zip(vals, widths)) + " |"
    return "\n".join([fmt(headers), sep] + [fmt(row) for row in rows])

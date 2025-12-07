"""Normalize HumanEval completions to executable function bodies."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Tuple


CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
THINK_BLOCK_PATTERN = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)

SKIP_COMMENT_KEYWORDS = (
    "test",
    "example",
    "expected",
    "usage",
    "assert",
    "should",
    ">>>",
)

SKIP_MARKUP_PREFIXES = (
    "```",
    "</",
    "<code",
    "<pre",
    "<div",
    "**",
    "final answer",
    "solution:",
)

SKIP_LINE_PREFIXES = (
    "assert",
    "print(",
    "print ",
    "raise AssertionError",
)

SKIP_LINE_CONTAINS = (
    "assert(",
    "print(",
    "input(",
    "sys.exit",
    "pdb.set_trace",
)


def extract_code(raw: str) -> tuple[str, bool]:
    """Return code content and whether it originated from a fenced block."""
    matches = CODE_FENCE_PATTERN.findall(raw)
    if matches:
        return matches[0], True
    if "```" in raw:
        return raw.replace("```python", "").replace("```", ""), True
    return raw, False


def strip_thinking_blocks(raw: str) -> Tuple[str, Optional[str]]:
    """Remove <think> blocks and return cleaned text and captured reasoning."""

    if "<think>" not in raw.lower():
        return raw, None

    thoughts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        thoughts.append(match.group(1).strip())
        return ""

    cleaned = THINK_BLOCK_PATTERN.sub(_collect, raw)
    reasoning = "\n\n".join(filter(None, thoughts)) or None
    return cleaned.strip(), reasoning


def strip_imports_and_signature(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    seen_body = False
    for line in lines:
        stripped = line.strip()
        if not seen_body and (
            stripped.startswith("from ") or stripped.startswith("import ")
        ):
            continue
        if stripped.startswith("def ") and not seen_body:
            seen_body = True
            continue
        cleaned.append(line)
        if stripped:
            seen_body = True
    return "\n".join(cleaned)


def indent_body(text: str) -> str:
    dedented = textwrap.dedent(text).strip("\n")
    if not dedented:
        return ""

    body_lines = []
    for line in dedented.splitlines():
        if line.strip() == "":
            body_lines.append("")
        else:
            body_lines.append("    " + line.rstrip())
    return "\n".join(body_lines)


def normalize_body(text: str) -> str:
    dedented = textwrap.dedent(text).strip("\n")
    if not dedented:
        return ""

    lines = dedented.splitlines()

    base_indent: int | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent > 0:
            base_indent = (
                indent if base_indent is None else min(base_indent, indent)
            )
    if base_indent is None:
        base_indent = 0

    body_lines: list[str] = []
    kept_any = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            if body_lines and body_lines[-1] == "":
                continue
            body_lines.append("")
            continue

        indent = len(raw_line) - len(raw_line.lstrip())
        lower = stripped.lower()

        if indent < base_indent:
            if kept_any:
                continue
            looks_like_code = stripped.startswith(
                (
                    "return",
                    "if ",
                    "elif ",
                    "else:",
                    "for ",
                    "while ",
                    "with ",
                    "try:",
                    "except ",
                    "pass",
                    "raise ",
                    "@",
                )
            ) or any(token in stripped for token in (":", "=", "("))
            if not looks_like_code:
                continue
        if any(lower.startswith(prefix) for prefix in SKIP_MARKUP_PREFIXES):
            continue
        if any(stripped.startswith(prefix) for prefix in SKIP_LINE_PREFIXES):
            continue
        if any(token in stripped for token in SKIP_LINE_CONTAINS):
            continue
        if stripped.startswith("#") and any(
            keyword in lower for keyword in SKIP_COMMENT_KEYWORDS
        ):
            continue
        if lower.startswith("from ") or lower.startswith("import "):
            continue

        relative_indent = max(0, indent - base_indent)
        normalized = "    " + (" " * relative_indent) + stripped.rstrip()
        body_lines.append(normalized)
        kept_any = True

    while body_lines and body_lines[-1] == "":
        body_lines.pop()

    return "\n".join(body_lines)


def postprocess_completion(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    text, from_fence = extract_code(text)
    text = strip_imports_and_signature(text)
    if from_fence:
        return indent_body(text)
    return normalize_body(text)


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSONL completions",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the post-processed JSONL",
    )
    parser.add_argument(
        "--strip-thinking",
        action="store_true",
        help="Remove <think> blocks and store them under a 'thinking' key",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    processed_rows = []
    for row in iter_jsonl(input_path):
        raw_completion = row.get("completion", "")
        if args.strip_thinking:
            raw_completion, reasoning = strip_thinking_blocks(raw_completion)
            if reasoning:
                row["thinking"] = reasoning
        row["completion"] = postprocess_completion(raw_completion)
        processed_rows.append(row)

    write_jsonl(output_path, processed_rows)
    print(f"Wrote {len(processed_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

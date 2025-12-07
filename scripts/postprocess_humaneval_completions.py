"""Normalize HumanEval completions to executable function bodies."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Iterable


CODE_FENCE_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(raw: str) -> str:
    """Return the first fenced code block if present, else the raw text."""
    match = CODE_FENCE_PATTERN.findall(raw)
    if match:
        return match[0]
    if "```" in raw:
        return raw.replace("```python", "").replace("```", "")
    return raw


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
            body_lines.append("    " + line.lstrip())
    return "\n".join(body_lines)


def postprocess_completion(raw: str) -> str:
    text = raw.strip()
    if not text:
        return text
    text = extract_code(text)
    text = strip_imports_and_signature(text)
    text = indent_body(text)
    return text


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    processed_rows = []
    for row in iter_jsonl(input_path):
        row["completion"] = postprocess_completion(row.get("completion", ""))
        processed_rows.append(row)

    write_jsonl(output_path, processed_rows)
    print(f"Wrote {len(processed_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

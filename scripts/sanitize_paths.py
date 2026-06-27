#!/usr/bin/env python3
"""Replace hardcoded model paths with T2SV_MODEL_ROOT-relative paths."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_JOIN = (
    'os.path.join(os.environ.get("T2SV_MODEL_ROOT", '
    'os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))), '
)

PATTERNS = [
    (re.compile(r'["\']/data1/kaisi/rebuttal/models/([^"\']+)["\']'),
     lambda m: f'{MODEL_JOIN}"{m.group(1)}")'),
    (re.compile(r'["\']/mnt/task_runtime/models/([^"\']+)["\']'),
     lambda m: f'{MODEL_JOIN}"{m.group(1)}")'),
]

SKIP = {"paths.py", "sanitize_paths.py"}


def needs_os_import(text: str) -> bool:
    return "T2SV_MODEL_ROOT" in text and "import os" not in text.split("needs_os_import")[0]


def ensure_os_import(text: str) -> str:
    if "import os" in text or "T2SV_MODEL_ROOT" not in text:
        return text
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, "import os\n")
    return "".join(lines)


def process_file(path: Path) -> bool:
    text = path.read_text()
    original = text
    for pattern, repl in PATTERNS:
        text = pattern.sub(repl, text)
    if text != original:
        text = ensure_os_import(text)
        path.write_text(text)
        return True
    return False


def main() -> None:
    changed = []
    for path in ROOT.rglob("*.py"):
        if path.name in SKIP:
            continue
        if process_file(path):
            changed.append(path.relative_to(ROOT))
    print(f"Updated {len(changed)} files")
    for p in changed:
        print(f"  - {p}")


if __name__ == "__main__":
    main()

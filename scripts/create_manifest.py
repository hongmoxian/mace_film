#!/usr/bin/env python3
"""Write file sizes and SHA256 checksums for this release package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "MANIFEST.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)))

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "size_bytes", "sha256"])
        writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} files")


if __name__ == "__main__":
    main()

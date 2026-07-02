#!/usr/bin/env python3
"""Subset an extxyz file by simple Atoms.info equality filters."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import io


def parse_filter(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Filter must be KEY=VALUE, got {raw!r}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Empty filter key in {raw!r}")
    return key, value


def matches(value, expected: str) -> bool:
    if str(value) == expected:
        return True
    try:
        return float(value) == float(expected)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter", action="append", default=[], help="Atoms.info filter, e.g. state_id=0")
    args = parser.parse_args()

    filters = [parse_filter(item) for item in args.filter]
    frames = io.read(args.input, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    kept = []
    for atoms in frames:
        ok = True
        for key, expected in filters:
            if key not in atoms.info or not matches(atoms.info[key], expected):
                ok = False
                break
        if ok:
            kept.append(atoms)

    if not kept:
        raise ValueError(f"No frames matched filters {filters}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    io.write(output, kept, format="extxyz")
    print(f"Wrote {len(kept)} / {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()

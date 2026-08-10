#!/usr/bin/env python3
"""Validate prerequisites without modifying the environment or dataset."""

from __future__ import annotations

from pathlib import Path

from ibibio_tts.environment import run_checks


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    checks = run_checks(project_root)
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

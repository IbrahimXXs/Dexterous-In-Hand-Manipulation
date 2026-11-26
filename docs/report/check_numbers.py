#!/usr/bin/env python3
"""Check that every number listed in docs/report/NUMBERS.md still appears in docs/report/REPORT.md.

For each table row of NUMBERS.md, the numeric tokens of the "Number as written" cell (signs, decimals,
thousands separators; Unicode minus normalised) must occur somewhere in REPORT.md. A row whose number was
edited out of the report is reported, so the source table never lists numbers the report no longer uses.
This is one direction only: a number added to the report still needs its own row, added by hand.

    python docs/report/check_numbers.py
Exit code 1 if any row's number is missing from the report.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUM = re.compile(r"[−+-]?\d[\d,]*(?:\.\d+)?")


def norm(s: str) -> str:
  return s.replace("−", "-").replace(",", "")


def main() -> int:
  report = norm((HERE / "REPORT.md").read_text())
  rows = [line for line in (HERE / "NUMBERS.md").read_text().splitlines()
          if line.startswith("| ") and not line.startswith("| Number") and not line.startswith("|---")]
  bad = []
  for row in rows:
    cell = row.split("|")[1].strip()
    missing = [t for t in (norm(x) for x in NUM.findall(cell)) if t.lstrip("+-") not in report]
    if missing:
      bad.append(f"{cell[:80]}  -> not in REPORT.md: {', '.join(missing)}")
  print("\n".join(bad) if bad else f"OK: {len(rows)} NUMBERS.md rows, every listed number appears in REPORT.md")
  return 1 if bad else 0


if __name__ == "__main__":
  raise SystemExit(main())

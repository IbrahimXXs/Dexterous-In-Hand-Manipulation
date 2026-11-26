#!/usr/bin/env python3
"""Check every relative link and anchor in the repository's Markdown files (no network).

For each [text](target) / ![alt](target) / <img src="target"> / srcset="target" with a relative target:
  * the file or directory must exist (relative to the Markdown file);
  * a #anchor into a Markdown file must match one of its headings, slugged as GitHub does
    (lower case, punctuation removed except '-' and '_', spaces -> '-', duplicates get -1, -2, ...).
External links (http, https, mailto) are listed but not fetched.

    python scripts/check_links.py [files ...]     # default: every tracked *.md file
Exit code 1 if any link is broken.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
HTML = re.compile(r"""(?:src|srcset|href)=["']([^"']+)["']""")
FENCE = re.compile(r"^(```|~~~)")
# Links in course-provided files that point into the course's own tree, which is not part of this repository.
# The course files are kept unchanged, so these are reported as known external links, not as broken.
KNOWN_EXTERNAL = {("docs/lesson.md", "../etc/01_Lessons/Class07_Cube_Visual_Feedback/lesson.md")}


def slugs(md: Path) -> set[str]:
  out, seen, fence = set(), {}, False
  for line in md.read_text(errors="replace").splitlines():
    if FENCE.match(line.strip()):
      fence = not fence
      continue
    m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
    if fence or not m:
      continue
    text = re.sub(r"<[^>]+>", "", m.group(2))
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> their text
    s = re.sub(r"[^\w\- ]", "", text.strip().lower().replace("`", "")).replace(" ", "-")
    n = seen.get(s, 0)
    seen[s] = n + 1
    out.add(s if n == 0 else f"{s}-{n}")
  return out


def targets(md: Path):
  fence = False
  for i, line in enumerate(md.read_text(errors="replace").splitlines(), 1):
    if FENCE.match(line.strip()):
      fence = not fence
      continue
    if fence:
      continue
    line = re.sub(r"`[^`]*`", "", line)  # inline code is not a link
    for m in LINK.finditer(line):
      yield i, m.group(1)
    for m in HTML.finditer(line):
      for part in m.group(1).split(","):
        yield i, part.strip().split(" ")[0]


def rel(p: Path) -> str:
  """Path relative to the repository root when inside it, else as given."""
  try:
    return str(p.relative_to(ROOT))
  except ValueError:
    return str(p)


def check(files: list[Path]) -> list[str]:
  bad, cache = [], {}
  for md in files:
    for line, t in targets(md):
      if re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I):  # http:, https:, mailto: ...
        continue
      if (rel(md), t) in KNOWN_EXTERNAL:
        continue
      path, _, anchor = t.partition("#")
      dest = md if path == "" else (md.parent / path).resolve()
      if not dest.exists():
        bad.append(f"{rel(md)}:{line}: missing {t}")
        continue
      if anchor and dest.suffix == ".md":
        if dest not in cache:
          cache[dest] = slugs(dest)
        if anchor.lower() not in cache[dest]:
          bad.append(f"{rel(md)}:{line}: no heading #{anchor} in {rel(dest)}")
  return bad


def main() -> int:
  if len(sys.argv) > 1:
    files = [Path(a).resolve() for a in sys.argv[1:]]
  else:
    ls = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = [ROOT / f for f in ls.split()]
  bad = check(files)
  print("\n".join(bad) if bad else f"OK: {len(files)} Markdown files, no broken relative link or anchor")
  return 1 if bad else 0


if __name__ == "__main__":
  raise SystemExit(main())

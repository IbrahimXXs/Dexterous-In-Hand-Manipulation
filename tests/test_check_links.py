"""Tests for scripts/check_links.py (relative links and GitHub-style heading anchors, no network)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_links", ROOT / "scripts" / "check_links.py")
check_links = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_links)


def test_slugs_follow_github_rules(tmp_path):
  md = tmp_path / "a.md"
  md.write_text("# Title\n## 8. Actuator audit → C5-EFFORT → C5-HW-V2\n## `code` and [link](x.md)\n"
                "## Title\n```\n# not a heading\n```\n")
  s = check_links.slugs(md)
  assert "title" in s and "title-1" in s  # duplicates get -1
  assert "8-actuator-audit--c5-effort--c5-hw-v2" in s  # punctuation dropped, spaces kept as hyphens
  assert "code-and-link" in s
  assert "not-a-heading" not in s  # fenced code is not a heading


def test_check_reports_missing_files_and_anchors(tmp_path):
  (tmp_path / "b.md").write_text("# Real heading\n")
  (tmp_path / "img.png").write_bytes(b"")
  md = tmp_path / "a.md"
  md.write_text("[ok](b.md#real-heading) ![i](img.png) [ext](https://example.org/x) [self](#top)\n"
                "# Top\n"
                "[bad file](missing.md) [bad anchor](b.md#nope) `[code](not-a-link.md)`\n"
                '<img src="gone.png">\n'
                "```\n[fenced](also-not-a-link.md)\n```\n")
  bad = check_links.check([md])
  assert len(bad) == 3, bad
  assert any("missing missing.md" in b for b in bad)
  assert any("no heading #nope" in b for b in bad)
  assert any("missing gone.png" in b for b in bad)


def test_known_external_course_link_is_skipped_only_where_listed():
  # the course lesson links into the course tree, which is not part of this repository
  lesson = check_links.ROOT / "docs" / "lesson.md"
  assert check_links.check([lesson]) == []
  assert ("docs/lesson.md", "../etc/01_Lessons/Class07_Cube_Visual_Feedback/lesson.md") in check_links.KNOWN_EXTERNAL

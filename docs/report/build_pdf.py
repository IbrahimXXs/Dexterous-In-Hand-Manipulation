#!/usr/bin/env python3
"""Render docs/report/REPORT.md to PDF (Markdown -> HTML -> headless Chrome print).

The PDF is a build artefact and is written to outputs/report/ (git-ignored), not committed:
the repository keeps only the Markdown source and the PNG/JPG figures it links to.

    python docs/report/build_pdf.py [--out outputs/report/REPORT.pdf] [--chrome google-chrome]
Needs: python-markdown, Google Chrome / Chromium.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CSS = """
@page { size: A4; margin: 14mm 13mm 15mm 13mm; }
body { font-family: "DejaVu Serif", "Times New Roman", serif; font-size: 8.5pt; line-height: 1.22; color: #111; }
.cols { column-count: 2; column-gap: 6mm; column-fill: balance; }
.head { column-span: all; }
h1 { font-size: 14pt; text-align: center; margin: 0 0 3pt 0; }
h2 { font-size: 10.4pt; margin: 7pt 0 2pt 0; border-bottom: 0.5pt solid #999; break-after: avoid; }
h3 { font-size: 9.4pt; margin: 5pt 0 1.5pt 0; break-after: avoid; }
p, li { margin: 1.1pt 0; text-align: justify; hyphens: auto; }
ul, ol { margin: 1.5pt 0 1.5pt 12pt; padding: 0; }
table { border-collapse: collapse; margin: 3pt 0; font-size: 7.1pt; width: 100%; table-layout: fixed; break-inside: avoid; }
td, th { overflow-wrap: anywhere; hyphens: auto; }
figure.wide { column-span: all; margin: 4pt 0 6pt 0; break-inside: avoid; }
figure.wide img { max-width: 80%; max-height: 45mm; }
table.wide { column-span: all; width: 96%; margin: 4pt auto; font-size: 7.4pt; }
figure.col { margin: 3pt 0 5pt 0; break-inside: avoid; }
figure.col img { max-height: 42mm; }
figure.col figcaption, figure.wide figcaption { font-size: 8.2pt; font-style: italic; text-align: justify; margin: 2pt 4% 0 4%; }
th, td { border: 0.4pt solid #888; padding: 1pt 2.5pt; vertical-align: top; }
th { background: #eee; }
img { display: block; margin: 3pt auto 0 auto; max-width: 100%; break-inside: avoid; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 7.4pt; }
.refs p { text-align: left; hyphens: manual; overflow-wrap: anywhere; font-size: 7.6pt; line-height: 1.18; margin: 1pt 0; }
blockquote { margin: 3pt 4pt; padding: 2pt 5pt; border-left: 2pt solid #c60; background: #fff7ee; }
"""


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--md", type=Path, default=HERE / "REPORT.md")
  ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "report" / "REPORT.pdf")
  ap.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome")
  a = ap.parse_args()
  text = a.md.read_text()
  # links to NUMBERS.md / other markdown stay as text in print; images resolve relative to the Markdown file
  body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
  body = re.sub(r'src="(?!https?:|/)([^"]+)"', lambda m: f'src="{(a.md.parent / m.group(1)).resolve().as_uri()}"', body)
  # figures: image paragraph + its "Figure N:" caption paragraph; they span both columns, except an image
  # whose alt text starts with "(column)", which stays inside its column
  def figure(m):
    img, cap = m.group(1), m.group(2)
    cls = "col" if 'alt="(column)' in img else "wide"
    img = img.replace('alt="(column) ', 'alt="')
    return "<figure class='" + cls + "'>" + img + "<figcaption>" + cap + "</figcaption></figure>"
  body = re.sub(r"<p>(<img [^>]+>)</p>\s*<p><em>(Figure \d+:.*?)</em></p>", figure, body, flags=re.S)
  # tables with six or more columns span both columns
  def widen(m):
    ncol = m.group(0).split("</tr>")[0].count("<th")
    return m.group(0).replace("<table>", "<table class='wide'>", 1) if ncol >= 6 else m.group(0)
  body = re.sub(r"<table>.*?</table>", widen, body, flags=re.S)
  # two-column workshop layout: title, byline and abstract span both columns
  cut = body.find("<h2>1.")
  head, rest = (body[:cut], body[cut:]) if cut > 0 else ("", body)
  # references and software URLs: left-aligned, so long DOIs and URLs do not stretch the line spacing
  if "<h2>References</h2>" in rest:
    rest = rest.replace("<h2>References</h2>", "<h2>References</h2><div class='refs'>", 1) + "</div>"
  html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>{CSS}</style></head>"
          f"<body><div class='head'>{head}</div><div class='cols'>{rest}</div></body></html>")
  a.out.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory() as td:
    h = Path(td) / "report.html"
    h.write_text(html)
    subprocess.run([a.chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={a.out}", h.as_uri()], check=True, capture_output=True)
  print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

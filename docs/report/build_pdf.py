#!/usr/bin/env python3
"""Export docs/report/REPORT.md to PDF (Markdown -> HTML -> headless Chrome print).

The PDF is written to outputs/report/REPORT.pdf (git-ignored; only the Markdown source is versioned).
Needs the `markdown` package and Google Chrome / Chromium.

    python docs/report/build_pdf.py [--out outputs/report/REPORT.pdf] [--chrome google-chrome]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

CSS = """
@page { size: A4; margin: 14mm 13mm 15mm 13mm; }
body { font-family: 'DejaVu Serif', 'Times New Roman', serif; font-size: 8.6pt; line-height: 1.27; color: #111;
       column-count: 2; column-gap: 6mm; }
h1 { column-span: all; font-size: 15pt; text-align: center; margin: 0 0 6pt 0; }
h2 { font-size: 10.2pt; margin: 8pt 0 2pt 0; border-bottom: 0.5pt solid #999; break-after: avoid; }
h3 { font-size: 9.2pt; margin: 5pt 0 1.5pt 0; break-after: avoid; }
p, li { text-align: justify; margin: 2pt 0; hyphens: auto; }
ul, ol { margin: 1.5pt 0 1.5pt 11pt; padding: 0; }
table { border-collapse: collapse; margin: 4pt 0; font-size: 7pt; line-height: 1.18; break-inside: avoid; width: 100%; }
th, td { border: 0.4pt solid #999; padding: 1pt 2.5pt; vertical-align: top; }
th { background: #eee; }
img { display: block; max-width: 100%; max-height: 62mm; margin: 4pt auto 1pt auto; break-inside: avoid; }
p.cap { font-size: 7.5pt; color: #333; text-align: left; }
blockquote { border-left: 2pt solid #c00; margin: 3pt 0; padding: 2pt 5pt; background: #fff4f4; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.4pt; overflow-wrap: anywhere; }
"""


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--src", type=Path, default=HERE / "REPORT.md")
  ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "report" / "REPORT.pdf")
  ap.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome")
  a = ap.parse_args()
  import markdown

  text = a.src.read_text()
  text = re.sub(r"^\*Draft project report\..*\*\n", "", text, flags=re.M)  # repo-only note
  body = markdown.markdown(text, extensions=["tables", "sane_lists"])
  body = re.sub(r"<p><em>(Figure \d+:.*?)</em></p>", r'<p class="cap"><em>\1</em></p>', body, flags=re.S)  # captions
  # image paths are relative to docs/report/: make them absolute file URLs for the printer
  body = re.sub(r'src="([^":]+)"', lambda m: f'src="file://{(a.src.parent / m.group(1)).resolve()}"', body)
  html = f"<!doctype html><html><head><meta charset='utf-8'><title>Report</title><style>{CSS}</style></head><body>{body}</body></html>"
  a.out.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory() as td:
    h = Path(td) / "report.html"
    h.write_text(html)
    subprocess.run([a.chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
                    "--no-pdf-header-footer", f"--print-to-pdf={a.out}", h.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

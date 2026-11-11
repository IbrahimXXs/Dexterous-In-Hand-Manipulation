#!/usr/bin/env python3
"""Generate the deterministic MBZUAI 2 x 3 MuJoCo cube texture atlas."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "mbzuai_cube.png"
CELL_WIDTH = 600
CELL_HEIGHT = 600

# PIL rows run top-to-bottom while OBJ V coordinates run bottom-to-top.
# These cells match dex_cube.obj exactly.
CELLS = {
  "-Z": (0, 0, "I", (245, 125, 30), "#171717"),
  "+Y": (1, 0, "Z", (25, 160, 85), "white"),
  "+X": (0, 1, "B", (220, 40, 50), "white"),
  "-X": (1, 1, "U", (35, 95, 210), "white"),
  "-Y": (0, 2, "A", (245, 245, 240), "#171717"),
  "+Z": (1, 2, "M", (255, 215, 0), "#171717"),
}


def _font(size: int) -> ImageFont.FreeTypeFont:
  for candidate in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
  ):
    if Path(candidate).is_file():
      return ImageFont.truetype(candidate, size)
  raise FileNotFoundError("Install DejaVu Sans or Liberation Sans to build the atlas.")


def main() -> int:
  image = Image.new("RGB", (2 * CELL_WIDTH, 3 * CELL_HEIGHT), "white")
  draw = ImageDraw.Draw(image)
  letter_font = _font(360)
  axis_font = _font(30)

  for face, (column, row, letter, color, ink) in CELLS.items():
    left = column * CELL_WIDTH
    top = row * CELL_HEIGHT
    right = left + CELL_WIDTH
    bottom = top + CELL_HEIGHT
    draw.rectangle((left, top, right, bottom), fill=color)
    draw.rectangle(
      (left + 14, top + 14, right - 14, bottom - 14),
      outline=ink,
      width=10,
    )
    box = draw.textbbox((0, 0), letter, font=letter_font, stroke_width=2)
    width = box[2] - box[0]
    height = box[3] - box[1]
    position = (
      left + (CELL_WIDTH - width) / 2 - box[0],
      top + (CELL_HEIGHT - height) / 2 - box[1] - 8,
    )
    draw.text(
      position,
      letter,
      font=letter_font,
      fill=ink,
      stroke_width=2,
      stroke_fill=ink,
    )
    draw.text(
      (left + 32, bottom - 62),
      f"MBZUAI  {face}",
      font=axis_font,
      fill=ink,
    )

  image.save(OUTPUT, optimize=True)
  print(f"wrote {OUTPUT}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

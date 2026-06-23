"""Shared terminal-style screenshot rendering with CJK font support."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# macOS CJK-capable fonts (Menlo does not render Chinese)
CJK_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]

BG = (22, 22, 28)
TITLE_BAR = (45, 45, 52)
GREEN = (100, 255, 120)
WHITE = (220, 220, 225)
YELLOW = (255, 220, 100)
ORANGE = (255, 170, 80)
CYAN = (100, 220, 255)
GRAY = (140, 140, 150)


def get_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def pick_line_color(line: str) -> tuple[int, int, int]:
    if line.startswith("==="):
        return ORANGE
    if line.startswith("$") or "python -m" in line:
        return GREEN
    if "✓" in line or "✅" in line or "OK" in line or "Merge completed" in line:
        return GREEN
    if line.startswith("23:") or "INFO" in line:
        return CYAN
    if "Accuracy" in line or "latency" in line.lower() or "p50" in line:
        return YELLOW
    if "合计" in line or "copaw-dpo 全量测试" in line:
        return ORANGE if "copaw" in line else YELLOW
    if "[Pipeline]" in line or "[Validator]" in line or "[Repair]" in line:
        return CYAN
    if line.startswith("==") or line.startswith("$") or line.startswith("  $"):
        return GREEN
    if "Step" in line and "loss=" in line:
        return YELLOW
    if "Summary" in line:
        return ORANGE
    return WHITE


def render_terminal(
    lines: list[str],
    png_path: Path | str,
    title: str,
    *,
    font_size: int = 13,
    line_h: int = 21,
    pad_x: int = 28,
    pad_y: int = 36,
    bg_color: tuple[int, int, int] = BG,
    show_title_bar: bool = True,
) -> None:
    png_path = Path(png_path)
    font = get_cjk_font(font_size)
    title_font = get_cjk_font(11)

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_w = max((_text_width(probe, line, font) for line in lines if line), default=400)
    img_w = int(max(max_w + pad_x * 2, 480))
    title_bar_h = 28 if show_title_bar else 0
    img_h = len(lines) * line_h + pad_y + title_bar_h

    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    if show_title_bar:
        draw.rectangle([(0, 0), (img_w, title_bar_h)], fill=TITLE_BAR)
        draw.text((14, 7), f"  {title}", fill=GRAY, font=title_font)
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            draw.ellipse([(img_w - 58 + i * 18, 9), (img_w - 46 + i * 18, 21)], fill=c)

    y = pad_y
    for line in lines:
        if not line:
            y += line_h // 2
            continue
        draw.text((pad_x, y), line, fill=pick_line_color(line), font=font)
        y += line_h

    draw.rectangle([(1, 1), (img_w - 2, img_h - 2)], outline=(55, 55, 65), width=1)
    img.save(png_path)
    print(f"Saved: {png_path} ({img_w}x{img_h})")

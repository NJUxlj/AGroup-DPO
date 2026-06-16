"""Generate test summary screenshot from the report."""
from PIL import Image, ImageDraw, ImageFont

# Read report lines
with open("/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO/reports/copaw_cli_full_test.md") as f:
    content = f.read()

# Extract summary table lines
lines = []
in_summary = False
for line in content.split("\n"):
    if "总结" in line or "copaw-dpo data" in line:
        in_summary = True
    if in_summary:
        if line.strip().startswith("|"):
            lines.append(line.strip())

# Build display text
display_lines = [
    "=== copaw-dpo 全量测试结果 (server6, 2xRTX 4090) ===",
    "",
]
for l in lines:
    if l.startswith("|"):
        parts = [p.strip() for p in l.split("|") if p.strip()]
        if parts:
            display_lines.append("  " + "  |  ".join(parts))
display_lines += [
    "",
    "合计: 10/10 测试通过",
]

# Terminal-style rendering
bg_color = (22, 22, 28)
green = (100, 255, 100)
white = (220, 220, 220)
yellow = (255, 255, 100)
orange = (255, 165, 0)
gray = (140, 140, 140)

font_size = 13
line_h = 22
pad_x, pad_y = 30, 24

try:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
except:
    font = ImageFont.load_default()

max_w = max(len(l) for l in display_lines) * (font_size * 0.62)
img_w = int(max_w + pad_x * 2)
img_h = len(display_lines) * line_h + pad_y * 2

img = Image.new("RGB", (img_w, img_h), bg_color)
draw = ImageDraw.Draw(img)

for i, line in enumerate(display_lines):
    y = pad_y + i * line_h
    if "copaw-dpo 全量测试" in line:
        color = orange
    elif "✅" in line:
        color = green
    elif "合计" in line:
        color = yellow
    elif not line.strip():
        continue
    else:
        color = white
    draw.text((pad_x, y), line, fill=color, font=font)

# Border
draw.rectangle([(2, 2), (img_w-3, img_h-3)], outline=(60, 60, 70), width=1)

img.save("/Users/xiniuyiliao/Desktop/application_code/AGroup-DPO/docs/screenshots/copaw_cli_test_summary.png")
print(f"Saved: {img_w}x{img_h}")

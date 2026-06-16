"""Generate a PNG title banner for README — with CJK support and proper centering."""
from PIL import Image, ImageDraw, ImageFont

W, H = 800, 160
img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background
draw.rounded_rectangle(
    [(20, 10), (W - 20, H - 10)],
    radius=20,
    fill=(255, 248, 240, 255),
    outline=(255, 120, 0, 60),
    width=2,
)

# Fonts
# English title — bold, impactful
font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Black.ttf", 62)
# Chinese subtitle — Arial Unicode covers CJK
font_sub = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 20)

# Colors
orange = (255, 80, 0)
shadow = (200, 200, 200, 80)

cx, cy = W // 2, H // 2  # canvas center

# Shadow (slightly offset)
draw.text((cx + 3, 52 + 3), "AGroup DPO", fill=shadow, font=font_title, anchor="mm")
# Main title — anchor="mm" = middle-middle, perfectly centered
draw.text((cx, 52), "AGroup DPO", fill=orange, font=font_title, anchor="mm")

# Decorative lines on both sides of subtitle
# 设置装饰线的垂直位置（Y轴坐标）
line_y = 116
# 设置装饰线的颜色，格式为(R, G, B, Alpha)，这里是一个带一定透明度的橙色
line_color = (255, 120, 0, 180)
# 绘制左侧的装饰矩形线：左上角坐标为 (160, line_y)，右下角坐标为 (300, line_y + 6)
draw.rectangle([(160, line_y), (300, line_y + 5)], fill=line_color)
# 绘制右侧的装饰矩形线：左上角坐标为 (500, line_y)，右下角坐标为 (640, line_y + 6)，与左侧对称
draw.rectangle([(500, line_y), (640, line_y + 5)], fill=line_color)

# Subtitle
draw.text((cx, line_y - 16), "保险场景  ·  偏好对齐  ·  工程实践", fill=(100, 100, 100), font=font_sub, anchor="mm")

img.save("title.png")
print(f"Saved title.png ({W}x{H})")

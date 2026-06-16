"""Generate a PNG title banner for the README."""
from PIL import Image, ImageDraw, ImageFont

W, H = 800, 160
img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Background: subtle rounded rectangle
draw.rounded_rectangle([(20, 10), (W-20, H-10)], radius=20, fill=(255, 248, 240, 255), outline=(255, 120, 0, 60), width=2)

# Main title
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Black.ttf", 60)
except:
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 60)
    except:
        font_title = ImageFont.load_default()

# Subtitle font
try:
    font_sub = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
except:
    font_sub = font_title

# Draw title with multiple layers for depth effect
orange_main = (255, 80, 0)
orange_dark = (200, 50, 0)
orange_light = (255, 140, 0)

# Shadow
draw.text((403, 48), "AGroup DPO", fill=(200, 200, 200, 100), font=font_title)
# Main text
draw.text((400, 45), "AGroup DPO", fill=orange_main, font=font_title)

# Decorative lines
for i, (x, w) in enumerate([(200, 60), (540, 60)]):
    draw.rectangle([(x, 110), (x+w, 114)], fill=(255, 120, 0, 180))

# Subtitle
draw.text((W//2, 122), "保险场景  ·  偏好对齐  ·  工程实践", fill=(120, 120, 120), font=font_sub, anchor="ma")

img.save("title.png")
print(f"Saved title.png ({W}x{H})")

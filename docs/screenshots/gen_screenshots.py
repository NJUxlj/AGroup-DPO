"""Generate terminal-style screenshots for README."""
from PIL import Image, ImageDraw, ImageFont
import os

def render_terminal(output_file: str, png_file: str, title: str):
    with open(output_file) as f:
        lines = f.readlines()
    
    # Terminal styling
    bg_color = (30, 30, 30)  # Dark terminal background
    green = (0, 255, 0)
    white = (255, 255, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    orange = (255, 165, 0)
    
    font_size = 14
    line_height = 20
    padding_x = 24
    padding_y = 20
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate image size
    max_line_width = max(len(line) for line in lines) * (font_size * 0.62)
    img_width = int(max_line_width + padding_x * 2 + 20)
    img_height = len(lines) * line_height + padding_y * 2 + 30
    
    img = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Title bar
    title_color = (60, 60, 60)
    draw.rectangle([(0, 0), (img_width, 28)], fill=title_color)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except:
        title_font = font
    draw.text((12, 6), f"  Terminal — {title}", fill=(200, 200, 200), font=title_font)
    
    # Red/yellow/green dots
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([(img_width - 60 + i*18, 8), (img_width - 48 + i*18, 20)], fill=color)
    
    y = padding_y + 12
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            y += line_height
            continue
        
        # Color coding
        if line.startswith('==='):
            color = orange
        elif line.startswith('$') or line.startswith('  $'):
            color = green
        elif '[Pipeline]' in line or '[Validator]' in line or '[Repair]' in line:
            color = cyan
        elif 'Step' in line and 'loss=' in line:
            color = yellow
        elif '✓' in line:
            color = green
        elif 'Summary' in line:
            color = orange
        else:
            color = white
        
        draw.text((padding_x, y), line, fill=color, font=font)
        y += line_height
    
    img.save(png_file)
    print(f"Saved: {png_file} ({img_width}x{img_height})")

base = os.path.dirname(os.path.abspath(__file__))

render_terminal(
    os.path.join(base, "dpo_data_output.txt"),
    os.path.join(base, "dpo_data_gen.png"),
    "DPO Data Pipeline — server6"
)

render_terminal(
    os.path.join(base, "dpo_train_output.txt"),
    os.path.join(base, "dpo_train.png"),
    "DPO Smoke Training — server6"
)

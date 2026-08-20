import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out_dir = Path("assets/textures")
out_dir.mkdir(parents=True, exist_ok=True)


def load_font(size):
    for name in ("arialbd.ttf", "ariblk.ttf", "arial.ttf", "segoeuib.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_label(text, bg, fg, path, size=(512, 256), border=None):
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    if border:
        draw.rectangle([6, 6, size[0] - 7, size[1] - 7], outline=border, width=10)
    font = load_font(180)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while (tw > size[0] - 40 or th > size[1] - 40) and font.size > 20:
        font = load_font(font.size - 8)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - tw) / 2 - bbox[0]
    y = (size[1] - th) / 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=fg)
    img.save(path)
    print("wrote", path)


make_label("RABAH", bg=(20, 20, 24), fg=(240, 240, 245), path=out_dir / "label_rabah.png",
           border=(90, 160, 220))
make_label("TRASH", bg=(245, 240, 210), fg=(20, 20, 20), path=out_dir / "label_trash.png",
           border=(200, 30, 30))

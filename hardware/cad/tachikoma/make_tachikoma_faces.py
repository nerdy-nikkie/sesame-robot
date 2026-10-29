"""Draw the Tachikoma OLED faces (128x64, 1-bit) and print them as image2cpp-style C arrays.

The Tachikoma's eye is a white "bowling ball" with a dark iris ring and three pinholes.
Faces generated:
  tachikoma            idle eye; frames _1.._5 glance left and right (played boomerang)
  tachikoma_blink      blink (half, closed, half)
  tachikoma_happy      closed, smiling eye
  tachikoma_surprised  wide eye with tiny pinholes

usage:
    pip install pillow
    python hardware/cad/tachikoma/make_tachikoma_faces.py firmware/tachikoma-faces.inc docs/images/sesamefaces/tachikoma.png
"""
import math
import sys
from PIL import Image, ImageDraw

W, H, SS = 128, 64, 4          # supersample for smooth circles
CX, CY = 64, 32

def canvas():
    img = Image.new("L", (W * SS, H * SS), 0)
    return img, ImageDraw.Draw(img)

def circle(d, cx, cy, r, fill):
    d.ellipse([(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS], fill=fill)

def ellipse(d, cx, cy, rx, ry, fill):
    d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=fill)

def finish(img):
    return img.resize((W, H), Image.LANCZOS).point(lambda v: 255 if v >= 128 else 0).convert("1")

def eye(look=0.0, r=29.0, ring=(23.5, 26.0), hole_r=3.2, hole_d=9.0, squash=1.0):
    """Eye ball with iris ring + three pinholes; `look` shifts the iris in pixels."""
    img, d = canvas()
    ellipse(d, CX, CY, r, r * squash, 255)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([(CX - r) * SS, (CY - r * squash) * SS, (CX + r) * SS, (CY + r * squash) * SS], fill=255)
    iris, di = canvas()
    ix = CX + look
    ellipse(di, ix, CY, ring[1], ring[1] * squash, 255)
    ellipse(di, ix, CY, ring[0], ring[0] * squash, 0)
    for a in (-90, 30, 150):
        hx = ix + hole_d * math.cos(math.radians(a))
        hy = CY + hole_d * math.sin(math.radians(a)) * squash
        ellipse(di, hx, hy, hole_r, max(hole_r * squash, 0.6), 255)
    # iris features are black, only inside the eye ball
    img.paste(0, mask=Image.composite(iris, Image.new("L", img.size, 0), mask))
    return finish(img)

def closed(thickness=4, smile=False):
    img, d = canvas()
    if smile:   # happy "^" arc
        d.arc([(CX - 26) * SS, (CY - 14) * SS, (CX + 26) * SS, (CY + 30) * SS], 200, 340, fill=255,
              width=thickness * SS)
    else:
        d.rounded_rectangle([(CX - 28) * SS, (CY - thickness / 2) * SS, (CX + 28) * SS, (CY + thickness / 2) * SS],
                            radius=thickness * SS / 2, fill=255)
    return finish(img)

FACES = {
    "tachikoma": eye(),
    "tachikoma_1": eye(look=-4),
    "tachikoma_2": eye(look=-7),
    "tachikoma_3": eye(look=0),
    "tachikoma_4": eye(look=4),
    "tachikoma_5": eye(look=7),
    "tachikoma_blink": eye(squash=0.45),
    "tachikoma_blink_1": closed(),
    "tachikoma_blink_2": eye(squash=0.45),
    "tachikoma_happy": closed(thickness=5, smile=True),
    "tachikoma_surprised": eye(r=31, ring=(27.5, 29.0), hole_r=2.0, hole_d=8.0),
}

def to_c(name, img):
    px = img.load()
    data = []
    for y in range(H):
        for xb in range(W // 8):
            b = 0
            for bit in range(8):
                if px[xb * 8 + bit, y]:
                    b |= 0x80 >> bit
            data.append(b)
    lines = [f"// '{name}', 128x64px", f"const unsigned char epd_bitmap_{name} [] PROGMEM = {{"]
    for i in range(0, len(data), 16):
        row = ", ".join(f"0x{v:02x}" for v in data[i:i + 16])
        lines.append("\t" + row + ("," if i + 16 < len(data) else ""))
    lines.append("};")
    return "\n".join(lines)

if __name__ == "__main__":
    out_inc, out_png = sys.argv[1], sys.argv[2]
    with open(out_inc, "w") as f:
        f.write("\n".join(to_c(n, im) for n, im in FACES.items()) + "\n")
    # preview sheet (scaled 2x, frames in a grid)
    cols = 4
    rows = (len(FACES) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (W * 2 + 8) + 8, rows * (H * 2 + 22) + 8), (30, 34, 40))
    d = ImageDraw.Draw(sheet)
    for i, (n, im) in enumerate(FACES.items()):
        x, y = 8 + (i % cols) * (W * 2 + 8), 8 + (i // cols) * (H * 2 + 22)
        sheet.paste(im.convert("RGB").resize((W * 2, H * 2), Image.NEAREST), (x, y))
        d.text((x, y + H * 2 + 4), n, fill=(200, 210, 220))
    sheet.save(out_png)
    print(f"wrote {len(FACES)} frames to {out_inc} and preview {out_png}")

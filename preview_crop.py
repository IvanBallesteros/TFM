from PIL import Image
from pathlib import Path

proj = Path(r"C:\Users\ibf\Desktop\TFM\Nou projecte")
val = proj / "VAL"

# find first image under VAL (recursively)
imgs = list(val.rglob("*.png"))
if not imgs:
    imgs = list(val.rglob("*.jpg"))
if not imgs:
    raise SystemExit("No images found in VAL")

p = imgs[0]
print(f"Selected image: {p}")

img = Image.open(p).convert("RGB")
out_dir = proj / "TFM"
out_dir.mkdir(parents=True, exist_ok=True)
orig = out_dir / "preview_original.png"
crop = out_dir / "preview_cropped.png"
img.save(orig)

h = img.height
crop_px = 120
crop_margin = min(crop_px, max((h - 1) // 2, 0))
if crop_margin <= 0:
    cropped = img
else:
    cropped = img.crop((0, crop_margin, img.width, h - crop_margin))

cropped.save(crop)
print(f"Saved: {orig}")
print(f"Saved: {crop}")

from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_IMAGE = PROJECT_DIR / "experiments/product_text_ocr/input/nek.jpg"
OUTPUT_DIR = PROJECT_DIR / "experiments/product_text_ocr/crops"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crop area for product title/spec column in the uploaded NEK sample
X1 = 250
Y1 = 170
X2 = 580
Y2 = 880

ROW_COUNT = 21

img = Image.open(INPUT_IMAGE).convert("RGB")

# Save full product column crop
column_crop = img.crop((X1, Y1, X2, Y2))
column_crop.save(OUTPUT_DIR / "nek_product_text_column.png")

# Save preview with rectangle
preview = img.copy()
draw = ImageDraw.Draw(preview)
draw.rectangle((X1, Y1, X2, Y2), outline="red", width=3)
preview.save(OUTPUT_DIR / "nek_product_text_preview.png")

# Save each row crop
row_height = (Y2 - Y1) / ROW_COUNT

for row in range(1, ROW_COUNT + 1):
    row_y1 = int(Y1 + (row - 1) * row_height)
    row_y2 = int(Y1 + row * row_height)

    crop = img.crop((X1, row_y1, X2, row_y2))
    crop.save(OUTPUT_DIR / f"nek_product_row_{row:02d}.png")

print("Done.")
print(f"Saved crops to: {OUTPUT_DIR}")
print(f"Column crop: {OUTPUT_DIR / 'nek_product_text_column.png'}")
print(f"Preview: {OUTPUT_DIR / 'nek_product_text_preview.png'}")
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageFilter

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_IMAGE = PROJECT_DIR / "experiments/product_text_ocr/input/nek.jpg"
OUTPUT_ROOT = PROJECT_DIR / "experiments/product_text_ocr/crops_v2"

PREVIEW_PATH = OUTPUT_ROOT / "preview_product_text_area.png"

# Important:
# This area is only for the product text column.
# It does NOT touch price dataset/crops/model files.
X1 = 252
Y1 = 170
X2 = 580
Y2 = 880

ROW_COUNT = 21

# Inner margin to remove table borders as much as possible.
INNER_LEFT = 8
INNER_RIGHT = 8
INNER_TOP = 4
INNER_BOTTOM = 4

# Padding after crop, so OCR does not see text too close to edges.
PAD_X = 18
PAD_Y = 10

UPSCALE = 4


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_padding(img: Image.Image, pad_x=PAD_X, pad_y=PAD_Y, fill="white"):
    new_img = Image.new(
        "RGB",
        (img.width + pad_x * 2, img.height + pad_y * 2),
        fill,
    )
    new_img.paste(img.convert("RGB"), (pad_x, pad_y))
    return new_img


def upscale(img: Image.Image, scale=UPSCALE):
    return img.resize(
        (max(1, img.width * scale), max(1, img.height * scale)),
        RESAMPLE_LANCZOS,
    )


def make_gray_up4(img: Image.Image):
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    out = gray.convert("RGB")
    out = upscale(out)
    return out


def make_sharp_up4(img: Image.Image):
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    out = gray.convert("RGB")
    out = upscale(out)
    out = out.filter(ImageFilter.SHARPEN)
    out = out.filter(ImageFilter.SHARPEN)
    return out


def make_bw_up4(img: Image.Image, threshold=185):
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    bw = gray.point(lambda p: 255 if p >= threshold else 0)
    out = bw.convert("RGB")
    out = upscale(out)
    return out


def save_variant(variant_name: str, row_num: int, img: Image.Image):
    variant_dir = OUTPUT_ROOT / variant_name
    ensure_dir(variant_dir)
    out_path = variant_dir / f"nek_product_row_{row_num:02d}.png"
    img.save(out_path)
    return out_path


def main():
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

    ensure_dir(OUTPUT_ROOT)

    source = Image.open(INPUT_IMAGE).convert("RGB")

    preview = source.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle((X1, Y1, X2, Y2), outline="red", width=3)
    preview.save(PREVIEW_PATH)

    row_height = (Y2 - Y1) / ROW_COUNT

    print(f"Input image: {INPUT_IMAGE}")
    print(f"Image size: {source.width}x{source.height}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Preview: {PREVIEW_PATH}")
    print("")

    for row_num in range(1, ROW_COUNT + 1):
        row_y1 = int(Y1 + (row_num - 1) * row_height)
        row_y2 = int(Y1 + row_num * row_height)

        # Full row crop inside product text column
        raw_crop = source.crop((X1, row_y1, X2, row_y2))

        # Inner crop to remove borders / table lines
        inner_crop = source.crop((
            X1 + INNER_LEFT,
            row_y1 + INNER_TOP,
            X2 - INNER_RIGHT,
            row_y2 - INNER_BOTTOM,
        ))

        padded = add_padding(inner_crop)

        raw_padded = padded
        gray_up4 = make_gray_up4(padded)
        sharp_up4 = make_sharp_up4(padded)
        bw_up4 = make_bw_up4(padded)

        save_variant("raw", row_num, raw_padded)
        save_variant("gray_up4", row_num, gray_up4)
        save_variant("sharp_up4", row_num, sharp_up4)
        save_variant("bw_up4", row_num, bw_up4)

        print(
            f"row {row_num:02d} | "
            f"raw={raw_padded.width}x{raw_padded.height} | "
            f"up4={gray_up4.width}x{gray_up4.height}"
        )

    print("")
    print("Done.")
    print("Open this folder and visually check crops:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
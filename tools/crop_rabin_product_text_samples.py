import argparse
import json
import shutil
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS


def get_layout(w, h):
    """
    Approximate Rabin table layout.
    This is intentionally a review-stage cropper, not the final dynamic extractor.
    """
    if h >= 1500:
        # PDF-like high resolution pages
        return {
            'x1': 0.270,
            'x2': 0.910,
            'y1': 0.112,
            'y2': 0.957,
            'rows': 33,
        }
    else:
        # Telegram/mobile images, usually 1080/1280 height
        return {
            'x1': 0.275,
            'x2': 0.910,
            'y1': 0.136,
            'y2': 0.965,
            'rows': 31,
        }


def pad_and_upscale(img, pad_x=18, pad_y=10, scale=3):
    out = Image.new('RGB', (img.width + pad_x * 2, img.height + pad_y * 2), 'white')
    out.paste(img.convert('RGB'), (pad_x, pad_y))
    out = out.resize((out.width * scale, out.height * scale), RESAMPLE)
    return out


def make_crop_sheet(crop_paths, out_path: Path, cols=5, thumb_w=260):
    if not crop_paths:
        return
    font = ImageFont.load_default()
    cells = []
    for p in crop_paths:
        img = Image.open(p).convert('RGB')
        tw = thumb_w
        th = max(1, int(img.height * (tw / img.width)))
        thumb = img.resize((tw, th), RESAMPLE)
        label = p.stem
        cell = Image.new('RGB', (tw, th + 30), 'white')
        cell.paste(thumb, (0, 0))
        ImageDraw.Draw(cell).text((4, th + 4), label, fill='black', font=font)
        cells.append(cell)
    pad = 12
    rows = (len(cells) + cols - 1) // cols
    cell_h = max(c.height for c in cells)
    sheet = Image.new('RGB', (cols * thumb_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), 'white')
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        sheet.paste(cell, (pad + c * (thumb_w + pad), pad + r * (cell_h + pad)))
    sheet.save(out_path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', default='data/rabin/images')
    parser.add_argument('--output', default='datasets/product_text/rabin_crop_review')
    parser.add_argument('--limit', type=int, default=5, help='How many images to crop for review first')
    parser.add_argument('--all', action='store_true', help='Crop all images')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_root = Path(args.output)
    crops_dir = out_root / 'images'
    previews_dir = out_root / 'previews'
    sheets_dir = out_root / 'sheets'

    if not images_dir.exists():
        raise FileNotFoundError(f'Images directory not found: {images_dir}')

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    crops_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
    if not args.all:
        image_paths = image_paths[:args.limit]

    manifest = []
    sheet_crops = []

    for img_path in image_paths:
        img = ImageOps.exif_transpose(Image.open(img_path)).convert('RGB')
        w, h = img.size
        layout = get_layout(w, h)
        x1 = int(w * layout['x1'])
        x2 = int(w * layout['x2'])
        y1 = int(h * layout['y1'])
        y2 = int(h * layout['y2'])
        rows = int(layout['rows'])
        row_h = (y2 - y1) / rows

        preview = img.copy()
        draw = ImageDraw.Draw(preview)
        draw.rectangle((x1, y1, x2, y2), outline='red', width=max(2, w // 300))
        preview.save(previews_dir / f'{img_path.stem}_preview.jpg', quality=92)

        for row in range(1, rows + 1):
            ry1 = int(y1 + (row - 1) * row_h)
            ry2 = int(y1 + row * row_h)
            crop = img.crop((x1, ry1, x2, ry2))
            crop = pad_and_upscale(crop, scale=3)
            out_name = f'{img_path.stem}_row_{row:03d}.png'
            out_path = crops_dir / out_name
            crop.save(out_path)
            if len(sheet_crops) < 120:
                sheet_crops.append(out_path)
            manifest.append({
                'source_image': img_path.name,
                'row': row,
                'image': str(out_path).replace('\\', '/'),
                'source_size': [w, h],
                'box': [x1, ry1, x2, ry2],
                'needs_label': True,
            })

        print(f'{img_path.name} | {w}x{h} | rows={rows} | box=({x1},{y1},{x2},{y2})')

    (out_root / 'manifest.json').write_text(json.dumps({'count': len(manifest), 'items': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    make_crop_sheet(sheet_crops, sheets_dir / 'crop_review_sheet.jpg')

    print('')
    print(f'Done. Crops: {len(manifest)}')
    print(f'Output: {out_root}')
    print(f'Preview rectangles: {previews_dir}')
    print(f'Review sheet: {sheets_dir / "crop_review_sheet.jpg"}')


if __name__ == '__main__':
    main()

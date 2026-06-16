import argparse
import json
import shutil
import zipfile
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def safe_open_image(path: Path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert('RGB')


def make_contact_sheet(items, out_path: Path, thumb_w=260, pad=18, cols=6):
    thumbs = []
    font = ImageFont.load_default()
    for item in items:
        img = Image.open(item['path']).convert('RGB')
        tw = thumb_w
        th = int(img.height * (tw / img.width))
        img = img.resize((tw, th))
        label = f"{item['id']} | {item['width']}x{item['height']}"
        cell_h = th + 38
        cell = Image.new('RGB', (tw, cell_h), 'white')
        cell.paste(img, (0, 0))
        d = ImageDraw.Draw(cell)
        d.text((4, th + 4), label, fill='black', font=font)
        thumbs.append(cell)

    if not thumbs:
        return

    rows = (len(thumbs) + cols - 1) // cols
    cell_w = thumb_w
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new('RGB', (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), 'white')
    for idx, cell in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        sheet.paste(cell, (x, y))
    sheet.save(out_path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', default='data/raw/Rabin.zip', help='Path to Rabin.zip')
    parser.add_argument('--output', default='data/rabin/images', help='Output images directory')
    parser.add_argument('--seller', default='rabin')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.output)
    base_dir = out_dir.parent
    manifest_path = base_dir / 'manifest.json'
    sheet_path = base_dir / 'contact_sheet.jpg'

    if not zip_path.exists():
        raise FileNotFoundError(f'Zip file not found: {zip_path}')

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = base_dir / '_tmp_unzip'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        infos = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in IMAGE_EXTS]
        for info in infos:
            z.extract(info, tmp_dir)

    manifest = []
    image_files = []
    # Preserve zip order, but resolve extracted file paths safely.
    for idx, info in enumerate(infos, start=1):
        src = tmp_dir / info.filename
        if not src.exists():
            matches = list(tmp_dir.rglob(Path(info.filename).name))
            if not matches:
                print(f'WARNING: could not find extracted file: {info.filename}')
                continue
            src = matches[0]

        img = safe_open_image(src)
        out_name = f'{args.seller}_{idx:03d}.jpg'
        out_path = out_dir / out_name
        img.save(out_path, quality=95)

        item = {
            'id': f'{args.seller}_{idx:03d}',
            'original_name': info.filename,
            'path': str(out_path).replace('\\', '/'),
            'width': img.width,
            'height': img.height,
        }
        manifest.append(item)
        image_files.append(item)
        print(f"{item['id']} | {img.width}x{img.height} | {info.filename}")

    manifest_path.write_text(json.dumps({'seller': args.seller, 'count': len(manifest), 'images': manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    make_contact_sheet(image_files, sheet_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    print('')
    print(f'Done. Images: {len(manifest)}')
    print(f'Manifest: {manifest_path}')
    print(f'Contact sheet: {sheet_path}')


if __name__ == '__main__':
    main()

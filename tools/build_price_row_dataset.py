import argparse
import json
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8-sig"))


def find_price_column(profile):
    for column in profile.get("columns", []):
        if column.get("type") == "price" or column.get("name") == "price":
            return column

    raise ValueError("No price column found in profile.")


def get_truth_rows(truth):
    if isinstance(truth, dict) and isinstance(truth.get("items"), list):
        rows = truth["items"]
    elif isinstance(truth, dict) and isinstance(truth.get("rows"), list):
        rows = truth["rows"]
    elif isinstance(truth, list):
        rows = truth
    else:
        rows = []

    truth_by_row = {}

    for item in rows:
        row = item.get("row")
        price = item.get("price") or item.get("expected_price") or item.get("unit_price")

        if row is None or price is None:
            continue

        truth_by_row[int(row)] = str(int(price))

    return truth_by_row


def crop_price_rows(
    image_path,
    truth_path,
    profile,
    output_images_dir,
    already_price_column=False,
    padding_y=12,
):
    image_path = Path(image_path)
    truth_path = Path(truth_path)

    img = cv2.imread(str(image_path))

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    truth = load_json(truth_path)
    truth_by_row = get_truth_rows(truth)

    h, w = img.shape[:2]

    row_count = int(profile["row_count"])
    table = profile["table"]

    if already_price_column and profile.get("_source_crop_y1") is not None and profile.get("_source_crop_y2") is not None:
        source_y1 = float(profile["_source_crop_y1"])
        source_y2 = float(profile["_source_crop_y2"])
        source_height = source_y2 - source_y1

        table_top_ratio = (float(table["top"]) - source_y1) / source_height
        row_step_ratio = float(table["row_step"]) / source_height
        row_height_ratio = float(table["row_height"]) / source_height
    else:
        table_top_ratio = float(table["top"])
        row_step_ratio = float(table["row_step"])
        row_height_ratio = float(table["row_height"])

    table_top = int(h * table_top_ratio)
    row_step = int(h * row_step_ratio)
    row_height = int(h * row_height_ratio)

    if already_price_column:
        x1 = 0
        x2 = w
    else:
        price_column = find_price_column(profile)
        x1 = int(w * float(price_column["x1"]))
        x2 = int(w * float(price_column["x2"]))

    labels = []
    manifest_items = []

    image_id = image_path.stem

    for row_index in range(row_count):
        row_number = row_index + 1

        if row_number not in truth_by_row:
            continue

        y1 = table_top + (row_index * row_step)
        y2 = y1 + row_height

        y1p = max(0, y1 - padding_y)
        y2p = min(h, y2 + padding_y)

        crop = img[y1p:y2p, x1:x2]

        if crop is None or crop.size == 0:
            continue

        output_name = f"{image_id}_row_{row_number:02d}.png"
        output_path = output_images_dir / output_name

        cv2.imwrite(str(output_path), crop)

        relative_path = f"images/{output_name}"
        label = truth_by_row[row_number]

        labels.append(f"{relative_path}\t{label}")

        manifest_items.append(
            {
                "source_image": str(image_path).replace("\\", "/"),
                "truth_file": str(truth_path).replace("\\", "/"),
                "row": row_number,
                "price": int(label),
                "crop": relative_path,
                "crop_box": [x1, y1p, x2, y2p],
            }
        )

    return labels, manifest_items


def collect_batch_items(images_dir, truth_dir, max_images=None):
    images_dir = Path(images_dir)
    truth_dir = Path(truth_dir)

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    items = []

    for image_path in image_paths:
        truth_path = truth_dir / f"{image_path.stem}.json"

        if not truth_path.exists():
            print(f"Skipping {image_path.name}: truth file not found")
            continue

        items.append((image_path, truth_path))

        if max_images and len(items) >= max_images:
            break

    return items


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--source-crop-y1", type=float, default=None)
    parser.add_argument("--source-crop-y2", type=float, default=None)
    parser.add_argument("--profile", required=True, help="Seller profile JSON path")
    parser.add_argument("--output", required=True, help="Output dataset directory")

    parser.add_argument("--image-path", help="Single image path")
    parser.add_argument("--truth-path", help="Single truth JSON path")

    parser.add_argument("--images-dir", help="Batch images directory")
    parser.add_argument("--truth-dir", help="Batch truth JSON directory")

    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--already-price-column", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    profile = load_json(args.profile)
    profile["_source_crop_y1"] = args.source_crop_y1
    profile["_source_crop_y2"] = args.source_crop_y2

    output_dir = Path(args.output)
    output_images_dir = output_dir / "images"

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Use --overwrite to replace it."
            )

    output_images_dir.mkdir(parents=True, exist_ok=True)

    if args.images_dir and args.truth_dir:
        batch_items = collect_batch_items(
            images_dir=args.images_dir,
            truth_dir=args.truth_dir,
            max_images=args.max_images,
        )
    else:
        image_path = args.image_path or profile.get("image_path")
        truth_path = args.truth_path

        if not image_path:
            raise ValueError("image path is required.")
        if not truth_path:
            raise ValueError("truth path is required.")

        batch_items = [(Path(image_path), Path(truth_path))]

    all_labels = []
    all_manifest_items = []

    for image_path, truth_path in batch_items:
        labels, manifest_items = crop_price_rows(
            image_path=image_path,
            truth_path=truth_path,
            profile=profile,
            output_images_dir=output_images_dir,
            already_price_column=args.already_price_column,
        )

        all_labels.extend(labels)
        all_manifest_items.extend(manifest_items)

        print(f"Processed: {image_path} -> {len(labels)} rows")

    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(all_labels), encoding="utf-8")

    manifest = {
        "profile": str(args.profile).replace("\\", "/"),
        "total_images": len(batch_items),
        "total_rows": len(all_labels),
        "already_price_column": bool(args.already_price_column),
        "items": all_manifest_items,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone")
    print(json.dumps({
        "output": str(output_dir),
        "labels": str(labels_path),
        "manifest": str(manifest_path),
        "total_images": len(batch_items),
        "total_rows": len(all_labels),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
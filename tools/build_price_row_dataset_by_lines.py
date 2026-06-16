
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def group_consecutive_numbers(numbers, max_gap=3):
    groups = []

    if len(numbers) == 0:
        return groups

    start = int(numbers[0])
    previous = int(numbers[0])

    for number in numbers[1:]:
        number = int(number)

        if number <= previous + max_gap:
            previous = number
        else:
            groups.append((start, previous))
            start = number
            previous = number

    groups.append((start, previous))

    return groups


def detect_row_lines(image, row_count):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    kernel_width = max(50, int(width * 0.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    projection = (horizontal_lines > 0).sum(axis=1)

    min_line_pixels = max(20, int(width * 0.18))
    line_y_positions = np.where(projection > min_line_pixels)[0]

    line_groups = group_consecutive_numbers(line_y_positions, max_gap=3)

    candidates = []

    for y1, y2 in line_groups:
        segment = projection[y1:y2 + 1]

        if len(segment) == 0:
            continue

        weights = segment.astype(float)
        positions = np.arange(y1, y2 + 1)

        if weights.sum() > 0:
            center = float(np.average(positions, weights=weights))
        else:
            center = float((y1 + y2) / 2)

        candidates.append(
            {
                "center": center,
                "strength": float(segment.max()),
                "group": [int(y1), int(y2)],
            }
        )

    candidates = sorted(candidates, key=lambda item: item["center"])

    clustered_candidates = []

    for candidate in candidates:
        if (
            clustered_candidates
            and candidate["center"] - clustered_candidates[-1][-1]["center"] < 25
        ):
            clustered_candidates[-1].append(candidate)
        else:
            clustered_candidates.append([candidate])

    raw_lines = []

    for cluster in clustered_candidates:
        best_candidate = max(cluster, key=lambda item: item["strength"])
        raw_lines.append(best_candidate["center"])

    expected_line_count = row_count + 1
    used_fallback = False

    if len(raw_lines) == expected_line_count:
        lines = raw_lines
    elif len(raw_lines) >= 2:
        lines = np.linspace(raw_lines[0], raw_lines[-1], expected_line_count).tolist()
        used_fallback = True
    else:
        raise ValueError(
            f"Cannot detect enough horizontal lines. Detected lines: {len(raw_lines)}"
        )

    return lines, raw_lines, used_fallback


def crop_tight_text(row_image, text_threshold=130, margin_x=14, margin_y=10):
    gray = cv2.cvtColor(row_image, cv2.COLOR_BGR2GRAY)
    mask = gray < text_threshold

    y_positions, x_positions = np.where(mask)

    if len(x_positions) < 10 or len(y_positions) < 10:
        return row_image

    height, width = row_image.shape[:2]

    x1 = max(0, int(x_positions.min()) - margin_x)
    x2 = min(width, int(x_positions.max()) + margin_x + 1)
    y1 = max(0, int(y_positions.min()) - margin_y)
    y2 = min(height, int(y_positions.max()) + margin_y + 1)

    return row_image[y1:y2, x1:x2]


def save_debug_image(image, lines, raw_lines, output_path):
    debug_image = image.copy()

    for y in raw_lines:
        y = int(round(y))
        cv2.line(debug_image, (0, y), (debug_image.shape[1] - 1, y), (0, 0, 255), 2)

    for y in lines:
        y = int(round(y))
        cv2.line(debug_image, (0, y), (debug_image.shape[1] - 1, y), (0, 180, 0), 2)

    cv2.imwrite(str(output_path), debug_image)


def crop_price_rows(
    image_path,
    truth_path,
    output_images_dir,
    output_debug_dir,
    row_count,
    inner_y_margin=5,
    x_margin=0,
    tight_text=False,
    debug=False,
):
    image_path = Path(image_path)
    truth_path = Path(truth_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    truth = load_json(truth_path)
    truth_by_row = get_truth_rows(truth)

    row_count = int(truth.get("row_count") or row_count)

    height, width = image.shape[:2]

    lines, raw_lines, used_fallback = detect_row_lines(image, row_count=row_count)

    if debug:
        output_debug_dir.mkdir(parents=True, exist_ok=True)
        save_debug_image(
            image=image,
            lines=lines,
            raw_lines=raw_lines,
            output_path=output_debug_dir / f"{image_path.stem}_debug.png",
        )

    labels = []
    manifest_items = []

    for row_index in range(row_count):
        row_number = row_index + 1

        if row_number not in truth_by_row:
            continue

        y1 = int(round(lines[row_index])) + inner_y_margin
        y2 = int(round(lines[row_index + 1])) - inner_y_margin

        x1 = x_margin
        x2 = width - x_margin

        y1 = max(0, y1)
        y2 = min(height, y2)
        x1 = max(0, x1)
        x2 = min(width, x2)

        if y2 <= y1 or x2 <= x1:
            continue

        crop = image[y1:y2, x1:x2]

        if tight_text:
            crop = crop_tight_text(crop)

        output_name = f"{image_path.stem}_row_{row_number:02d}.png"
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
                "crop_box": [x1, y1, x2, y2],
            }
        )

    return labels, manifest_items, {
        "detected_raw_lines": len(raw_lines),
        "final_lines": len(lines),
        "used_fallback": used_fallback,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--truth-dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--row-count", type=int, default=21)
    parser.add_argument("--max-images", type=int, default=None)

    parser.add_argument("--inner-y-margin", type=int, default=5)
    parser.add_argument("--x-margin", type=int, default=0)
    parser.add_argument("--tight-text", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_images_dir = output_dir / "images"
    output_debug_dir = output_dir / "debug"

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                f"Use --overwrite to replace it."
            )

    output_images_dir.mkdir(parents=True, exist_ok=True)

    batch_items = collect_batch_items(
        images_dir=args.images_dir,
        truth_dir=args.truth_dir,
        max_images=args.max_images,
    )

    all_labels = []
    all_manifest_items = []
    detection_reports = []

    for image_path, truth_path in batch_items:
        try:
            labels, manifest_items, detection_report = crop_price_rows(
                image_path=image_path,
                truth_path=truth_path,
                output_images_dir=output_images_dir,
                output_debug_dir=output_debug_dir,
                row_count=args.row_count,
                inner_y_margin=args.inner_y_margin,
                x_margin=args.x_margin,
                tight_text=args.tight_text,
                debug=args.debug,
            )
        except Exception as error:
            print(f"Failed: {image_path} -> {error}")
            continue

        all_labels.extend(labels)
        all_manifest_items.extend(manifest_items)

        detection_reports.append(
            {
                "image": str(image_path).replace("\\", "/"),
                **detection_report,
                "rows": len(labels),
            }
        )

        print(
            f"Processed: {image_path} -> {len(labels)} rows "
            f"| raw lines: {detection_report['detected_raw_lines']} "
            f"| fallback: {detection_report['used_fallback']}"
        )

    labels_path = output_dir / "labels.txt"
    labels_path.write_text("\n".join(all_labels), encoding="utf-8")

    manifest = {
        "total_images": len(batch_items),
        "total_rows": len(all_labels),
        "row_count": args.row_count,
        "inner_y_margin": args.inner_y_margin,
        "x_margin": args.x_margin,
        "tight_text": bool(args.tight_text),
        "items": all_manifest_items,
        "detection_reports": detection_reports,
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

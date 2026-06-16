import argparse
import json
import shutil
from pathlib import Path


def read_labels(labels_path):
    rows = []

    for line in Path(labels_path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line:
            continue

        image_path, label = line.split("\t", 1)
        image_path = image_path.replace("\\", "/")
        image_name = Path(image_path).name
        source_id = "_".join(Path(image_name).stem.split("_")[:2])

        rows.append({
            "image_path": image_path,
            "label": label.strip(),
            "image_name": image_name,
            "source_id": source_id,
        })

    return rows


def write_label_file(path, rows, image_prefix="images"):
    lines = []

    for row in rows:
        lines.append(f"{image_prefix}/{row['image_name']}\t{row['label']}")

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-image-ids", default="nek_001,nek_002")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)

    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output already exists: {output_dir}")

    images_out = output_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)

    rows = read_labels(dataset_dir / "labels.txt")

    test_ids = {
        item.strip()
        for item in args.test_image_ids.split(",")
        if item.strip()
    }

    train_rows = []
    test_rows = []

    for row in rows:
        src = dataset_dir / "images" / row["image_name"]
        dst = images_out / row["image_name"]

        if not src.exists():
            raise FileNotFoundError(f"Missing image: {src}")

        shutil.copy2(src, dst)

        if row["source_id"] in test_ids:
            test_rows.append(row)
        else:
            train_rows.append(row)

    write_label_file(output_dir / "train.txt", train_rows)
    write_label_file(output_dir / "test.txt", test_rows)

    report = {
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_image_ids": sorted(test_ids),
        "output": str(output_dir).replace("\\", "/"),
    }

    (output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
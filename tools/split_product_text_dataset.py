from pathlib import Path
import argparse
import shutil
import json


def parse_prefix_from_rel_path(rel_path: str):
    # images/rabin_002_row_01.png -> rabin_002
    name = Path(rel_path).name
    parts = name.split("_row_")
    return parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument(
        "--test-prefixes",
        required=True,
        help="Comma separated prefixes, example: rabin_004",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.dataset)
    out_dir = Path(args.output)
    labels_path = src_dir / args.labels
    src_images_dir = src_dir / "images"

    test_prefixes = set(x.strip() for x in args.test_prefixes.split(",") if x.strip())

    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    if not src_images_dir.exists():
        raise FileNotFoundError(src_images_dir)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)

    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    train_lines = []
    test_lines = []

    all_lines = labels_path.read_text(encoding="utf-8").splitlines()

    copied = 0
    skipped = 0

    for line in all_lines:
        if not line.strip():
            continue

        rel_path, text = line.split("\t", 1)
        rel_path = rel_path.replace("\\", "/").strip()
        text = text.strip()

        src_img = src_dir / rel_path
        dst_img = out_dir / rel_path

        if not src_img.exists():
            print(f"Missing image, skipped: {src_img}")
            skipped += 1
            continue

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
        copied += 1

        prefix = parse_prefix_from_rel_path(rel_path)

        new_line = f"{rel_path}\t{text}"

        if prefix in test_prefixes:
            test_lines.append(new_line)
        else:
            train_lines.append(new_line)

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")

    report = {
        "source_dataset": str(src_dir).replace("\\", "/"),
        "output": str(out_dir).replace("\\", "/"),
        "labels": args.labels,
        "test_prefixes": sorted(test_prefixes),
        "total": len(train_lines) + len(test_lines),
        "train": len(train_lines),
        "test": len(test_lines),
        "copied": copied,
        "skipped": skipped,
    }

    (out_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
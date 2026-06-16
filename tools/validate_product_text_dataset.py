from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset folder path")
    parser.add_argument("--labels", default="labels.txt", help="Label file name")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_path = dataset_dir / args.labels

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    image_files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    image_rel_set = set(f"images/{p.name}" for p in image_files)

    label_lines = labels_path.read_text(encoding="utf-8").splitlines()

    label_rel_list = []
    empty_text = []
    bad_format = []

    for i, line in enumerate(label_lines, start=1):
        if not line.strip():
            continue

        if "\t" not in line:
            bad_format.append((i, line))
            continue

        rel_path, text = line.split("\t", 1)
        rel_path = rel_path.replace("\\", "/").strip()
        text = text.strip()

        label_rel_list.append(rel_path)

        if not text:
            empty_text.append((i, rel_path))

    label_rel_set = set(label_rel_list)

    missing_labels = sorted(image_rel_set - label_rel_set)
    missing_images = sorted(label_rel_set - image_rel_set)
    duplicate_labels = sorted([x for x in label_rel_set if label_rel_list.count(x) > 1])

    print("Dataset:", dataset_dir)
    print("Images:", len(image_files))
    print("Label lines:", len(label_lines))
    print("Valid label paths:", len(label_rel_list))
    print("")

    print("Missing labels:", len(missing_labels))
    for x in missing_labels[:20]:
        print("  image without label:", x)

    print("Missing images:", len(missing_images))
    for x in missing_images[:20]:
        print("  label without image:", x)

    print("Duplicate label paths:", len(duplicate_labels))
    for x in duplicate_labels[:20]:
        print("  duplicate:", x)

    print("Bad format lines:", len(bad_format))
    for line_no, line in bad_format[:20]:
        print(f"  line {line_no}: {line}")

    print("Empty texts:", len(empty_text))
    for line_no, rel_path in empty_text[:20]:
        print(f"  line {line_no}: {rel_path}")

    ok = (
        len(missing_labels) == 0
        and len(missing_images) == 0
        and len(duplicate_labels) == 0
        and len(bad_format) == 0
        and len(empty_text) == 0
    )

    print("")
    if ok:
        print("OK: Dataset is valid.")
    else:
        print("ERROR: Dataset has problems.")


if __name__ == "__main__":
    main()
import argparse
import json
import os
import re
from pathlib import Path

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR


def normalize_digits(text):
    if text is None:
        return ""

    text = str(text)

    table = str.maketrans({
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "،": ",",
        "٫": ".",
        "٬": ",",
    })

    return text.translate(table).strip()


def parse_price(text):
    text = normalize_digits(text)
    digits = re.sub(r"[^\d]", "", text)

    if not digits:
        return None

    return int(digits)


def load_labels(labels_path):
    labels = []

    for line in Path(labels_path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()

        if not line:
            continue

        if "\t" in line:
            image_path, label = line.split("\t", 1)
        else:
            parts = line.split()
            image_path = parts[0]
            label = parts[-1]

        labels.append({
            "image_path": image_path.replace("\\", "/"),
            "expected_price": int(label.strip()),
        })

    return labels


def get_source_id(relative_image_path):
    stem = Path(relative_image_path).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"

    return stem


def should_include(relative_image_path, image_ids):
    if not image_ids:
        return True

    return get_source_id(relative_image_path) in image_ids


def make_ocr(lang):
    try:
        return PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        return PaddleOCR(lang=lang)


def collect_texts_from_object(obj, texts):
    if obj is None:
        return

    if hasattr(obj, "json"):
        collect_texts_from_object(obj.json, texts)
        return

    if isinstance(obj, dict):
        data = obj.get("res", obj)

        rec_texts = data.get("rec_texts")
        rec_scores = data.get("rec_scores", [])

        if isinstance(rec_texts, list):
            for index, text in enumerate(rec_texts):
                score = 0.0

                if isinstance(rec_scores, list) and index < len(rec_scores):
                    try:
                        score = float(rec_scores[index])
                    except Exception:
                        score = 0.0

                texts.append({
                    "text": str(text),
                    "score": score,
                })

        for value in obj.values():
            collect_texts_from_object(value, texts)

        return

    if isinstance(obj, (list, tuple)):
        if len(obj) >= 2:
            first = obj[0]
            second = obj[1]

            if isinstance(first, str) and isinstance(second, (int, float)):
                texts.append({
                    "text": first,
                    "score": float(second),
                })
                return

            if isinstance(second, (list, tuple)) and len(second) >= 2:
                if isinstance(second[0], str) and isinstance(second[1], (int, float)):
                    texts.append({
                        "text": second[0],
                        "score": float(second[1]),
                    })
                    return

        for item in obj:
            collect_texts_from_object(item, texts)


def run_ocr(ocr, image_path):
    if hasattr(ocr, "predict"):
        result = ocr.predict(str(image_path))
    else:
        result = ocr.ocr(str(image_path), cls=False)

    texts = []
    collect_texts_from_object(result, texts)

    return texts


def choose_best_price(texts):
    candidates = []

    for item in texts:
        raw_text = item.get("text", "")
        score = float(item.get("score", 0) or 0)

        price = parse_price(raw_text)

        if price is None:
            continue

        digits_len = len(str(price))

        candidates.append({
            "raw": raw_text,
            "price": price,
            "score": score,
            "rank": (score * 10) + digits_len,
        })

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["rank"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="fa")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    labels_path = dataset_dir / args.labels

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    image_ids = {
        item.strip()
        for item in args.image_ids.split(",")
        if item.strip()
    }

    all_rows = load_labels(labels_path)

    selected_rows = [
        row for row in all_rows
        if should_include(row["image_path"], image_ids)
    ]

    print(f"Selected row images: {len(selected_rows)}")

    ocr = make_ocr(args.lang)

    result_rows = []

    correct_count = 0
    wrong_count = 0
    missing_count = 0

    for row in selected_rows:
        relative_image_path = row["image_path"]
        expected_price = row["expected_price"]

        image_path = dataset_dir / relative_image_path

        if not image_path.exists():
            status = "missing_image"
            actual_price = None
            raw = None
            score = 0.0
            missing_count += 1
        else:
            try:
                texts = run_ocr(ocr, image_path)
                best = choose_best_price(texts)

                if best is None:
                    status = "missing"
                    actual_price = None
                    raw = None
                    score = 0.0
                    missing_count += 1
                else:
                    actual_price = best["price"]
                    raw = best["raw"]
                    score = best["score"]

                    if actual_price == expected_price:
                        status = "correct"
                        correct_count += 1
                    else:
                        status = "wrong"
                        wrong_count += 1

            except Exception as error:
                status = "ocr_error"
                actual_price = None
                raw = str(error)
                score = 0.0
                missing_count += 1

        result_rows.append({
            "image": relative_image_path,
            "expected_price": expected_price,
            "actual_price": actual_price,
            "status": status,
            "raw": raw,
            "score": score,
        })

        print(
            f"{relative_image_path} | expected={expected_price} | actual={actual_price} | {status}"
        )

    total_rows = len(result_rows)
    accuracy_percent = round((correct_count / total_rows) * 100, 2) if total_rows else 0

    report = {
        "dataset": str(dataset_dir).replace("\\", "/"),
        "labels": args.labels,
        "image_ids": sorted(image_ids),
        "total_rows": total_rows,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "missing_count": missing_count,
        "accuracy_percent": accuracy_percent,
        "rows": result_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps({
        "total_rows": total_rows,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "missing_count": missing_count,
        "accuracy_percent": accuracy_percent,
        "output": str(output_path).replace("\\", "/"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()